"""FastAPI application for the loopback-only Nexus Gateway."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from adapters.diffusers import QUALITY_PROFILES, resolve_image_generation_params
from adapters.tool_calls import create_tool_parser
from api.middleware.auth import make_auth_dependency
from core.hardware_detector import detect_hardware
from core.model_registry import ModelRegistry
from core.model_staging import stage_model_directory
from core.resource_manager import ResourceManager
from core.worker_manager import WorkerError, WorkerManager
from services.conversation import normalize_openai_messages, resolve_reasoning_controls
from services.image_generation import save_generated_image_artifact
from services.progress import ProgressService
from workers.worker_protocol import WorkerGenerateRequest, WorkerImageRequest, WorkerLoadRequest


@dataclass
class GatewayConfig:
    api_key: str
    models_root: Optional[Path] = None
    adapters_root: Optional[Path] = None
    upscalers_root: Optional[Path] = None
    capabilities_sidecar: Optional[Path] = None
    local_stage_root: Path = Path("/tmp/nexus-staged-models")
    output_image_dir: Path = Path("/tmp/nexus-generated-images")
    plugin_index_path: Optional[Path] = None
    min_stage_free_margin_bytes: int = 2 * 1024 * 1024 * 1024


def create_app(
    config: GatewayConfig,
    *,
    registry: Optional[ModelRegistry] = None,
    progress: Optional[ProgressService] = None,
    resources: Optional[ResourceManager] = None,
    worker: Optional[WorkerManager] = None,
) -> FastAPI:
    app = FastAPI(title="Nexus Gateway", version="1.0.0")
    require_auth = make_auth_dependency(config.api_key)

    reg = registry or ModelRegistry(
        config.models_root,
        adapters_root=config.adapters_root,
        upscalers_root=config.upscalers_root,
        capabilities_sidecar=config.capabilities_sidecar,
    )
    reg.scan()
    prog = progress or ProgressService()
    res_mgr = resources or ResourceManager()
    worker_mgr = worker or WorkerManager()

    app.state.config = config
    app.state.registry = reg
    app.state.progress = prog
    app.state.resources = res_mgr
    app.state.worker = worker_mgr
    app.state.started_at = time.time()

    def _stage_and_load(kind: str, model_id: str) -> tuple[Any, str]:
        model = reg.get(model_id)
        if model is None or model.validation_status != "compatible":
            raise HTTPException(status_code=404, detail=f"Unknown or incompatible model: {model_id}")
        if kind == "chat" and not model.is_chat_model:
            raise HTTPException(status_code=400, detail=f"Model {model_id} is not a chat model")
        if kind == "diffusion" and not model.is_image_model:
            raise HTTPException(status_code=400, detail=f"Model {model_id} is not an image-generation model")

        op_kind = "chat_model_load" if kind == "chat" else "diffusion_pipeline_load"
        op = prog.start(op_kind, model_id, phase="staging", message=f"Preparing {model_id}")
        try:
            source_path = Path(model.source_path)
            staged_path = stage_model_directory(
                source_path,
                config.local_stage_root,
                model_id=model_id,
                progress=prog,
                operation_id=op.operation_id,
                min_free_margin_bytes=config.min_stage_free_margin_bytes,
            )
            prog.update(op.operation_id, phase="weight_loading", indeterminate=True, message=f"Loading {model_id} in isolated worker")
            worker_mgr.load_model(
                WorkerLoadRequest(
                    kind=kind,  # type: ignore[arg-type]
                    model_id=model_id,
                    model_path=str(staged_path),
                    architecture=model.architecture,
                    loader_family=model.loader_family,
                )
            )
            prog.complete(op.operation_id, phase="ready", message=f"{model_id} ready")
            return model, op.operation_id
        except WorkerError as exc:
            prog.fail(op.operation_id, str(exc))
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except Exception as exc:
            prog.fail(op.operation_id, str(exc))
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.get("/health")
    def health() -> dict[str, Any]:
        ws = worker_mgr.status()
        return {
            "status": "ok",
            "version": "1.0.0",
            "uptime_s": round(time.time() - app.state.started_at, 2),
            "worker": ws.to_dict(),
            "chat_models": len(reg.chat_models()),
            "image_models": len(reg.image_models()),
        }

    @app.get("/v1/models")
    def list_models(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [m.to_openai_model() for m in reg.chat_models()],
        }

    @app.get("/v1/image-models")
    @app.get("/runtime/image-models")
    def list_image_models(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        return {
            "object": "list",
            "quality_profiles": list(QUALITY_PROFILES.keys()),
            "data": [m.to_openai_model() for m in reg.image_models()],
        }

    @app.get("/v1/runtime")
    def runtime_status(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        hw = detect_hardware(config.local_stage_root if config.local_stage_root.exists() else None)
        return {
            "hardware": hw.to_dict(),
            "worker": worker_mgr.status().to_dict(),
            "busy": res_mgr.is_busy,
            "active_model": res_mgr.active_model_id,
            "active_kind": res_mgr.active_kind,
            "adapters": reg.adapters(),
            "upscalers": reg.upscalers(),
            "diagnostics": reg.all_diagnostics(),
        }

    @app.post("/v1/runtime/load")
    def runtime_load(body: dict[str, Any], _auth: None = Depends(require_auth)) -> dict[str, Any]:
        model_id = str(body.get("model") or "").strip()
        kind = str(body.get("kind") or "chat")
        if not model_id:
            raise HTTPException(status_code=400, detail="model is required")
        with res_mgr.acquire(kind, model_id, lambda: _stage_and_load(kind, model_id), unload=lambda _: worker_mgr.unload()):  # type: ignore[arg-type]
            return {"status": "loaded", "model": model_id, "kind": kind, "worker": worker_mgr.status().to_dict()}

    @app.post("/v1/runtime/unload")
    def runtime_unload(body: Optional[dict[str, Any]] = None, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        force_terminate = bool((body or {}).get("force_terminate", False))
        res_mgr.unload_active()
        if force_terminate:
            worker_mgr.terminate()
            return {"status": "terminated", "worker": worker_mgr.status().to_dict()}
        result = worker_mgr.unload(terminate_on_failure=True)
        return {"status": result.get("status", "unloaded"), "worker": worker_mgr.status().to_dict()}

    @app.get("/v1/progress")
    @app.get("/runtime/progress")
    def get_progress(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        snap = prog.snapshot()
        snap["worker"] = worker_mgr.status().to_dict()
        return snap

    @app.get("/v1/progress/{operation_id}")
    def get_operation_progress(operation_id: str, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        record = prog.get(operation_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown operation_id: {operation_id}")
        return record

    @app.post("/v1/tools/parse")
    def parse_tools_endpoint(body: dict[str, Any], _auth: None = Depends(require_auth)) -> dict[str, Any]:
        family = str(body.get("parser_family") or "qwen")
        text = str(body.get("text") or "")
        tools = body.get("tools") or []
        tool_choice = body.get("tool_choice", "auto")
        parser = create_tool_parser(family)
        parser.feed(text)
        result = parser.finish(tools, tool_choice=tool_choice)
        return {
            "text": result.text,
            "tool_calls": [tc.to_openai_dict() for tc in result.tool_calls],
            "errors": result.errors,
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict[str, Any], _auth: None = Depends(require_auth)) -> Any:
        model_id = str(body.get("model") or "").strip()
        if not model_id:
            raise HTTPException(status_code=400, detail="model is required")
        model = reg.get(model_id)
        if model is None or not model.is_chat_model or model.validation_status != "compatible":
            raise HTTPException(status_code=404, detail=f"Unknown or unsupported chat model: {model_id}")

        messages = normalize_openai_messages(body.get("messages") or [])
        tools = body.get("tools") or []
        tool_choice = body.get("tool_choice", "auto")
        if tool_choice == "none":
            tools = []
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or model.max_output_tokens or 4096)
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.9))

        requested_effort = body.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort") if isinstance(body.get("reasoning"), dict) else body.get("reasoning_effort")
        wire_effort, enable_thinking = resolve_reasoning_controls(
            model,
            requested_effort=str(requested_effort) if requested_effort is not None else None,
            requested_enable_thinking=body.get("enable_thinking"),
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        gen_req = WorkerGenerateRequest(
            model_id=model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=wire_effort,
            enable_thinking=enable_thinking,
            stream=stream,
        )

        if not stream:
            try:
                with res_mgr.acquire("chat", model_id, lambda: _stage_and_load("chat", model_id), unload=lambda _: worker_mgr.unload()):
                    raw = worker_mgr.generate(gen_req)
            except WorkerError as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
            parser = create_tool_parser(model.tool_parser_family)
            parser.feed(str(raw.get("text") or ""))
            parsed = parser.finish(tools, tool_choice=tool_choice)
            message: dict[str, Any] = {"role": "assistant", "content": parsed.text}
            finish_reason = "stop"
            if parsed.tool_calls:
                message["tool_calls"] = [tc.to_openai_dict() for tc in parsed.tool_calls]
                finish_reason = "tool_calls"
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            }

        def sse_events():
            yield "data: " + json.dumps({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }) + "\n\n"
            parser = create_tool_parser(model.tool_parser_family)
            try:
                with res_mgr.acquire("chat", model_id, lambda: _stage_and_load("chat", model_id), unload=lambda _: worker_mgr.unload()):
                    for chunk in worker_mgr.generate_stream(gen_req):
                        visible = parser.feed(chunk)
                        if visible:
                            yield "data: " + json.dumps({
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {"content": visible}, "finish_reason": None}],
                            }, ensure_ascii=False) + "\n\n"
                parsed = parser.finish(tools, tool_choice=tool_choice)
                finish_reason = "tool_calls" if parsed.tool_calls else "stop"
                for idx, tc in enumerate(parsed.tool_calls):
                    yield "data: " + json.dumps({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [{"index": idx, **tc.to_openai_dict()}]},
                            "finish_reason": None,
                        }],
                    }, ensure_ascii=False) + "\n\n"
                yield "data: " + json.dumps({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }) + "\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield "data: " + json.dumps({
                    "error": {"message": f"{type(exc).__name__}: {exc}", "type": "nexus_inference_error"}
                }) + "\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @app.post("/v1/images/generations")
    def image_generations(body: dict[str, Any], _auth: None = Depends(require_auth)) -> dict[str, Any]:
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        model_id = str(body.get("model") or "").strip()
        if not model_id:
            candidates = reg.image_models()
            if not candidates:
                raise HTTPException(status_code=503, detail="No compatible local Diffusers image model discovered")
            model_id = candidates[0].id
        model = reg.get(model_id)
        if model is None or not model.is_image_model or model.validation_status != "compatible":
            raise HTTPException(status_code=404, detail=f"Unknown or incompatible image model: {model_id}")

        quality = str(body.get("quality") or "detailed")
        try:
            params = resolve_image_generation_params(
                model_id=model_id,
                architecture=model.architecture,
                quality=quality,  # type: ignore[arg-type]
                width=body.get("width"),
                height=body.get("height"),
                steps=body.get("steps"),
                num_images=int(body.get("num_images") or body.get("n") or 1),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        op = prog.start("image_generation", model_id, phase="denoising", message=f"Generating image with {model_id}")
        prog.update(op.operation_id, steps_total=params["steps"], step=0)
        try:
            with res_mgr.acquire("diffusion", model_id, lambda: _stage_and_load("diffusion", model_id), unload=lambda _: worker_mgr.unload()):
                raw = worker_mgr.generate_image(
                    WorkerImageRequest(
                        model_id=model_id,
                        prompt=prompt,
                        quality=params["quality"],
                        width=params["width"],
                        height=params["height"],
                        steps=params["steps"],
                        seed=body.get("seed"),
                        num_images=params["num_images"],
                        output_dir=str(config.output_image_dir),
                    )
                )
            prog.update(op.operation_id, phase="saving", step=params["steps"], steps_total=params["steps"], message="Saving PNG to Drive")
            saved = save_generated_image_artifact(
                output_dir=config.output_image_dir,
                b64_png=str(raw["b64_png"]),
                model_id=model_id,
                prompt=prompt,
                quality=params["quality"],
                width=int(raw.get("width") or params["width"]),
                height=int(raw.get("height") or params["height"]),
                steps=int(raw.get("steps") or params["steps"]),
                seed=raw.get("seed"),
                operation_id=op.operation_id,
                workspace=body.get("workspace"),
            )
            prog.complete(op.operation_id, phase="ready", message=f"Saved {saved['filename']}")
            return saved
        except WorkerError as exc:
            prog.fail(op.operation_id, str(exc))
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except Exception as exc:
            prog.fail(op.operation_id, str(exc))
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.get("/v1/plugin-index.json")
    def plugin_index() -> JSONResponse:
        """Serve the curated Nexus plugin index for Harness Discover (loopback-only)."""
        path = config.plugin_index_path
        if path is not None and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return JSONResponse(content=data, headers={"Cache-Control": "no-store"})
        return JSONResponse(
            content={"format": 1, "title": {"en": "Nexus Curated Plugins"}, "metadata": {}, "plugins": []},
            headers={"Cache-Control": "no-store"},
        )

    return app
