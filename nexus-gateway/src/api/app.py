"""FastAPI application for the single-model loopback Nexus Gateway (v1.1.0)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from adapters.tool_calls import create_tool_parser, has_native_tool_parser
from api.middleware.auth import make_auth_dependency
from core.hardware_detector import detect_hardware
from core.memory_planner import can_coreside_image_model
from core.resource_manager import ResourceManager
from core.worker_manager import WorkerError, WorkerManager
from models.base import SessionRuntimeConfig
from services.conversation import normalize_openai_messages, resolve_reasoning_controls
from services.image_generation import save_generated_image_artifact
from services.progress import ProgressService
from workers.worker_protocol import WorkerGenerateRequest, WorkerImageRequest, WorkerLoadRequest, WorkerStatus

GATEWAY_VERSION = "2.1.0"
IMAGE_QUALITY_PROFILES = ("normal", "detailed", "maximum")


@dataclass
class GatewayConfig:
    api_key: str
    session: SessionRuntimeConfig
    output_image_dir: Path = Path("/tmp/nexus-generated-images")
    plugin_index_path: Optional[Path] = None


def create_app(
    config: GatewayConfig,
    *,
    progress: Optional[ProgressService] = None,
    resources: Optional[ResourceManager] = None,
    worker: Optional[WorkerManager] = None,
) -> FastAPI:
    app = FastAPI(title="Nexus Gateway", version=GATEWAY_VERSION)
    require_auth = make_auth_dependency(config.api_key)

    lang_spec = config.session.language_model
    img_spec = config.session.image_model
    prog = progress or ProgressService()
    res_mgr = resources or ResourceManager()
    worker_mgr = worker or WorkerManager()

    app.state.config = config
    app.state.progress = prog
    app.state.resources = res_mgr
    app.state.worker = worker_mgr
    app.state.started_at = time.time()
    app.state.bootstrapped = False

    def _live_readiness(ws: WorkerStatus) -> tuple[bool, str]:
        if res_mgr.is_swapping:
            return False, "swapping"
        if ws.alive and (
            ws.active_language_id == lang_spec.id
            or (ws.active_kind == "language" and ws.active_model_id == lang_spec.id)
        ):
            return True, "ready"
        if ws.last_error:
            return False, "degraded"
        return False, "unloaded"

    def _language_runtime_options() -> dict[str, Any]:
        opts: dict[str, Any] = {}
        for key in ("n_ctx", "n_gpu_layers", "batch_size", "ubatch_size", "threads"):
            val = getattr(lang_spec, key, None)
            if val is not None:
                opts[key] = int(val)
        return opts

    def _load_selected_language_unlocked(op_id: str, *, keep_other_slot: bool = False) -> WorkerStatus:
        prog.update(
            op_id,
            phase="weight_loading",
            indeterminate=True,
            message=f"Loading {lang_spec.id} ({lang_spec.implementation}) in isolated GPU worker",
        )
        worker_mgr.load_model(
            WorkerLoadRequest(
                kind="language",
                model_id=lang_spec.id,
                model_path=lang_spec.model_path,
                implementation=lang_spec.implementation,
                architecture=lang_spec.architecture,
                keep_other_slot=keep_other_slot,
                runtime_options=_language_runtime_options(),
            )
        )
        prog.update(op_id, phase="placement", indeterminate=True, message=f"Verifying worker readiness for {lang_spec.id}")
        ws = worker_mgr.status()
        if not ws.alive or (ws.active_language_id != lang_spec.id and ws.active_model_id != lang_spec.id):
            raise WorkerError(
                f"Worker failed to confirm active language model {lang_spec.id} "
                f"(alive={ws.alive}, kind={ws.active_kind}, active={ws.active_model_id})",
                status_code=502,
            )
        return ws

    @app.get("/health")
    def health() -> dict[str, Any]:
        ws = worker_mgr.status()
        ready, state = _live_readiness(ws)
        return {
            "status": "ok",
            "version": GATEWAY_VERSION,
            "ready": ready,
            "state": state,
            "uptime_s": round(time.time() - app.state.started_at, 2),
            "selected_language_model": lang_spec.id,
            "selected_image_model": img_spec.id if img_spec is not None else None,
            "worker": ws.to_dict(),
        }

    @app.get("/v1/models")
    def list_models(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [lang_spec.to_openai_model()],
        }

    @app.get("/v1/runtime")
    def runtime_status(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        staged_root = Path(lang_spec.model_path).parent
        hw = detect_hardware(staged_root if staged_root.exists() else None)
        ws = worker_mgr.status()
        ready, state = _live_readiness(ws)
        return {
            "version": GATEWAY_VERSION,
            "ready": ready,
            "state": state,
            "busy": res_mgr.is_busy,
            "swapping": res_mgr.is_swapping,
            "hardware": hw.to_dict(),
            "worker": ws.to_dict(),
            "language_model": lang_spec.to_openai_model(),
            "image_model": img_spec.to_dict() if img_spec is not None else None,
            "quality_profiles": list(IMAGE_QUALITY_PROFILES),
        }

    @app.post("/v1/runtime/bootstrap")
    def runtime_bootstrap(body: Optional[dict[str, Any]] = None, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        """
        Colab-owned warm-load endpoint for the single configured language model.
        Rejects any attempt to load a different model ID (single-model lock).
        """
        payload = body or {}
        requested_model = str(payload.get("model") or lang_spec.id).strip()
        replace = bool(payload.get("replace", False))
        if requested_model != lang_spec.id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Single-model runtime is locked to '{lang_spec.id}'; cannot load '{requested_model}'. "
                    "Update SELECTED_LANGUAGE_MODEL in Colab and restart the runtime."
                ),
            )

        with res_mgr.exclusive():
            ws = worker_mgr.status()
            if ws.alive and ws.active_kind == "language" and ws.active_model_id == lang_spec.id and not replace:
                return {
                    "status": "ready",
                    "reused": True,
                    "model": lang_spec.id,
                    "worker": ws.to_dict(),
                }
            op = prog.start(
                "language_model_load",
                lang_spec.id,
                phase="resolving",
                indeterminate=True,
                message=f"Bootstrapping {lang_spec.id}",
            )
            try:
                if ws.alive and (replace or ws.active_model_id is not None):
                    worker_mgr.unload(terminate_on_failure=True)
                live_ws = _load_selected_language_unlocked(op.operation_id)
                app.state.bootstrapped = True
                prog.complete(op.operation_id, phase="ready", message=f"{lang_spec.id} ready")
                return {
                    "status": "ready",
                    "reused": False,
                    "operation_id": op.operation_id,
                    "model": lang_spec.id,
                    "worker": live_ws.to_dict(),
                }
            except WorkerError as exc:
                prog.fail(op.operation_id, str(exc))
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
            except Exception as exc:
                prog.fail(op.operation_id, f"{type(exc).__name__}: {exc}")
                raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/v1/runtime/preload-image")
    def runtime_preload_image(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        """Co-load the selected image pipeline alongside the LLM if live free VRAM suffices."""
        if img_spec is None:
            return {"status": "skipped", "reason": "no_image_model"}
        if not lang_spec.can_coreside_image:
            return {"status": "skipped", "reason": "coresidency_disabled"}
        img_dir = Path(img_spec.model_path)
        stage_lock = img_dir.parent / f".{img_spec.id}.staging.lock"
        if stage_lock.is_file() or not img_dir.is_dir():
            return {"status": "skipped", "reason": "staging_incomplete"}
        with res_mgr.exclusive():
            ws = worker_mgr.status()
            if ws.active_image_id == img_spec.id:
                return {"status": "ready", "co_resident": ws.co_resident, "worker": ws.to_dict()}
            hw = detect_hardware(Path(lang_spec.model_path).parent)
            free_vram = ws.device_free_vram_bytes or hw.free_vram_bytes
            if not can_coreside_image_model(free_vram_bytes=free_vram, image_min_vram_gib=img_spec.min_vram_gib):
                return {"status": "skipped", "reason": "insufficient_free_vram", "free_vram_bytes": free_vram}
            try:
                worker_mgr.load_model(
                    WorkerLoadRequest(
                        kind="image",
                        model_id=img_spec.id,
                        model_path=img_spec.model_path,
                        implementation=img_spec.implementation,
                        architecture=img_spec.architecture,
                        keep_other_slot=True,
                    )
                )
                return {"status": "ready", "co_resident": True, "worker": worker_mgr.status().to_dict()}
            except Exception as exc:
                worker_mgr.unload(kind="image", terminate_on_failure=False)
                return {"status": "skipped", "reason": f"preload_failed: {exc}"}

    @app.post("/v1/runtime/unload")
    def runtime_unload(body: Optional[dict[str, Any]] = None, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        force_terminate = bool((body or {}).get("force_terminate", False))
        with res_mgr.exclusive():
            if force_terminate:
                worker_mgr.terminate()
                return {"status": "terminated", "worker": worker_mgr.status().to_dict()}
            result = worker_mgr.unload(terminate_on_failure=True)
            return {"status": result.get("status", "unloaded"), "worker": worker_mgr.status().to_dict()}

    @app.post("/v1/runtime/cancel")
    def runtime_cancel(body: Optional[dict[str, Any]] = None, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        op_id = (body or {}).get("operation_id")
        res = worker_mgr.cancel_operation(str(op_id) if op_id else None)
        for cancelled_id in res.get("operations") or []:
            prog.update(cancelled_id, status="cancelled", indeterminate=False, message="Cancelled by user")
        return res

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

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict[str, Any], _auth: None = Depends(require_auth)) -> Any:
        model_id = str(body.get("model") or "").strip()
        if not model_id:
            raise HTTPException(status_code=400, detail="model is required")
        if model_id != lang_spec.id:
            raise HTTPException(
                status_code=400,
                detail=f"This runtime is locked to language model '{lang_spec.id}'; received '{model_id}'",
            )

        ws = worker_mgr.status()
        if not ws.alive or ws.active_kind != "language" or ws.active_model_id != lang_spec.id:
            if not app.state.bootstrapped:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Language model '{lang_spec.id}' is not loaded in the GPU worker "
                        f"(alive={ws.alive}, active={ws.active_model_id}). Run the Colab bootstrap cell."
                    ),
                )
            # Previously bootstrapped: wait for any in-flight GPU swap to unwind and auto-restore if needed.
            with res_mgr.exclusive():
                ws = worker_mgr.status()
                if not ws.alive or ws.active_kind != "language" or ws.active_model_id != lang_spec.id:
                    rec_op = prog.start(
                        "language_model_load",
                        lang_spec.id,
                        phase="restoring_language_model",
                        indeterminate=True,
                        message=f"Recovering {lang_spec.id} after previous failure",
                    )
                    try:
                        worker_mgr.unload(terminate_on_failure=True)
                        _load_selected_language_unlocked(rec_op.operation_id)
                        prog.complete(rec_op.operation_id, phase="ready", message=f"{lang_spec.id} recovered")
                    except Exception as exc:
                        prog.fail(rec_op.operation_id, str(exc))
                        raise HTTPException(
                            status_code=503,
                            detail=f"Failed to recover language model '{lang_spec.id}': {exc}",
                        ) from exc

        messages = normalize_openai_messages(body.get("messages") or [])
        if not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        tools = body.get("tools") or []
        tool_choice = body.get("tool_choice", "auto")
        if tool_choice not in ("auto", "none") and not isinstance(tool_choice, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported tool_choice {tool_choice!r}; supported: 'auto', 'none', or function selector",
            )
        if isinstance(tool_choice, dict):
            selected_name = (tool_choice.get("function") or {}).get("name")
            if not isinstance(selected_name, str) or not selected_name:
                raise HTTPException(status_code=400, detail="tool_choice object must specify function.name")
            tools = [t for t in tools if (t.get("function") or t).get("name") == selected_name]
            if not tools:
                raise HTTPException(status_code=400, detail=f"tool_choice requested unadvertised tool {selected_name!r}")
        if tool_choice == "none":
            tools = []
        if tools and not has_native_tool_parser(lang_spec.implementation):
            raise HTTPException(
                status_code=422,
                detail=f"Language model '{lang_spec.id}' ({lang_spec.implementation}) has no native tool parser",
            )

        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or lang_spec.max_output_tokens or 4096)
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.9))

        requested_effort = (
            (body.get("reasoning") or {}).get("effort")
            if isinstance(body.get("reasoning"), dict)
            else body.get("reasoning_effort")
        )
        wire_effort, enable_thinking = resolve_reasoning_controls(
            lang_spec,
            requested_effort=str(requested_effort) if requested_effort is not None else None,
            requested_enable_thinking=body.get("enable_thinking"),
        )
        started_in_think = lang_spec.implementation == "qwen3_5" and enable_thinking is not False

        request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        gen_req = WorkerGenerateRequest(
            model_id=lang_spec.id,
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
            gen_op = prog.start(
                "chat_generation",
                lang_spec.id,
                phase="generating",
                indeterminate=True,
                message=f"Generating response with {lang_spec.id}",
            )
            try:
                with res_mgr.exclusive():
                    live = worker_mgr.status()
                    if not live.alive or live.active_kind != "language" or live.active_model_id != lang_spec.id:
                        raise WorkerError(f"Language model '{lang_spec.id}' is not loaded in worker", status_code=503)
                    raw = worker_mgr.generate(gen_req)
                prog.complete(gen_op.operation_id, phase="ready", message="Response complete")
            except WorkerError as exc:
                prog.fail(gen_op.operation_id, str(exc))
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
            except Exception as exc:
                prog.fail(gen_op.operation_id, str(exc))
                raise

            parser = create_tool_parser(lang_spec.implementation, started_in_think=started_in_think)
            parser.feed(str(raw.get("text") or ""))
            parsed = parser.finish(tools, tool_choice=tool_choice)
            if parsed.errors:
                raise HTTPException(
                    status_code=422,
                    detail={"error": {"message": "; ".join(parsed.errors), "type": "tool_call_parse_error"}},
                )
            message: dict[str, Any] = {"role": "assistant", "content": parsed.text}
            if parsed.reasoning:
                message["reasoning_content"] = parsed.reasoning
            finish_reason = "stop"
            if parsed.tool_calls:
                message["tool_calls"] = [tc.to_openai_dict() for tc in parsed.tool_calls]
                finish_reason = "tool_calls"
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": lang_spec.id,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            }

        def sse_events():
            yield "data: " + json.dumps({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": lang_spec.id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }) + "\n\n"
            parser = create_tool_parser(lang_spec.implementation, started_in_think=started_in_think)
            gen_op = prog.start(
                "chat_generation",
                lang_spec.id,
                phase="generating",
                indeterminate=True,
                message=f"Generating response with {lang_spec.id}",
            )
            try:
                with res_mgr.exclusive():
                    live = worker_mgr.status()
                    if not live.alive or live.active_kind != "language" or live.active_model_id != lang_spec.id:
                        raise WorkerError(f"Language model '{lang_spec.id}' is not loaded in worker", status_code=503)
                    for chunk in worker_mgr.generate_stream(gen_req):
                        delta = parser.feed(chunk)
                        if delta.reasoning:
                            yield "data: " + json.dumps({
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": lang_spec.id,
                                "choices": [{"index": 0, "delta": {"reasoning_content": delta.reasoning}, "finish_reason": None}],
                            }, ensure_ascii=False) + "\n\n"
                        if delta.content:
                            yield "data: " + json.dumps({
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": lang_spec.id,
                                "choices": [{"index": 0, "delta": {"content": delta.content}, "finish_reason": None}],
                            }, ensure_ascii=False) + "\n\n"

                parsed = parser.finish(tools, tool_choice=tool_choice)
                if parsed.trailing_reasoning:
                    yield "data: " + json.dumps({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": lang_spec.id,
                        "choices": [{"index": 0, "delta": {"reasoning_content": parsed.trailing_reasoning}, "finish_reason": None}],
                    }, ensure_ascii=False) + "\n\n"
                if parsed.trailing_text:
                    yield "data: " + json.dumps({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": lang_spec.id,
                        "choices": [{"index": 0, "delta": {"content": parsed.trailing_text}, "finish_reason": None}],
                    }, ensure_ascii=False) + "\n\n"
                if parsed.errors:
                    prog.fail(gen_op.operation_id, "; ".join(parsed.errors))
                    yield "data: " + json.dumps({
                        "error": {"message": "; ".join(parsed.errors), "type": "tool_call_parse_error"}
                    }) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
                finish_reason = "tool_calls" if parsed.tool_calls else "stop"
                for idx, tc in enumerate(parsed.tool_calls):
                    yield "data: " + json.dumps({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": lang_spec.id,
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [{"index": idx, **tc.to_openai_dict()}]},
                            "finish_reason": None,
                        }],
                    }, ensure_ascii=False) + "\n\n"
                prog.complete(gen_op.operation_id, phase="ready", message="Response complete")
                yield "data: " + json.dumps({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": lang_spec.id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }) + "\n\n"
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                prog.update(gen_op.operation_id, status="cancelled", indeterminate=False, message="Cancelled by client disconnect")
                raise
            except Exception as exc:
                prog.fail(gen_op.operation_id, f"{type(exc).__name__}: {exc}")
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
        if img_spec is None:
            raise HTTPException(
                status_code=409,
                detail="Image generation is disabled for this session (no SELECTED_IMAGE_MODEL configured in Colab).",
            )
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        requested_model = str(body.get("model") or "").strip()
        if requested_model and requested_model != img_spec.id:
            raise HTTPException(
                status_code=400,
                detail=f"This runtime is locked to image model '{img_spec.id}'; received '{requested_model}'",
            )
        quality = str(body.get("quality") or "detailed")
        if quality not in IMAGE_QUALITY_PROFILES:
            raise HTTPException(status_code=400, detail=f"Unsupported quality profile {quality!r}; choose normal, detailed, or maximum")
        num_images = int(body.get("num_images") or body.get("n") or 1)
        if num_images != 1:
            raise HTTPException(status_code=400, detail="Nexus v1.1.0 enforces num_images=1")

        op = prog.start(
            "image_generation",
            img_spec.id,
            phase="swapping_to_image",
            indeterminate=True,
            message=f"Swapping GPU from {lang_spec.id} to {img_spec.id}",
        )

        # If Colab is staging the image model in the background, wait cleanly before unloading the LLM.
        img_dir = Path(img_spec.model_path)
        stage_lock = img_dir.parent / f".{img_spec.id}.staging.lock"
        if stage_lock.is_file() or not img_dir.is_dir():
            prog.update(
                op.operation_id,
                phase="resolving",
                indeterminate=True,
                message=f"Waiting for background staging of {img_spec.id} to finish...",
            )
            wait_deadline = time.monotonic() + 600.0
            while stage_lock.is_file() or not img_dir.is_dir():
                if time.monotonic() > wait_deadline:
                    prog.fail(op.operation_id, f"Timed out waiting for background staging of {img_spec.id}")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Image model '{img_spec.id}' is still staging from Drive; please retry in a moment.",
                    )
                time.sleep(1.0)
            prog.update(
                op.operation_id,
                phase="swapping_to_image",
                indeterminate=True,
                message=f"Swapping GPU from {lang_spec.id} to {img_spec.id}",
            )

        def _relay_progress(ev: dict[str, Any]) -> None:
            prog.update(
                op.operation_id,
                phase=ev.get("phase") or "denoising",
                step=ev.get("step"),
                steps_total=ev.get("steps_total"),
                progress=ev.get("progress"),
                indeterminate=ev.get("indeterminate"),
                message=ev.get("message"),
            )

        saved: Optional[dict[str, Any]] = None
        gen_exc: Optional[BaseException] = None
        restore_exc: Optional[BaseException] = None
        used_coresident = False

        def _is_oom_error(err: BaseException) -> bool:
            msg = str(err).lower()
            return "out of memory" in msg or "outofmemoryerror" in msg or "cuda error: out of memory" in msg

        def _run_image_gen_and_save() -> dict[str, Any]:
            raw = worker_mgr.generate_image(
                WorkerImageRequest(
                    model_id=img_spec.id,
                    prompt=prompt,
                    operation_id=op.operation_id,
                    quality=quality,  # type: ignore[arg-type]
                    width=body.get("width"),
                    height=body.get("height"),
                    steps=body.get("steps"),
                    seed=body.get("seed"),
                    num_images=1,
                    output_dir=str(config.output_image_dir),
                ),
                on_progress=_relay_progress,
            )
            prog.update(
                op.operation_id,
                phase="saving",
                indeterminate=True,
                message="Saving PNG + metadata to Drive",
            )
            return save_generated_image_artifact(
                output_dir=config.output_image_dir,
                b64_png=str(raw["b64_png"]),
                model_id=img_spec.id,
                prompt=prompt,
                quality=str(raw.get("quality") or quality),
                width=int(raw.get("width") or body.get("width") or 1024),
                height=int(raw.get("height") or body.get("height") or 1024),
                steps=int(raw.get("steps") or body.get("steps") or img_spec.default_steps),
                seed=raw.get("seed"),
                operation_id=op.operation_id,
                workspace=body.get("workspace"),
                drive_root=config.output_image_dir.parents[1] if len(config.output_image_dir.parents) >= 2 else None,
            )

        with res_mgr.exclusive(swapping=True):
            ws_now = worker_mgr.status()
            already_coresident = ws_now.alive and ws_now.active_image_id == img_spec.id and ws_now.active_language_id == lang_spec.id
            hw_now = detect_hardware(Path(lang_spec.model_path).parent)
            free_vram_now = ws_now.device_free_vram_bytes or hw_now.free_vram_bytes
            try_coresident = already_coresident or (
                lang_spec.can_coreside_image
                and can_coreside_image_model(free_vram_bytes=free_vram_now, image_min_vram_gib=img_spec.min_vram_gib)
            )

            if try_coresident:
                try:
                    if not already_coresident:
                        prog.update(
                            op.operation_id,
                            phase="weight_loading",
                            indeterminate=True,
                            message=f"Co-loading {img_spec.id} alongside {lang_spec.id} in VRAM",
                        )
                        worker_mgr.load_model(
                            WorkerLoadRequest(
                                kind="image",
                                model_id=img_spec.id,
                                model_path=img_spec.model_path,
                                implementation=img_spec.implementation,
                                architecture=img_spec.architecture,
                                keep_other_slot=True,
                            )
                        )
                    saved = _run_image_gen_and_save()
                    used_coresident = True
                except BaseException as exc:
                    if _is_oom_error(exc):
                        # OOM during co-resident load/generation: unload image and fall back to exclusive swap.
                        worker_mgr.unload(kind="image", terminate_on_failure=False)
                        used_coresident = False
                    else:
                        gen_exc = exc

            if saved is None and gen_exc is None:
                try:
                    # Exclusive swap fallback: unload LLM, load image, generate, then restore LLM.
                    worker_mgr.unload(terminate_on_failure=True)
                    prog.update(
                        op.operation_id,
                        phase="weight_loading",
                        indeterminate=True,
                        message=f"Loading pre-staged {img_spec.id} from local SSD",
                    )
                    worker_mgr.load_model(
                        WorkerLoadRequest(
                            kind="image",
                            model_id=img_spec.id,
                            model_path=img_spec.model_path,
                            implementation=img_spec.implementation,
                            architecture=img_spec.architecture,
                        )
                    )
                    saved = _run_image_gen_and_save()
                except BaseException as exc:
                    gen_exc = exc
                finally:
                    try:
                        prog.update(
                            op.operation_id,
                            phase="restoring_language_model",
                            indeterminate=True,
                            message=f"Restoring {lang_spec.id} to GPU VRAM",
                        )
                        worker_mgr.unload(terminate_on_failure=True)
                        _load_selected_language_unlocked(op.operation_id)
                    except BaseException as exc:
                        restore_exc = exc

        if gen_exc is not None:
            if isinstance(gen_exc, WorkerError) and gen_exc.status_code == 499:
                prog.update(op.operation_id, status="cancelled", indeterminate=False, error=str(gen_exc), message="Cancelled")
                raise HTTPException(status_code=499, detail=str(gen_exc)) from gen_exc
            prog.fail(op.operation_id, str(gen_exc))
            status_code = gen_exc.status_code if isinstance(gen_exc, WorkerError) else 500
            raise HTTPException(status_code=status_code, detail=str(gen_exc)) from gen_exc

        if restore_exc is not None:
            prog.fail(op.operation_id, f"Image saved, but failed to restore {lang_spec.id}: {restore_exc}")
            raise HTTPException(
                status_code=503,
                detail=f"Image generated, but restoring language model '{lang_spec.id}' failed: {restore_exc}",
            ) from restore_exc

        assert saved is not None
        done_msg = (
            f"Saved {saved['filename']} (co-resident)"
            if used_coresident
            else f"Saved {saved['filename']} and restored {lang_spec.id}"
        )
        prog.complete(op.operation_id, phase="ready", message=done_msg)
        return saved

    @app.get("/v1/plugin-index.json")
    def plugin_index() -> JSONResponse:
        path = config.plugin_index_path
        if path is not None and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return JSONResponse(content=data, headers={"Cache-Control": "no-store"})
        return JSONResponse(
            content={"format": 1, "title": {"en": "Nexus Curated Plugins"}, "metadata": {}, "plugins": []},
            headers={"Cache-Control": "no-store"},
        )

    return app
