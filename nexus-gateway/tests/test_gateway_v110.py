"""Comprehensive v1.1.0 tests for single-model Colab inventory, staging, parsers, Gateway lock, and GPU swap-and-restore."""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from fastapi.testclient import TestClient
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GW_SRC = _REPO_ROOT / "nexus-gateway" / "src"
_COLAB_DIR = _REPO_ROOT / "colab"
for _p in (_GW_SRC, _COLAB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_v110_notebook  # noqa: E402
import model_inventory  # noqa: E402
import staging  # noqa: E402
from adapters.tool_calls import create_tool_parser, has_native_tool_parser, validate_tool_arguments  # noqa: E402
from api.app import GatewayConfig, create_app  # noqa: E402
from api.middleware.auth import assert_loopback_host, generate_bearer_token, verify_bearer_token  # noqa: E402
from models.base import (  # noqa: E402
    GenerationCancelledError,
    SelectedImageSpec,
    SelectedLanguageSpec,
    SessionRuntimeConfig,
    select_torch_dtype,
)
from models.image.flux2_klein import Flux2KleinImageModel, resolve_flux2_klein_params  # noqa: E402
from models.language.common import decode_vlm_image_url, deserialize_tool_call_arguments  # noqa: E402
from services.conversation import resolve_reasoning_controls  # noqa: E402
from workers.worker_protocol import WorkerGenerateRequest, WorkerImageRequest, WorkerLoadRequest, WorkerStatus  # noqa: E402

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "nexus_generate_image",
            "description": "Generate an image",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "steps": {"type": "integer"},
                    "quality": {"type": "string", "enum": ["normal", "detailed", "maximum"]},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    }
]


class FakeSingleModelWorker:
    """Deterministic worker stub verifying strict unload -> image -> unload -> restore sequence."""

    def __init__(self) -> None:
        self.alive = False
        self.active_kind: Optional[str] = None
        self.active_model_id: Optional[str] = None
        self.events: list[str] = []
        self.next_text = 'Working on it. <tool_call>{"name":"nexus_generate_image","arguments":{"prompt":"sunset","steps":4}}</tool_call>'
        self.fail_next_load_kind: Optional[str] = None
        self.last_gen_req: Optional[WorkerGenerateRequest] = None

    def status(self) -> WorkerStatus:
        return WorkerStatus(
            pid=4242 if self.alive else 0,
            alive=self.alive,
            active_kind=self.active_kind,  # type: ignore[arg-type]
            active_model_id=self.active_model_id,
            cuda_allocated_bytes=16 * 1024 ** 3 if self.active_model_id else 0,
        )

    def load_model(self, req: WorkerLoadRequest, *, timeout: float = 900.0) -> dict[str, Any]:
        self.events.append(f"load:{req.kind}:{req.model_id}:{req.implementation}")
        if self.fail_next_load_kind == req.kind:
            self.fail_next_load_kind = None
            raise RuntimeError(f"Simulated {req.kind} load failure")
        self.alive = True
        self.active_kind = req.kind
        self.active_model_id = req.model_id
        return {"status": "ready"}

    def unload(self, *, terminate_on_failure: bool = True) -> dict[str, Any]:
        self.events.append(f"unload:{self.active_kind}:{self.active_model_id}")
        self.active_kind = None
        self.active_model_id = None
        return {"status": "unloaded"}

    def terminate(self) -> None:
        self.events.append("terminate")
        self.alive = False
        self.active_kind = None
        self.active_model_id = None

    def generate(self, req: WorkerGenerateRequest, *, timeout: float = 600.0) -> dict[str, Any]:
        self.last_gen_req = req
        return {"text": self.next_text, "model_id": req.model_id}

    def generate_stream(self, req: WorkerGenerateRequest, *, timeout: float = 600.0) -> Iterator[str]:
        self.last_gen_req = req
        for i in range(0, len(self.next_text), 9):
            yield self.next_text[i : i + 9]

    def generate_image(
        self,
        req: WorkerImageRequest,
        *,
        on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
        timeout: float = 900.0,
    ) -> dict[str, Any]:
        self.events.append(f"gen_image:{req.model_id}")
        if on_progress is not None:
            on_progress({"phase": "denoising", "step": 2, "steps_total": 4, "progress": 50.0, "indeterminate": False})
            on_progress({"phase": "denoising", "step": 4, "steps_total": 4, "progress": 100.0, "indeterminate": False})
        return {
            "model": req.model_id,
            "quality": req.quality,
            "width": req.width or 768,
            "height": req.height or 768,
            "steps": req.steps or 4,
            "seed": req.seed,
            "b64_png": base64.b64encode(b"\x89PNG\r\n\x1a\nFAKE").decode("ascii"),
        }

    def cancel_operation(self, operation_id: Optional[str] = None) -> dict[str, Any]:
        return {"status": "cancelled", "operations": [operation_id] if operation_id else []}


def _make_drive_models(models_root: Path) -> None:
    # 1. Qwen3-VL
    qvl = models_root / "Qwen-3-VL-8B-Instruct-heretic"
    qvl.mkdir(parents=True)
    (qvl / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}))
    (qvl / "chat_template.jinja").write_text("{% if tools %}<tool_call>{% endif %}")
    (qvl / "model.safetensors").write_bytes(b"QVL_WEIGHTS")
    (qvl / ".cache" / "huggingface").mkdir(parents=True)
    (qvl / ".cache" / "huggingface" / "junk.lock").write_bytes(b"")

    # 2. Qwen3.5
    q35 = models_root / "Qwen3.5-4B"
    q35.mkdir(parents=True)
    (q35 / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}))
    (q35 / "chat_template.jinja").write_text("{% if tools %}<tool_call><function=x></function></tool_call>{% endif %}{% if enable_thinking %}<think>{% endif %}")
    (q35 / "model.safetensors").write_bytes(b"Q35_WEIGHTS")

    # 3. GPT-OSS
    gpt = models_root / "GPT-OSS-20B-Abliterated"
    gpt.mkdir(parents=True)
    (gpt / "config.json").write_text(json.dumps({"model_type": "gpt_oss", "architectures": ["GptOssForCausalLM"]}))
    (gpt / "chat_template.jinja").write_text("{% if tools %}<|call|>{% endif %}")
    (gpt / "model.safetensors").write_bytes(b"GPT_WEIGHTS")

    # 4. FLUX.2-klein-4B with scalar Diffusers flags (is_distilled, force_zeros_for_empty_prompt)
    flux = models_root / "FLUX.2-klein-4B"
    flux.mkdir(parents=True)
    (flux / "model_index.json").write_text(
        json.dumps({
            "_class_name": "Flux2KleinPipeline",
            "is_distilled": True,
            "force_zeros_for_empty_prompt": True,
            "transformer": ["diffusers", "Flux2Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLFlux2"],
            "safety_checker": [None, None],
        })
    )
    (flux / "transformer").mkdir()
    (flux / "transformer" / "config.json").write_text("{}")
    (flux / "vae").mkdir()
    (flux / "vae" / "config.json").write_text("{}")

    # 5. Excluded: GGUF & incomplete fp8 & video
    gguf = models_root / "Qwen3.5-9B-Defiant-Fable-Uncensored"
    gguf.mkdir(parents=True)
    (gguf / "model.gguf").write_bytes(b"GGUF")

    fp8 = models_root / "FLUX.2-klein-9b-fp8"
    fp8.mkdir(parents=True)
    (fp8 / "README.md").write_text("incomplete")

    vid = models_root / "HunyuanVideo-1.5"
    vid.mkdir(parents=True)
    (vid / "config.json").write_text(json.dumps({"model_type": "hunyuan_video"}))


def test_colab_inventory_validation_and_upfront_staging(tmp_path: Path) -> None:
    models_root = tmp_path / "Nexus" / "models"
    _make_drive_models(models_root)

    inv = model_inventory.inspect_drive_models(models_root, total_vram_bytes=24 * 1024 ** 3)
    assert [m.id for m in inv.language_models] == [
        "GPT-OSS-20B-Abliterated",
        "Qwen-3-VL-8B-Instruct-heretic",
        "Qwen3.5-4B",
    ]
    assert [m.id for m in inv.image_models] == ["FLUX.2-klein-4B"]
    assert {m.id for m in inv.excluded_models} == {
        "FLUX.2-klein-9b-fp8",
        "HunyuanVideo-1.5",
        "Qwen3.5-9B-Defiant-Fable-Uncensored",
    }

    lang, img = model_inventory.validate_selected_pair(
        models_root,
        "Qwen-3-VL-8B-Instruct-heretic",
        "FLUX.2-klein-4B",
        total_vram_bytes=24 * 1024 ** 3,
    )
    assert lang.implementation == "qwen3_vl"
    assert img is not None and img.implementation == "flux2_klein"

    with pytest.raises(ValueError, match="not a runnable"):
        model_inventory.validate_selected_pair(models_root, "Qwen3.5-9B-Defiant-Fable-Uncensored")

    stage_root = tmp_path / "staged"
    manifest_out = stage_root / "session-runtime.json"
    session = staging.stage_selected_models(
        local_stage_root=stage_root,
        language_candidate=lang,
        image_candidate=img,
        manifest_path=manifest_out,
        min_free_margin_bytes=0,
    )
    assert manifest_out.is_file()
    assert session.language_model.id == "Qwen-3-VL-8B-Instruct-heretic"
    assert session.image_model is not None and session.image_model.id == "FLUX.2-klein-4B"
    # .cache/ junk MUST NOT be copied into local SSD stage directory.
    assert not (Path(session.language_model.model_path) / ".cache").exists()


def test_dedicated_parsers_and_reasoning_controls() -> None:
    assert has_native_tool_parser("qwen3_vl") is True
    assert has_native_tool_parser("qwen3_5") is True
    assert has_native_tool_parser("gpt_oss") is True
    assert has_native_tool_parser("unknown") is False

    # 1. Qwen3.5 XML tool call + prefilled <think> stream
    p35 = create_tool_parser("qwen3_5", started_in_think=True)
    d1 = p35.feed("thinking step 1</think>\nSure!\n<tool_call>\n<function=nexus_generate_image>\n<parameter=prompt>\nmoon\n</parameter>\n<parameter=steps>\n4\n</parameter>\n</function>\n</tool_call>")
    res35 = p35.finish(SAMPLE_TOOLS)
    assert "thinking step 1" in d1.reasoning
    assert "Sure!" in res35.text
    assert res35.errors == []
    assert len(res35.tool_calls) == 1
    assert res35.tool_calls[0].name == "nexus_generate_image"
    assert res35.tool_calls[0].arguments == {"prompt": "moon", "steps": 4}

    # 2. GPT-OSS Harmony channels
    pgpt = create_tool_parser("gpt_oss")
    pgpt.feed('<|channel|>analysis<|message|>plan<|end|><|start|>assistant to=functions.nexus_generate_image<|channel|>commentary json<|message|>{"prompt":"prism","steps":6}<|call|>')
    resgpt = pgpt.finish(SAMPLE_TOOLS)
    assert resgpt.reasoning == "plan"
    assert resgpt.errors == []
    assert len(resgpt.tool_calls) == 1
    assert resgpt.tool_calls[0].arguments == {"prompt": "prism", "steps": 6}

    # 3. Tool-call arguments dict normalization for Jinja |items and |tojson
    normalized = deserialize_tool_call_arguments([
        {"role": "assistant", "tool_calls": [{"function": {"name": "demo", "arguments": '{"a": 1}'}}]}
    ])
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    # 4. Explicit "off" effort disables thinking
    q35_spec = SelectedLanguageSpec(
        id="Qwen3.5-4B",
        display_name="Qwen3.5-4B",
        model_path="/tmp/q35",
        implementation="qwen3_5",
        architecture="Qwen3_5ForConditionalGeneration",
        reasoning_mode="binary_toggle",
        reasoning_efforts={"off": "off", "high": "high"},
        default_reasoning_effort="high",
    )
    assert resolve_reasoning_controls(q35_spec, "off") == ("off", False)
    assert resolve_reasoning_controls(q35_spec, "high") == ("high", True)


def test_gateway_single_model_lock_readiness_and_image_swap_restore(tmp_path: Path) -> None:
    lang_dir = tmp_path / "staged" / "Qwen-3-VL-8B-Instruct-heretic"
    img_dir = tmp_path / "staged" / "FLUX.2-klein-4B"
    lang_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    session = SessionRuntimeConfig(
        language_model=SelectedLanguageSpec(
            id="Qwen-3-VL-8B-Instruct-heretic",
            display_name="Qwen-3-VL-8B-Instruct-heretic",
            model_path=str(lang_dir),
            implementation="qwen3_vl",
            architecture="Qwen3VLForConditionalGeneration",
            modalities=["text", "image"],
        ),
        image_model=SelectedImageSpec(
            id="FLUX.2-klein-4B",
            display_name="FLUX.2-klein-4B",
            model_path=str(img_dir),
            implementation="flux2_klein",
            architecture="Flux2KleinPipeline",
            default_steps=6,
        ),
    )
    worker = FakeSingleModelWorker()
    token = "test-v110-secret"
    headers = {"Authorization": f"Bearer {token}"}
    app = create_app(
        GatewayConfig(
            api_key=token,
            session=session,
            output_image_dir=tmp_path / "out",
            plugin_index_path=_REPO_ROOT / "colab" / "plugin-index.json",
        ),
        worker=worker,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    # 1. Before Colab bootstrap, /health reports ready=False and /v1/chat/completions returns 503 (NO lazy load!).
    assert client.get("/health").json()["ready"] is False
    pre_chat = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "Qwen-3-VL-8B-Instruct-heretic", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert pre_chat.status_code == 503
    assert worker.events == []

    # 2. Second / unselected language model bootstrap is rejected with HTTP 409.
    bad_boot = client.post("/v1/runtime/bootstrap", headers=headers, json={"model": "Other-Model"})
    assert bad_boot.status_code == 409

    # 3. Colab bootstrap warm-loads the single selected language model -> ready=True.
    ok_boot = client.post("/v1/runtime/bootstrap", headers=headers, json={})
    assert ok_boot.status_code == 200
    assert client.get("/health").json()["ready"] is True
    assert len(client.get("/v1/models", headers=headers).json()["data"]) == 1

    # 4. Unselected model in /v1/chat/completions returns HTTP 400.
    wrong_chat = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "Other-Model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert wrong_chat.status_code == 400

    # 5. Full tool loop: Chat -> tool_call -> Image generation (atomic unload LLM -> load FLUX -> gen -> unload FLUX -> reload LLM) -> Follow-up chat!
    chat_res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "Qwen-3-VL-8B-Instruct-heretic",
            "messages": [{"role": "user", "content": "draw sunset"}],
            "tools": SAMPLE_TOOLS,
        },
    )
    assert chat_res.status_code == 200
    assert chat_res.json()["choices"][0]["finish_reason"] == "tool_calls"

    img_res = client.post(
        "/v1/images/generations",
        headers=headers,
        json={"prompt": "sunset", "quality": "normal", "num_images": 1},
    )
    assert img_res.status_code == 200
    assert Path(img_res.json()["master_path"]).is_file()
    assert worker.events[-5:] == [
        "unload:language:Qwen-3-VL-8B-Instruct-heretic",
        "load:image:FLUX.2-klein-4B:flux2_klein",
        "gen_image:FLUX.2-klein-4B",
        "unload:image:FLUX.2-klein-4B",
        "load:language:Qwen-3-VL-8B-Instruct-heretic:qwen3_vl",
    ]
    assert client.get("/v1/runtime", headers=headers).json()["ready"] is True

    # Follow-up turn succeeds immediately with LLM already restored on GPU!
    worker.next_text = "Here is your sunset image."
    follow = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "Qwen-3-VL-8B-Instruct-heretic",
            "messages": [
                {"role": "user", "content": "draw sunset"},
                {"role": "tool", "tool_call_id": "call_1", "content": "Generated image sunset.png"},
            ],
        },
    )
    assert follow.status_code == 200
    assert follow.json()["choices"][0]["message"]["content"] == "Here is your sunset image."

    # 6. Worker crash immediately flips live readiness to False.
    worker.terminate()
    assert client.get("/health").json()["ready"] is False


def test_security_vlm_validation_flux_cancel_and_notebook_builder(tmp_path: Path) -> None:
    assert assert_loopback_host("127.0.0.1") == "127.0.0.1"
    with pytest.raises(ValueError):
        assert_loopback_host("0.0.0.0")
    tok = generate_bearer_token()
    assert verify_bearer_token(tok, f"Bearer {tok}") is True
    assert verify_bearer_token(tok, "Bearer wrong") is False

    with pytest.raises(ValueError, match="Unsupported image_url scheme"):
        decode_vlm_image_url("https://example.com/ssrf.png")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        float32 = "fp32"
        float16 = "fp16"
        bfloat16 = "bf16"
        cuda = FakeCuda()

    assert select_torch_dtype(FakeTorch()) == "fp32"
    assert resolve_flux2_klein_params(quality="normal")["steps"] == 4
    with pytest.raises(ValueError):
        resolve_flux2_klein_params(num_images=2)

    # FLUX.2-klein step cancellation
    class FakePipe:
        def __call__(self, prompt: str, width: int, height: int, num_inference_steps: int, guidance_scale: float = 3.5, callback_on_step_end=None):
            for s in range(num_inference_steps):
                if callback_on_step_end:
                    callback_on_step_end(self, s, 0, {})
            return type("Out", (), {"images": []})()

    m = Flux2KleinImageModel(model_id="FLUX.2-klein-4B", model_path=tmp_path)
    m.pipe = FakePipe()
    cancel_ev = threading.Event()
    seen: list[int] = []
    with pytest.raises(GenerationCancelledError):
        m.generate({"prompt": "test", "steps": 4}, on_progress=lambda ev: (seen.append(ev["step"]), cancel_ev.set()), cancel_event=cancel_ev)
    assert seen == [1]

    # Notebook builder produces valid AST-verified v1.1.0 notebook with zero outputs.
    nb = build_v110_notebook.build_v110_notebook()
    assert len(nb["cells"]) >= 12
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
