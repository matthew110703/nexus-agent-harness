"""Loopback-only uvicorn entrypoint for Nexus Gateway."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from api.app import GatewayConfig, create_app  # noqa: E402
from api.middleware.auth import assert_loopback_host  # noqa: E402


def build_config_from_env() -> GatewayConfig:
    api_key = os.environ.get("NEXUS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NEXUS_API_KEY must be set in the server-side environment")
    drive_root = Path(os.environ.get("NEXUS_DRIVE_ROOT", "/content/drive/MyDrive/Nexus"))
    models_root = Path(os.environ.get("NEXUS_MODELS_DIR", str(drive_root / "models")))
    adapters_root = Path(os.environ.get("NEXUS_ADAPTERS_DIR", str(drive_root / "adapters")))
    upscalers_root = Path(os.environ.get("NEXUS_UPSCALERS_DIR", str(drive_root / "upscalers")))
    sidecar = Path(
        os.environ.get("NEXUS_CAPABILITIES_SIDECAR", str(drive_root / "colab-src" / "config" / "model-capabilities.json"))
    )
    stage_root = Path(os.environ.get("NEXUS_STAGE_DIR", "/content/nexus-staged-models"))
    output_dir = Path(os.environ.get("NEXUS_IMAGE_OUTPUT_DIR", str(drive_root / "outputs" / "generated_images")))
    default_index = Path(__file__).resolve().parents[2] / "colab" / "plugin-index.json"
    plugin_index = Path(os.environ.get("NEXUS_PLUGIN_INDEX_PATH", str(default_index)))
    return GatewayConfig(
        api_key=api_key,
        models_root=models_root,
        adapters_root=adapters_root,
        upscalers_root=upscalers_root,
        capabilities_sidecar=sidecar,
        local_stage_root=stage_root,
        output_image_dir=output_dir,
        plugin_index_path=plugin_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus Gateway (loopback-only)")
    parser.add_argument("--host", default=os.environ.get("NEXUS_GATEWAY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NEXUS_GATEWAY_PORT", "7861")))
    args = parser.parse_args()
    host = assert_loopback_host(args.host)
    app = create_app(build_config_from_env())
    uvicorn.run(app, host=host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
