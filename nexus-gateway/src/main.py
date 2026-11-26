"""Loopback-only uvicorn entrypoint for Nexus Gateway (v1.1.0)."""

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
from models.base import load_session_runtime_config  # noqa: E402


def build_config_from_env() -> GatewayConfig:
    api_key = os.environ.get("NEXUS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NEXUS_API_KEY must be set in the server-side environment")
    stage_root = Path(os.environ.get("NEXUS_STAGE_DIR", "/content/nexus-staged-models"))
    manifest_path = Path(
        os.environ.get("NEXUS_SESSION_MANIFEST", str(stage_root / "session-runtime.json"))
    )
    session = load_session_runtime_config(manifest_path)

    drive_root = Path(os.environ.get("NEXUS_DRIVE_ROOT", "/content/drive/MyDrive/Nexus"))
    output_dir = Path(os.environ.get("NEXUS_IMAGE_OUTPUT_DIR", str(drive_root / "outputs" / "generated_images")))
    default_index = Path(__file__).resolve().parents[2] / "colab" / "plugin-index.json"
    plugin_index = Path(os.environ.get("NEXUS_PLUGIN_INDEX_PATH", str(default_index)))
    return GatewayConfig(
        api_key=api_key,
        session=session,
        output_image_dir=output_dir,
        plugin_index_path=plugin_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus Gateway v1.1.0 (loopback-only, single-model)")
    parser.add_argument("--host", default=os.environ.get("NEXUS_GATEWAY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NEXUS_GATEWAY_PORT", "7861")))
    args = parser.parse_args()
    host = assert_loopback_host(args.host)
    app = create_app(build_config_from_env())
    uvicorn.run(app, host=host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
