"""Direct Google Drive API v3 parallel multi-range transfer with server md5Checksum verification."""

from __future__ import annotations

import hashlib
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

DRIVE_MYDRIVE_PREFIX = "/content/drive/MyDrive/"
RANGE_THRESHOLD_BYTES = 128 * 1024 * 1024  # 128 MiB: split into parallel HTTP Range streams
RANGE_STREAMS_PER_FILE = 4
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB streaming buffer per worker
SKIPPED_DRIVE_DIRS = frozenset({".cache", "__pycache__", ".git", ".ipynb_checkpoints"})


class DriveApiUnavailableError(RuntimeError):
    """Raised when Colab Drive API v3 credentials or libraries are unavailable."""


@dataclass(frozen=True)
class DriveRemoteFile:
    file_id: str
    rel_path: str
    size: int
    md5_checksum: Optional[str]


def _get_colab_drive_credentials() -> Any:
    """Acquire Google OAuth2 credentials inside Colab (prompts once if needed)."""
    try:
        from google.auth import default as google_auth_default  # type: ignore[import-not-found]
        from google.auth.transport.requests import Request  # type: ignore[import-not-found]
        from google.colab import auth as colab_auth  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DriveApiUnavailableError(f"Colab Drive API libraries unavailable: {exc}") from exc

    try:
        creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
        if not creds.valid:
            creds.refresh(Request())
        return creds
    except Exception:
        pass

    try:
        colab_auth.authenticate_user()
        creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
        if not creds.valid:
            creds.refresh(Request())
        return creds
    except Exception as exc:
        raise DriveApiUnavailableError(f"Colab Drive authentication failed: {exc}") from exc


def _build_drive_service(creds: Any) -> Any:
    try:
        from googleapiclient.discovery import build  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DriveApiUnavailableError(f"google-api-python-client not installed: {exc}") from exc
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _resolve_folder_id_by_path(service: Any, rel_drive_path: str) -> Optional[str]:
    """Walk MyDrive path segments (e.g. 'Nexus/models/Gemma4-...') to find its Drive folder ID."""
    parts = [seg for seg in rel_drive_path.strip("/").split("/") if seg]
    parent_id = "root"
    for seg in parts:
        escaped = seg.replace("'", "\\'")
        q = (
            f"'{parent_id}' in parents and name = '{escaped}' "
            "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        res = service.files().list(q=q, fields="files(id, name)", pageSize=5).execute()
        files = res.get("files") or []
        if not files:
            return None
        parent_id = str(files[0]["id"])
    return parent_id


def _list_remote_tree(service: Any, folder_id: str, prefix: str = "") -> list[DriveRemoteFile]:
    """Recursively list non-trashed files under a Drive folder with size and md5Checksum."""
    results: list[DriveRemoteFile] = []
    page_token: Optional[str] = None
    while True:
        res = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, size, md5Checksum)",
                pageSize=200,
                pageToken=page_token,
            )
            .execute()
        )
        for item in res.get("files") or []:
            name = str(item.get("name") or "")
            if not name or name in SKIPPED_DRIVE_DIRS:
                continue
            mime = str(item.get("mimeType") or "")
            rel = f"{prefix}/{name}" if prefix else name
            if mime == "application/vnd.google-apps.folder":
                results.extend(_list_remote_tree(service, str(item["id"]), rel))
            elif not mime.startswith("application/vnd.google-apps."):
                size = int(item.get("size") or 0)
                md5 = item.get("md5Checksum")
                results.append(
                    DriveRemoteFile(
                        file_id=str(item["id"]),
                        rel_path=rel,
                        size=size,
                        md5_checksum=str(md5) if md5 else None,
                    )
                )
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return results


def _download_range_slice(
    *,
    file_id: str,
    token: str,
    start: int,
    end: int,
    fd: int,
    write_lock: Optional[threading.Lock],
    on_bytes: Callable[[int], None],
    retries: int = 4,
) -> None:
    """Download an inclusive byte range [start, end] from Drive API v3 into fd at offset `start`."""
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    pos = start
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Range": f"bytes={pos}-{end}",
                    "User-Agent": "nexus-fast-stage/1.2.1",
                },
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                while True:
                    buf = resp.read(DOWNLOAD_CHUNK_BYTES)
                    if not buf:
                        break
                    if hasattr(os, "pwrite"):
                        os.pwrite(fd, buf, pos)
                    else:
                        assert write_lock is not None
                        with write_lock:
                            os.lseek(fd, pos, os.SEEK_SET)
                            os.write(fd, buf)
                    pos += len(buf)
                    on_bytes(len(buf))
            if pos <= end:
                raise IOError(f"Short read on range {start}-{end}: stopped at {pos}")
            return
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(8.0, 1.5 ** attempt))
    raise RuntimeError(f"Range download failed for {file_id} ({start}-{end}): {last_exc}") from last_exc


def download_file_via_drive_api(
    *,
    remote: DriveRemoteFile,
    dest_file: Path,
    token: str,
    on_bytes: Callable[[int], None],
    range_streams: int = RANGE_STREAMS_PER_FILE,
) -> Optional[str]:
    """
    Download a single Drive file atomically to `dest_file` using parallel HTTP Range streams
    for large files, then verify against Drive's `md5Checksum` when present.
    Returns the verified local MD5 hex string (or None if unverified).
    """
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = dest_file.with_name(f".{dest_file.name}.staging.tmp")

    try:
        if remote.size == 0:
            tmp_file.write_bytes(b"")
            os.replace(tmp_file, dest_file)
            return hashlib.md5(b"").hexdigest()

        # Pre-allocate destination file so concurrent Range writers can write at independent offsets.
        fd = os.open(str(tmp_file), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, remote.size)
            write_lock = None if hasattr(os, "pwrite") else threading.Lock()
            if remote.size >= RANGE_THRESHOLD_BYTES and range_streams > 1:
                slice_size = (remote.size + range_streams - 1) // range_streams
                ranges: list[tuple[int, int]] = []
                offset = 0
                while offset < remote.size:
                    end = min(remote.size - 1, offset + slice_size - 1)
                    ranges.append((offset, end))
                    offset = end + 1
                with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
                    futs = [
                        pool.submit(
                            _download_range_slice,
                            file_id=remote.file_id,
                            token=token,
                            start=s,
                            end=e,
                            fd=fd,
                            write_lock=write_lock,
                            on_bytes=on_bytes,
                        )
                        for s, e in ranges
                    ]
                    for fut in as_completed(futs):
                        fut.result()
            else:
                _download_range_slice(
                    file_id=remote.file_id,
                    token=token,
                    start=0,
                    end=remote.size - 1,
                    fd=fd,
                    write_lock=write_lock,
                    on_bytes=on_bytes,
                )
        finally:
            os.close(fd)

        if tmp_file.stat().st_size != remote.size:
            raise IOError(
                f"Size mismatch for {remote.rel_path}: expected {remote.size}, got {tmp_file.stat().st_size}"
            )

        # Verify against Drive's server-side md5Checksum on local NVMe (fast sequential read, ~700 MB/s).
        verified_md5: Optional[str] = None
        if remote.md5_checksum:
            h_md5 = hashlib.md5()
            with tmp_file.open("rb") as fh:
                while True:
                    chunk = fh.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    h_md5.update(chunk)
            verified_md5 = h_md5.hexdigest()
            if verified_md5.lower() != remote.md5_checksum.lower():
                raise IOError(
                    f"MD5 mismatch for {remote.rel_path}: Drive={remote.md5_checksum}, local={verified_md5}"
                )

        os.replace(tmp_file, dest_file)
        return verified_md5
    except Exception:
        tmp_file.unlink(missing_ok=True)
        raise


def list_drive_model_files(source_dir: Path) -> tuple[str, dict[str, DriveRemoteFile]]:
    """
    Resolve a mounted `/content/drive/MyDrive/...` model folder to Drive API v3 remote files.
    Returns `(access_token, {rel_posix_path: DriveRemoteFile})`.
    """
    src_str = str(source_dir)
    if not src_str.startswith(DRIVE_MYDRIVE_PREFIX):
        raise DriveApiUnavailableError(f"Path {src_str} is not under {DRIVE_MYDRIVE_PREFIX}")

    creds = _get_colab_drive_credentials()
    token = getattr(creds, "token", None)
    if not isinstance(token, str) or not token:
        raise DriveApiUnavailableError("No valid OAuth access token available for Drive API v3")

    service = _build_drive_service(creds)
    rel_drive = src_str[len(DRIVE_MYDRIVE_PREFIX) :]
    folder_id = _resolve_folder_id_by_path(service, rel_drive)
    if not folder_id:
        raise DriveApiUnavailableError(f"Could not resolve Drive folder ID for {rel_drive!r}")

    items = _list_remote_tree(service, folder_id)
    return token, {item.rel_path: item for item in items}
