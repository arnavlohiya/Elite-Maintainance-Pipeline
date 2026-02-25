"""
Minimal runnable demo of the Local Watcher Agent step.

- Watches a ToProcess folder for new {wbid}/ uploads
- Waits for: newest unpulled *.360 + newest manifest*.json + size stable for N seconds
- Pulls both to ./_to_process_local/ as:
    {wbid}.360
    {wbid}.manifest.json
- Persists pulled file IDs to watcher_state.json so it won't re-pull


Run:
  LOCAL_TOPROCESS_DIR="./mock_drive/ToProcess" python3 local_agent.py
"""
from __future__ import annotations

from pathlib import Path
import logging
import time
import json
import os

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
POLL_SECONDS = 2
STABLE_SECONDS = 2

BASE = Path(__file__).resolve().parent
LOCAL_OUT_DIR = BASE / "_to_process_local"
STATE_PATH = BASE / "watcher_state.json"

LOCAL_TOPROCESS_DIR = os.getenv("LOCAL_TOPROCESS_DIR")  # required

LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- State helpers ---
def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"pulled_file_ids": []}
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"pulled_file_ids": []}


def save_state(state: dict):
    tmp = STATE_PATH.with_suffix(".json.partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(STATE_PATH)


# --- Local folder adapter ---
class LocalFolderDrive:
    """Adapter for interacting with a local folder that mirrors Drive structure."""
    def __init__(self, to_process_dir: Path):
        self.to_process_dir = to_process_dir
        self.to_process_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Watching LOCAL_TOPROCESS_DIR={self.to_process_dir}")

    def list_child_folders(self) -> list[Path]:
        out = []
        for p in self.to_process_dir.iterdir():
            if p.is_dir():
                out.append(p)
        # newest first
        out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return out

    def list_files(self, folder: Path) -> list[Path]:
        out = []
        if not folder.exists():
            return out
        for p in folder.iterdir():
            if p.is_file():
                out.append(p)
        # newest first
        out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return out


# --- Completion checks ---
def is_size_stable(file_path: Path, stable_seconds: int) -> bool:
    """Returns True if file size + mtime are unchanged over stable_seconds."""
    try:
        st1 = file_path.stat()
        size1 = st1.st_size
        mtime1 = st1.st_mtime
        if size1 <= 0:
            return False

        time.sleep(stable_seconds)

        st2 = file_path.stat()
        size2 = st2.st_size
        mtime2 = st2.st_mtime

        return size1 == size2 and mtime1 == mtime2 and size2 > 0
    except Exception as e:
        logger.warning(f"stability check failed for {file_path}: {e}")
        return False


def pick_newest(paths: list[Path], predicate) -> Path | None:
    for p in paths:
        if predicate(p):
            return p
    return None


def safe_copy(src: Path, dest: Path):
    """Copy src to dest using temp + atomic rename."""
    tmp = dest.with_suffix(dest.suffix + ".partial")

    with src.open("rb") as r, tmp.open("wb") as w:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            w.write(chunk)

        w.flush()
        os.fsync(w.fileno())

    tmp.replace(dest)


def main():
    if not LOCAL_TOPROCESS_DIR:
        raise SystemExit("LOCAL_TOPROCESS_DIR is not set. Example: LOCAL_TOPROCESS_DIR=./mock_drive/ToProcess")

    drive = LocalFolderDrive(Path(LOCAL_TOPROCESS_DIR))
    state = load_state()
    pulled_ids = set(state.get("pulled_file_ids", []))

    logger.info(f"Output dir: {LOCAL_OUT_DIR}")
    logger.info(f"Polling every {POLL_SECONDS}s, stability N={STABLE_SECONDS}s")

    while True:
        try:
            for wbid_folder in drive.list_child_folders():
                wbid = wbid_folder.name

                files = drive.list_files(wbid_folder)

                newest_video = pick_newest(files, lambda p: p.name.lower().endswith(".360"))
                newest_manifest = pick_newest(files, lambda p: p.name.lower().startswith("manifest") and p.name.lower().endswith(".json"))

                if not newest_video or not newest_manifest:
                    continue

                video_id = str(newest_video.resolve())
                manifest_id = str(newest_manifest.resolve())

                # Only process if we haven't already pulled these exact files
                if video_id in pulled_ids and manifest_id in pulled_ids:
                    continue

                # Extra safety: ensure video is stable before pulling
                if not is_size_stable(newest_video, STABLE_SECONDS):
                    continue

                local_video = LOCAL_OUT_DIR / f"{wbid}.360"
                local_manifest = LOCAL_OUT_DIR / f"{wbid}.manifest.json"

                logger.info(f"[{wbid}] pulling video={newest_video.name} manifest={newest_manifest.name}")

                safe_copy(newest_video, local_video)
                safe_copy(newest_manifest, local_manifest)

                pulled_ids.add(video_id)
                pulled_ids.add(manifest_id)
                state["pulled_file_ids"] = sorted(pulled_ids)
                save_state(state)

                logger.info(f"[{wbid}] pulled to local successfully")

        except Exception as e:
            logger.exception(f"watch loop error: {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
