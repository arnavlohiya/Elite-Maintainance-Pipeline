"""
Minimal runnable demo of the Drive → watcher → local stash step.

What this gives you (single file):
- Polls Google Drive "ToProcess" folder for new {wbid}/ uploads
- Waits for: newest unpulled *.360 + newest manifest*.json + size stable for N seconds
- Downloads both to ./_to_process_local/ as:
    {wbid}.360
    {wbid}.manifest.json
- Calls backend to update status:
    POST {BACKEND_BASE_URL}/agent/ready/{whiteboard_id}
    body: {"status": "READY_FOR_PROCESSING"}
- Persists pulled Drive file IDs to watcher_state.json so it won't re-pull

Install & run:
  pip install google-api-python-client google-auth-oauthlib
  python watcherAgentDemo.py

Notes:
- Uses the same Google user auth flow as the backend demo (client_secret.json + token.json).
- Set BACKEND_BASE_URL if your FastAPI backend isn't on localhost:8000.
- Set GDRIVE_TOPROCESS_FOLDER_ID if you want to override the default.
"""
from __future__ import annotations

from pathlib import Path
import logging
import time
import json
import os
import io
import urllib.request
import urllib.error

# --- Google Drive Integration ---
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
SCOPES = ['https://www.googleapis.com/auth/drive']

# Drive folder to monitor (ToProcess)
TO_PROCESS_FOLDER_ID = os.getenv(
    "GDRIVE_TOPROCESS_FOLDER_ID",
    "1F5a46ARx7FljMFgmhRDY9Ns194K_FacK"
)

# Backend callback
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")

# Local output
BASE = Path(__file__).resolve().parent
LOCAL_OUT_DIR = BASE / "_to_process_local"
STATE_PATH = BASE / "watcher_state.json"

# Polling and stability
POLL_SECONDS = 5
STABLE_SECONDS = 2

# Auth files
# Matches the backend demo assumption (repo root). Override if needed.
CLIENT_SECRET_FILE = Path(os.getenv("CLIENT_SECRET_FILE", str(BASE.parent.parent / "client_secret.json")))
TOKEN_PATH = BASE / "token.json"

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


# --- Google Drive adapter ---
class GoogleDrive:
    """Adapter for interacting with Google Drive."""
    def __init__(self, client_secret_file: Path):
        creds = None

        # The file token.json stores the user's access and refresh tokens
        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not client_secret_file.exists():
                    raise FileNotFoundError(f"Client secret not found at {client_secret_file}")
                flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), SCOPES)
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(TOKEN_PATH, "w") as token:
                token.write(creds.to_json())

        self.service = build("drive", "v3", credentials=creds)
        logger.info("Successfully connected to Google Drive (User Auth).")

    def list_child_folders(self, parent_id: str) -> list[dict]:
        """Lists direct child folders under parent."""
        query = (
            f"'{parent_id}' in parents and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f"trashed=false"
        )
        resp = self.service.files().list(
            q=query,
            spaces="drive",
            fields="files(id,name,modifiedTime)",
            orderBy="modifiedTime desc",
            pageSize=100
        ).execute()
        return resp.get("files", [])

    def list_files(self, parent_id: str) -> list[dict]:
        """Lists files under a folder (non-recursive)."""
        query = (
            f"'{parent_id}' in parents and "
            f"trashed=false"
        )
        resp = self.service.files().list(
            q=query,
            spaces="drive",
            fields="files(id,name,size,modifiedTime,mimeType)",
            orderBy="modifiedTime desc",
            pageSize=200
        ).execute()
        return resp.get("files", [])

    def get_metadata(self, file_id: str) -> dict:
        return self.service.files().get(
            fileId=file_id,
            fields="id,name,size,modifiedTime"
        ).execute()

    def download_file(self, file_id: str, dest_path: Path):
        """Downloads a Drive file to dest_path using a temp file + atomic rename."""
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
        request = self.service.files().get_media(fileId=file_id)

        with tmp_path.open("wb") as f:
            downloader = MediaIoBaseDownload(f, request, chunksize=1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    logger.info(f"download {dest_path.name}: {pct}%")

            f.flush()
            os.fsync(f.fileno())

        tmp_path.replace(dest_path)


# --- Completion checks ---
def wait_for_size_stable(drive: GoogleDrive, file_id: str, stable_seconds: int) -> bool:
    """Returns True if file size + modifiedTime are unchanged over stable_seconds."""
    try:
        m1 = drive.get_metadata(file_id)
        size1 = int(m1.get("size") or 0)
        mod1 = m1.get("modifiedTime")
        if size1 <= 0:
            return False

        time.sleep(stable_seconds)

        m2 = drive.get_metadata(file_id)
        size2 = int(m2.get("size") or 0)
        mod2 = m2.get("modifiedTime")

        return size1 == size2 and mod1 == mod2 and size2 > 0
    except Exception as e:
        logger.warning(f"stability check failed for {file_id}: {e}")
        return False


# --- Backend callback ---
def notify_backend_ready(whiteboard_id: str, status: str):
    url = f"{BACKEND_BASE_URL.rstrip('/')}/agent/ready/{whiteboard_id}"
    body = json.dumps({"status": status}).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp_body = resp.read().decode("utf-8", errors="ignore")
            logger.info(f"backend ready update ok: {resp.status} {resp_body}")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="ignore")
        logger.warning(f"backend ready update failed: {e.code} {msg}")
    except Exception as e:
        logger.warning(f"backend ready update error: {e}")


# --- Main watcher loop ---
def pick_newest_unpulled(files: list[dict], pulled_ids: set[str], predicate) -> dict | None:
    """Files should already be ordered by modifiedTime desc."""
    for f in files:
        fid = f.get("id")
        if not fid or fid in pulled_ids:
            continue
        if predicate(f):
            return f
    return None


def main():
    drive = GoogleDrive(CLIENT_SECRET_FILE)
    state = load_state()
    pulled_ids = set(state.get("pulled_file_ids", []))

    logger.info(f"Monitoring ToProcess folder id={TO_PROCESS_FOLDER_ID}")
    logger.info(f"Local output dir: {LOCAL_OUT_DIR}")
    logger.info(f"Backend callback: {BACKEND_BASE_URL}")
    logger.info(f"Polling every {POLL_SECONDS}s, stability N={STABLE_SECONDS}s")

    while True:
        try:
            # Each child folder is expected to be a whiteboard_id (wbid)
            wbid_folders = drive.list_child_folders(TO_PROCESS_FOLDER_ID)

            for folder in wbid_folders:
                wbid = folder.get("name")
                folder_id = folder.get("id")
                if not wbid or not folder_id:
                    continue

                files = drive.list_files(folder_id)

                newest_video = pick_newest_unpulled(
                    files,
                    pulled_ids,
                    lambda f: (f.get("name") or "").lower().endswith(".360")
                )

                newest_manifest = pick_newest_unpulled(
                    files,
                    pulled_ids,
                    lambda f: (f.get("name") or "").lower().startswith("manifest") and (f.get("name") or "").lower().endswith(".json")
                )

                # Only act when both exist
                if not newest_video or not newest_manifest:
                    continue

                video_id = newest_video["id"]
                manifest_id = newest_manifest["id"]

                # Extra safety: ensure video is stable before pulling
                if not wait_for_size_stable(drive, video_id, STABLE_SECONDS):
                    continue

                # Download to local (avoid collisions with wbid prefix)
                local_video = LOCAL_OUT_DIR / f"{wbid}.360"
                local_manifest = LOCAL_OUT_DIR / f"{wbid}.manifest.json"

                logger.info(f"[{wbid}] pulling video={newest_video.get('name')} manifest={newest_manifest.get('name')}")

                drive.download_file(video_id, local_video)
                drive.download_file(manifest_id, local_manifest)

                # Mark pulled
                pulled_ids.add(video_id)
                pulled_ids.add(manifest_id)
                state["pulled_file_ids"] = sorted(pulled_ids)
                save_state(state)

                # Notify backend
                notify_backend_ready(wbid, "READY_FOR_PROCESSING")

                logger.info(f"[{wbid}] pulled to local and marked READY_FOR_PROCESSING")

        except Exception as e:
            logger.exception(f"watch loop error: {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()