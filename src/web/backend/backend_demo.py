"""
Minimal runnable demo of the inspector → backend → Google Drive → agent → model round‑trip.

What this gives you (single file):
- FastAPI backend with /healthz, /upload, /jobs, /jobs/{id}, /agent/process/{whiteboard_id}
- Background task that "OCRs" the upload (fake) and pushes to Google Drive
- Agent endpoint to simulate Agisoft processing and publishing results back to Drive
- In‑memory job store (no DB) so it runs without setup

Install & run:
  pip install fastapi uvicorn python-multipart google-api-python-client google-auth-oauthlib
  uvicorn backendDemo:app --reload

Try it:
  1) POST a .360 file to /upload (use curl or Postman)
     curl -F "file=@sample.360" http://127.0.0.1:8000/upload
  2) Check jobs:
     curl http://127.0.0.1:8000/jobs
  3) Simulate the sponsor agent processing (choose the whiteboard_id from the job):
     curl -X POST http://127.0.0.1:8000/agent/process/GS010016
  4) Re-check /jobs and see PROCESSED with artifact paths

Folders created in the working directory:
  _uploads/                 # transient upload stash

Replace the fake OCR logic later with real processing.
"""
from __future__ import annotations
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil, hashlib, logging
import time, json, re, uuid, io
import os
from dotenv import load_dotenv

# --- Google Drive Integration ---
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
SCOPES = ['https://www.googleapis.com/auth/drive']
# Default ID provided, but allows override via environment variable
GDRIVE_ROOT_FOLDER_ID = os.getenv("GDRIVE_ROOT_FOLDER_ID")
if not GDRIVE_ROOT_FOLDER_ID:
    raise RuntimeError("GDRIVE_ROOT_FOLDER_ID is not set. Please check your .env file.")

# Paths
BASE = Path(__file__).resolve().parent
UPLOADS_DIR = BASE / "_uploads"
# Assumes client_secret.json is three levels up from this script (repo root)
CLIENT_SECRET_FILE = BASE.parent.parent.parent / "client_secret.json"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# --- In-memory data models ---
class Job(BaseModel):
    id: str
    original_name: str
    whiteboard_id: str | None = None
    status: str = "UPLOADED"  # UPLOADED, OCR_OK, NEEDS_REVIEW, PUSHED_TO_DRIVE, PROCESSING, PROCESSED, FAILED
    ocr_confidence: float | None = None
    created_at: float = time.time()
    updated_at: float = time.time()
    artifacts: list[dict] = []  # [{kind, path}]
    notes: list[str] = []

JOBS: dict[str, Job] = {}
IDX_BY_WBID: dict[str, str] = {}  # whiteboard_id -> job_id

# --- Helper Functions ---
def calculate_sha256(file_path: Path) -> str:
    """Calculates the SHA256 checksum of a file efficiently."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read in 8k chunks to avoid memory issues with large files
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return f"sha256:{sha256_hash.hexdigest()}"
    except FileNotFoundError:
        return "sha256:ERROR_FILE_NOT_FOUND"

# --- Google Drive adapter ---
class GoogleDrive:
    """Adapter for interacting with Google Drive."""
    def __init__(self, client_secret_file: Path, root_folder_id: str):
        try:
            creds = None
            token_path = BASE / 'token.json'
            
            # The file token.json stores the user's access and refresh tokens
            if token_path.exists():
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            
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
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())

            self.service = build('drive', 'v3', credentials=creds)
            self.root_folder_id = root_folder_id
            self.folder_ids_cache = {}  # Cache to reduce API calls: (name, parent_id) -> folder_id
            print("Successfully connected to Google Drive (User Auth).")
        except Exception as e:
            print(f"ERROR: Failed to initialize Google Drive: {e}")
            self.service = None

    def _get_or_create_folder(self, name: str, parent_id: str) -> str | None:
        if not self.service: return None
        cache_key = (name, parent_id)
        if cache_key in self.folder_ids_cache:
            return self.folder_ids_cache[cache_key]
        
        query = f"name='{name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        response = self.service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        files = response.get('files', [])
        
        if files:
            folder_id = files[0]['id']
        else:
            file_metadata = {'name': name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_id]}
            folder = self.service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder['id']
            
        self.folder_ids_cache[cache_key] = folder_id
        return folder_id
        
    def ensure_to_process(self, wbid: str) -> str | None:
        """Ensures ToProcess/{wbid} folder exists and returns the wbid folder ID."""
        if not self.service: return None
        to_process_root_id = self._get_or_create_folder("ToProcess", self.root_folder_id)
        if not to_process_root_id: return None
        return self._get_or_create_folder(wbid, to_process_root_id)

    def ensure_processed(self, wbid: str) -> str | None:
        """Ensures Processed/{wbid} folder exists and returns the wbid folder ID."""
        if not self.service: return None
        processed_root_id = self._get_or_create_folder("Processed", self.root_folder_id)
        if not processed_root_id: return None
        return self._get_or_create_folder(wbid, processed_root_id)

    def write_manifest(self, manifest: dict, parent_folder_id: str | None):
        if not self.service or not parent_folder_id: return
        file_metadata = {'name': 'manifest.json', 'parents': [parent_folder_id]}
        manifest_bytes = json.dumps(manifest, indent=2).encode('utf-8')
        media = MediaIoBaseUpload(io.BytesIO(manifest_bytes), mimetype='application/json', resumable=True)
        self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    def _get_unique_name(self, parent_id: str, desired_name: str) -> str:
        if not self.service: return desired_name
        
        # Get all filenames in the parent folder
        query = f"'{parent_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, fields="files(name)").execute()
        existing_names = {f['name'] for f in results.get('files', [])}
        
        if desired_name not in existing_names:
            return desired_name
            
        name_stem, name_ext = os.path.splitext(desired_name)
        counter = 1
        while True:
            new_name = f"{name_stem} ({counter}){name_ext}"
            if new_name not in existing_names:
                return new_name
            counter += 1

    def push_video(self, src_path: Path, parent_folder_id: str | None, name: str):
        if not self.service or not parent_folder_id: return None
        final_name = self._get_unique_name(parent_folder_id, name)
        file_metadata = {'name': final_name, 'parents': [parent_folder_id]}
        media = MediaFileUpload(str(src_path), mimetype='application/octet-stream', resumable=True)
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')

    def upload_content(self, content: bytes, name: str, parent_folder_id: str | None, mimetype: str) -> str | None:
        """Uploads bytes content to a file in Google Drive."""
        if not self.service or not parent_folder_id: return None
        file_metadata = {'name': name, 'parents': [parent_folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mimetype, resumable=True)
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')

# Initialize Drive
try:
    DRIVE = GoogleDrive(CLIENT_SECRET_FILE, GDRIVE_ROOT_FOLDER_ID)
except Exception as e:
    print(f"CRITICAL: Could not instantiate GoogleDrive client. {e}")
    DRIVE = None

# --- Fake OCR (extract digits/letters from filename) ---
WBID_REGEX = re.compile(r"([A-Za-z]{2}\d{6}|\d{4,})")

def fake_ocr_guess_id(file_path: Path) -> tuple[str, float]:
    """Pretend to OCR the whiteboard. Pull an ID‑looking token from name.
    Confidence 0.95 if match, else synthesize an ID with 0.5.
    """
    m = WBID_REGEX.search(file_path.stem)
    if m:
        return m.group(1), 0.95
    # synthesize deterministic short ID from UUID
    short = uuid.uuid4().hex[:8].upper()
    return f"WB{short}", 0.50

# --- Background pipeline steps ---
def run_ocr_and_push(job_id: str, src_path: Path):
    job = JOBS[job_id]
    try:
        # 1) OCR
        wbid, conf = fake_ocr_guess_id(Path(job.original_name))
        job.whiteboard_id = wbid
        job.ocr_confidence = conf
        job.status = "OCR_OK" if conf >= 0.85 else "NEEDS_REVIEW"
        job.updated_at = time.time()

        # 2) Canonicalize name & push to "Drive"
        canonical_name = job.original_name
        
        # Calculate Checksum
        checksum = calculate_sha256(src_path)

        manifest = {
            "job_id": job.id,
            "whiteboard_id": wbid,
            "source": str(src_path.name),
            "canonical_name": canonical_name,
            "source_checksum": checksum,
            "size_bytes": src_path.stat().st_size,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ocr_confidence": conf,
        }
        
        if not DRIVE or not DRIVE.service:
            raise Exception("Google Drive service is not available.")

        to_process_folder_id = DRIVE.ensure_to_process(wbid)
        DRIVE.push_video(src_path, to_process_folder_id, canonical_name)
        DRIVE.write_manifest(manifest, to_process_folder_id)
        IDX_BY_WBID[wbid] = job.id
        job.status = "PUSHED_TO_DRIVE"
        job.updated_at = time.time()
    except Exception as e:
        job.status = "FAILED"
        job.notes.append(f"pipeline error: {e}")
        job.updated_at = time.time()
        print(f"Job {job_id} failed: {e}")
        logger.exception(f"Job {job_id} failed")

def simulate_agent_process(wbid: str):
    """Simulate the sponsor agent: download from ToProcess, run Agisoft, upload to Processed."""
    job_id = IDX_BY_WBID.get(wbid)
    if not job_id:
        raise RuntimeError(f"no job found for whiteboard_id={wbid}")
    job = JOBS[job_id]

    # Sanity checks
    if job.status != "PUSHED_TO_DRIVE":
        raise RuntimeError(f"Job {job_id} is not ready for processing (status: {job.status})")
    if not DRIVE or not DRIVE.service:
        raise RuntimeError("Google Drive service is not available for agent processing.")

    # Mark processing
    job.status = "PROCESSING"
    job.updated_at = time.time()

    # The real agent would now download the video from the 'ToProcess' folder in Google Drive.
    # Simulate this by just moving to the next step.

    # "Run Agisoft" (sleep to simulate), then publish outputs
    time.sleep(1.0)
    processed_folder_id = DRIVE.ensure_processed(wbid)
    if not processed_folder_id:
        raise RuntimeError(f"Could not create 'Processed' folder for {wbid} in Google Drive.")

    # Create subfolders and upload placeholder artifacts
    model_folder_id = DRIVE._get_or_create_folder("model", processed_folder_id)
    report_folder_id = DRIVE._get_or_create_folder("report", processed_folder_id)
    logs_folder_id = DRIVE._get_or_create_folder("logs", processed_folder_id)

    model_id = DRIVE.upload_content(b"glb-placeholder", "viewer.glb", model_folder_id, "model/gltf-binary")
    report_id = DRIVE.upload_content(f"Demo report for job {job_id}".encode(), f"{wbid}_report.txt", report_folder_id, "text/plain")
    log_id = DRIVE.upload_content("Simulated Metashape run OK".encode(), "run.log", logs_folder_id, "text/plain")

    # done.json
    done = {
        "job_id": job_id,
        "whiteboard_id": wbid,
        "artifacts": [
            {"kind": "MODEL", "drive_id": model_id},
            {"kind": "REPORT", "drive_id": report_id},
            {"kind": "LOG", "drive_id": log_id},
        ],
    }
    DRIVE.upload_content(json.dumps(done, indent=2).encode(), "done.json", processed_folder_id, "application/json")

    # Callback (inline for demo): mark PROCESSED
    job.artifacts = done["artifacts"]
    job.status = "PROCESSED"
    job.updated_at = time.time()

# --- FastAPI app & endpoints ---
app = FastAPI(title="Mini Inspector Pipeline")

@app.get("/")
def root():
    return {"message": "Elite Maintenance Pipeline Backend is running. Visit /docs for the API UI."}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename:
        raise HTTPException(400, "filename required")
    # Very light validation
    if not file.filename.lower().endswith(".360"):
        raise HTTPException(422, "file must have .360 extension for this demo")

    # Save to uploads
    job_id = "job_" + uuid.uuid4().hex[:10]
    dest = UPLOADS_DIR / f"{job_id}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    job = Job(id=job_id, original_name=file.filename)
    JOBS[job_id] = job

    # Kick off OCR->Drive push in background
    background_tasks.add_task(run_ocr_and_push, job_id, dest)

    return {"job_id": job_id, "status": job.status}

@app.get("/jobs")
def list_jobs():
    return {"data": [j.model_dump() for j in JOBS.values()]}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job

@app.post("/agent/process/{whiteboard_id}")
def agent_process(whiteboard_id: str):
    try:
        simulate_agent_process(whiteboard_id)
        job_id = IDX_BY_WBID.get(whiteboard_id)
        return {"ok": True, "job_id": job_id}
    except Exception as e:
        raise HTTPException(400, str(e))
