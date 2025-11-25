"""
Minimal runnable demo of the inspector → backend → "Drive" → agent → model round‑trip.

What this gives you (single file):
- FastAPI backend with /healthz, /upload, /jobs, /jobs/{id}, /agent/process/{whiteboard_id}
- Background task that "OCRs" the upload (fake) and pushes to a local folder that simulates Google Drive
- Agent endpoint to simulate Agisoft processing and publishing results back to the "Drive"
- In‑memory job store (no DB) so it runs without setup

Install & run:
  pip install fastapi uvicorn python-multipart
  uvicorn mini_backend_pipeline:app --reload

Try it:
  1) POST a .360 file to /upload (use curl or Postman)
     curl -F "file=@sample.360" http://127.0.0.1:8000/upload
  2) Check jobs:
     curl http://127.0.0.1:8000/jobs
  3) Simulate the sponsor agent processing (choose the whiteboard_id from the job):
     curl -X POST http://127.0.0.1:8000/agent/process/GS010016
  4) Re‑check /jobs and see PROCESSED with artifact paths

Folders created in the working directory:
  _uploads/                 # transient upload stash
  _drive/ToProcess/{ID}/    # simulates Google Drive ToProcess
  _drive/Processed/{ID}/    # simulates Google Drive Processed

Replace the LocalDrive class later with a real Google Drive adapter.
"""

from __future__ import annotations
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import time
import json
import re
import uuid

# ----------------------------
# Paths & pseudo-Drive layout
# ----------------------------
BASE = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE / "_uploads"
DRIVE_ROOT = BASE / "_drive"
TO_PROCESS = DRIVE_ROOT / "ToProcess"
PROCESSED = DRIVE_ROOT / "Processed"
for p in (UPLOADS_DIR, TO_PROCESS, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# In-memory data models
# ----------------------------
from typing import Optional

class Job(BaseModel):
    id: str
    original_name: str
    whiteboard_id: Optional[str] = None
    status: str = "UPLOADED"
    ocr_confidence: Optional[float] = None
    created_at: float = time.time()
    updated_at: float = time.time()
    artifacts: list[dict] = []
    notes: list[str] = []


JOBS: dict[str, Job] = {}
IDX_BY_WBID: dict[str, str] = {}

# ----------------------------
# Simulated Drive adapter
# ----------------------------
class LocalDrive:
    def __init__(self, root: Path):
        self.root = root

    def ensure_to_process(self, wbid: str) -> Path:
        d = TO_PROCESS / wbid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensure_processed(self, wbid: str) -> Path:
        d = PROCESSED / wbid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_manifest(self, wbid: str, manifest: dict):
        folder = self.ensure_to_process(wbid)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def push_video(self, src_path: Path, wbid: str):
        folder = self.ensure_to_process(wbid)
        dest = folder / "video.360"
        shutil.copy2(src_path, dest)
        return dest

DRIVE = LocalDrive(DRIVE_ROOT)

# ----------------------------
# Fake OCR logic
# ----------------------------
WBID_REGEX = re.compile(r"([A-Za-z]{2}\d{6}|\d{4,})")

def fake_ocr_guess_id(file_path: Path) -> tuple[str, float]:
    m = WBID_REGEX.search(file_path.stem)
    if m:
        return m.group(1), 0.95
    short = uuid.uuid4().hex[:8].upper()
    return f"WB{short}", 0.50

def run_ocr_and_push(job_id: str, src_path: Path):
    job = JOBS[job_id]
    try:
        wbid, conf = fake_ocr_guess_id(src_path)
        job.whiteboard_id = wbid
        job.ocr_confidence = conf
        job.status = "OCR_OK" if conf >= 0.85 else "NEEDS_REVIEW"
        job.updated_at = time.time()

        canonical_name = f"{wbid}.360"
        manifest = {
            "job_id": job.id,
            "whiteboard_id": wbid,
            "source": str(src_path.name),
            "canonical_name": canonical_name,
            "source_checksum": "sha256:TODO",
            "size_bytes": src_path.stat().st_size,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ocr_confidence": conf,
        }
        DRIVE.push_video(src_path, wbid)
        DRIVE.write_manifest(wbid, manifest)
        IDX_BY_WBID[wbid] = job.id
        job.status = "PUSHED_TO_DRIVE"
        job.updated_at = time.time()
    except Exception as e:
        job.status = "FAILED"
        job.notes.append(f"pipeline error: {e}")
        job.updated_at = time.time()
        raise

def simulate_agent_process(wbid: str):
    job_id = IDX_BY_WBID.get(wbid)
    if not job_id:
        raise RuntimeError(f"no job found for whiteboard_id={wbid}")
    job = JOBS[job_id]

    toproc = TO_PROCESS / wbid
    video = toproc / "video.360"
    manifest = toproc / "manifest.json"
    if not video.exists() or not manifest.exists():
        raise RuntimeError("missing ToProcess/video.360 or manifest.json")

    job.status = "PROCESSING"
    job.updated_at = time.time()

    time.sleep(1.0)
    out = DRIVE.ensure_processed(wbid)

    (out / "model").mkdir(exist_ok=True)
    (out / "report").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    (out / "model" / "viewer.glb").write_bytes(b"glb-placeholder")
    (out / "report" / f"{wbid}_report.txt").write_text("Demo report for job " + job_id)
    (out / "logs" / "run.log").write_text("Simulated Metashape run OK")

    done = {
        "job_id": job_id,
        "whiteboard_id": wbid,
        "artifacts": [
            {"kind": "MODEL", "path": str((out / "model" / "viewer.glb").resolve())},
            {"kind": "REPORT", "path": str((out / "report" / f"{wbid}_report.txt").resolve())},
            {"kind": "LOG", "path": str((out / "logs" / "run.log").resolve())},
        ],
    }
    (out / "done.json").write_text(json.dumps(done, indent=2))

    job.artifacts = done["artifacts"]
    job.status = "PROCESSED"
    job.updated_at = time.time()

# ----------------------------
# FastAPI app & endpoints
# ----------------------------
app = FastAPI(title="Mini Inspector Pipeline")

# CORS configuration to allow your frontend origin
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename:
        raise HTTPException(400, "filename required")
    if not file.filename.lower().endswith(".360"):
        raise HTTPException(422, "file must have .360 extension for this demo")

    job_id = "job_" + uuid.uuid4().hex[:10]
    dest = UPLOADS_DIR / f"{job_id}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    job = Job(id=job_id, original_name=file.filename)
    JOBS[job_id] = job

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
