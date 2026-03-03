"""
Minimal runnable demo of the inspector → backend → "Drive" → agent → model round‑trip.

What this gives you (single file):
- FastAPI backend with /healthz, /upload, /jobs, /jobs/{id}, /agent/process/{whiteboard_id}
- Background task that "OCRs" the upload (fake) and pushes to a local folder that simulates Google Drive
- Agent endpoint to simulate Agisoft processing and publishing results back to the "Drive"
- SQLite database for persistent job storage (survives server restarts)

Install & run:
  pip install fastapi uvicorn python-multipart
  uvicorn backend_demo:app --reload

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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import shutil
import time
import json
import re
import uuid
import sqlite3
from contextlib import contextmanager

# ----------------------------
# Paths & pseudo-Drive layout
# ----------------------------
BASE = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE / "_uploads"
DRIVE_ROOT = BASE / "_drive"
TO_PROCESS = DRIVE_ROOT / "ToProcess"
PROCESSED = DRIVE_ROOT / "Processed"
DB_PATH = BASE / "jobs.db"
MODELS_DIR = BASE / "_models"
for p in (UPLOADS_DIR, TO_PROCESS, PROCESSED, MODELS_DIR):
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


# ----------------------------
# SQLite Database
# ----------------------------
@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                whiteboard_id TEXT,
                status TEXT NOT NULL,
                ocr_confidence REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                artifacts TEXT,
                notes TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_whiteboard_id
            ON jobs(whiteboard_id)
        """)
        conn.commit()

def save_job(job: Job):
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO jobs
            (id, original_name, whiteboard_id, status, ocr_confidence,
             created_at, updated_at, artifacts, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.id,
            job.original_name,
            job.whiteboard_id,
            job.status,
            job.ocr_confidence,
            job.created_at,
            job.updated_at,
            json.dumps(job.artifacts),
            json.dumps(job.notes)
        ))
        conn.commit()

def get_job_by_id(job_id: str) -> Optional[Job]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if not row:
            return None
        return Job(
            id=row["id"],
            original_name=row["original_name"],
            whiteboard_id=row["whiteboard_id"],
            status=row["status"],
            ocr_confidence=row["ocr_confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            artifacts=json.loads(row["artifacts"]) if row["artifacts"] else [],
            notes=json.loads(row["notes"]) if row["notes"] else []
        )

def get_job_by_whiteboard_id(whiteboard_id: str) -> Optional[Job]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE whiteboard_id = ?", (whiteboard_id,)
        ).fetchone()
        if not row:
            return None
        return Job(
            id=row["id"],
            original_name=row["original_name"],
            whiteboard_id=row["whiteboard_id"],
            status=row["status"],
            ocr_confidence=row["ocr_confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            artifacts=json.loads(row["artifacts"]) if row["artifacts"] else [],
            notes=json.loads(row["notes"]) if row["notes"] else []
        )

def get_all_jobs() -> list[Job]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        return [
            Job(
                id=row["id"],
                original_name=row["original_name"],
                whiteboard_id=row["whiteboard_id"],
                status=row["status"],
                ocr_confidence=row["ocr_confidence"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                artifacts=json.loads(row["artifacts"]) if row["artifacts"] else [],
                notes=json.loads(row["notes"]) if row["notes"] else []
            )
            for row in rows
        ]

# Initialize database on startup
init_db()

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
    job = get_job_by_id(job_id)
    if not job:
        raise RuntimeError(f"Job {job_id} not found")
    try:
        wbid, conf = fake_ocr_guess_id(src_path)
        job.whiteboard_id = wbid
        job.ocr_confidence = conf
        job.status = "OCR_OK" if conf >= 0.85 else "NEEDS_REVIEW"
        job.updated_at = time.time()
        save_job(job)

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
        job.status = "PUSHED_TO_DRIVE"
        job.updated_at = time.time()
        save_job(job)
    except Exception as e:
        job.status = "FAILED"
        job.notes.append(f"pipeline error: {e}")
        job.updated_at = time.time()
        save_job(job)
        raise

def simulate_agent_process(wbid: str):
    job = get_job_by_whiteboard_id(wbid)
    if not job:
        raise RuntimeError(f"no job found for whiteboard_id={wbid}")

    toproc = TO_PROCESS / wbid
    video = toproc / "video.360"
    manifest = toproc / "manifest.json"
    if not video.exists() or not manifest.exists():
        raise RuntimeError("missing ToProcess/video.360 or manifest.json")

    job.status = "PROCESSING"
    job.updated_at = time.time()
    save_job(job)

    time.sleep(1.0)
    out = DRIVE.ensure_processed(wbid)

    (out / "model").mkdir(exist_ok=True)
    (out / "report").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    (out / "model" / "viewer.glb").write_bytes(b"glb-placeholder")
    (out / "report" / f"{wbid}_report.txt").write_text("Demo report for job " + job.id)
    (out / "logs" / "run.log").write_text("Simulated Metashape run OK")

    done = {
        "job_id": job.id,
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
    save_job(job)

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
app.mount("/static/models", StaticFiles(directory=str(MODELS_DIR)), name="static_models")

ALLOWED_MODEL_EXTS = {'.glb', '.gltf'}

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
    save_job(job)

    background_tasks.add_task(run_ocr_and_push, job_id, dest)

    return {"job_id": job_id, "status": job.status}

@app.get("/jobs")
def list_jobs():
    return {"data": [j.model_dump() for j in get_all_jobs()]}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = get_job_by_id(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job

@app.post("/agent/process/{whiteboard_id}")
def agent_process(whiteboard_id: str):
    try:
        simulate_agent_process(whiteboard_id)
        job = get_job_by_whiteboard_id(whiteboard_id)
        job_id = job.id if job else None
        return {"ok": True, "job_id": job_id}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/models/upload")
async def upload_model(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "filename required")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_MODEL_EXTS:
        raise HTTPException(422, f"Unsupported type. Allowed: {', '.join(ALLOWED_MODEL_EXTS)}")

    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest = MODELS_DIR / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "filename": safe_name,
        "original_name": file.filename,
        "url": f"/static/models/{safe_name}",
    }

@app.get("/models")
def list_models():
    models = []
    for f in sorted(MODELS_DIR.iterdir(), key=lambda x: -x.stat().st_mtime):
        if f.suffix.lower() in ALLOWED_MODEL_EXTS:
            models.append({
                "filename": f.name,
                "url": f"/static/models/{f.name}",
                "size_bytes": f.stat().st_size,
                "uploaded_at": f.stat().st_mtime,
            })
    return {"data": models}
