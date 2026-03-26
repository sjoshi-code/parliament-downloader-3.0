"""
Parliament TV Downloader API

A FastAPI service that downloads video from parliamentlive.tv,
extracts the audio, and makes it available for transcription services.

Designed to sit behind an n8n workflow:
  1. n8n POSTs a Parliament TV URL here
  2. This service downloads the video + extracts audio in the background
  3. n8n polls for completion
  4. n8n fetches the audio file and sends it to AssemblyAI
"""

import os
import uuid
import asyncio
import subprocess
import shutil
import logging
from datetime import datetime
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Find ffmpeg - check multiple locations
# ---------------------------------------------------------------------------

def find_tool(name: str) -> str:
    """Find a tool on PATH or in known locations."""
    # First check PATH
    found = shutil.which(name)
    if found:
        return found
    # Check common locations
    for path in [
        f"/usr/local/bin/{name}",
        f"/usr/bin/{name}",
        f"/opt/{name}",
    ]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    # Try static_ffmpeg package
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        found = shutil.which(name)
        if found:
            return found
    except ImportError:
        pass
    return name  # fallback to just the name, hope it works

FFMPEG_PATH = find_tool("ffmpeg")
FFPROBE_PATH = find_tool("ffprobe")
YTDLP_PATH = find_tool("yt-dlp")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOWNLOAD_DIR = Path(os.getenv("DOWNLOAD_DIR", "/tmp/parliament"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

CLEANUP_HOURS = int(os.getenv("CLEANUP_HOURS", "24"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "3"))
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parliament-downloader")

logger.info(f"ffmpeg path: {FFMPEG_PATH}")
logger.info(f"ffprobe path: {FFPROBE_PATH}")
logger.info(f"yt-dlp path: {YTDLP_PATH}")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    extracting_audio = "extracting_audio"
    completed = "completed"
    failed = "failed"


class DownloadRequest(BaseModel):
    url: str
    audio_only: bool = True
    extract_subtitles: bool = True


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    url: str
    created_at: str
    message: str | None = None
    audio_file: str | None = None
    subtitle_file: str | None = None
    video_file: str | None = None
    duration_seconds: float | None = None
    file_size_mb: float | None = None


jobs: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Parliament TV Downloader",
    description="Downloads and processes parliamentlive.tv sessions for transcription.",
    version="3.0.0",
)

# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    logger.info(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def process_download(job_id: str):
    job = jobs[job_id]
    job_dir = DOWNLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    url = job["url"]
    video_path = job_dir / "video.mp4"
    audio_path = job_dir / "audio.mp3"

    try:
        # Step 1: Download video
        job["status"] = JobStatus.downloading
        logger.info(f"[{job_id}] Downloading: {url}")

        cmd = [
            YTDLP_PATH, url,
            "-o", str(video_path),
            "--no-playlist",
            "--merge-output-format", "mp4",
            "--ffmpeg-location", os.path.dirname(FFMPEG_PATH),
        ]

        if job.get("extract_subtitles", True):
            cmd.extend([
                "--write-subs", "--write-auto-subs",
                "--sub-lang", "en", "--sub-format", "vtt",
            ])

        result = run_command(cmd, timeout=3600)

        if result.returncode != 0:
            logger.error(f"[{job_id}] yt-dlp failed: {result.stderr}")
            job["status"] = JobStatus.failed
            job["message"] = f"yt-dlp error: {result.stderr[:500]}"
            return

        if not video_path.exists():
            video_files = list(job_dir.glob("video.*"))
            if video_files:
                video_path = video_files[0]
            else:
                job["status"] = JobStatus.failed
                job["message"] = "Download completed but no video file found."
                return

        logger.info(f"[{job_id}] Download complete: {video_path}")

        sub_files = list(job_dir.glob("*.vtt")) + list(job_dir.glob("*.srt"))
        if sub_files:
            job["subtitle_file"] = str(sub_files[0].name)

        # Step 2: Extract audio with ffmpeg
        job["status"] = JobStatus.extracting_audio
        logger.info(f"[{job_id}] Extracting audio using: {FFMPEG_PATH}")

        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-i", str(video_path),
            "-vn",
            "-acodec", "libmp3lame",
            "-q:a", "4",
            "-y",
            str(audio_path),
        ]

        result = run_command(ffmpeg_cmd, timeout=1800)

        if result.returncode != 0:
            logger.error(f"[{job_id}] ffmpeg failed: {result.stderr}")
            job["status"] = JobStatus.failed
            job["message"] = f"ffmpeg error: {result.stderr[:500]}"
            return

        if not audio_path.exists():
            job["status"] = JobStatus.failed
            job["message"] = "Audio extraction completed but no file produced."
            return

        audio_size = audio_path.stat().st_size / (1024 * 1024)
        job["audio_file"] = "audio.mp3"
        job["file_size_mb"] = round(audio_size, 1)

        try:
            probe = run_command([
                FFPROBE_PATH, "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ], timeout=30)
            if probe.returncode == 0 and probe.stdout.strip():
                job["duration_seconds"] = round(float(probe.stdout.strip()), 1)
        except Exception:
            pass

        if job.get("audio_only", True):
            video_path.unlink(missing_ok=True)
        else:
            job["video_file"] = video_path.name

        job["status"] = JobStatus.completed
        job["message"] = f"Audio extracted: {audio_size:.1f}MB, {job.get('duration_seconds', 'unknown')}s"
        logger.info(f"[{job_id}] Complete: {job['message']}")

    except subprocess.TimeoutExpired:
        job["status"] = JobStatus.failed
        job["message"] = "Process timed out (exceeded 1 hour)."
    except Exception as e:
        job["status"] = JobStatus.failed
        job["message"] = f"Unexpected error: {str(e)[:500]}"


async def async_process_download(job_id: str):
    async with semaphore:
        await asyncio.to_thread(process_download, job_id)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/download", response_model=JobInfo)
async def create_download(request: DownloadRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.queued,
        "url": request.url,
        "audio_only": request.audio_only,
        "extract_subtitles": request.extract_subtitles,
        "created_at": datetime.utcnow().isoformat(),
        "message": "Queued for download.",
        "audio_file": None,
        "subtitle_file": None,
        "video_file": None,
        "duration_seconds": None,
        "file_size_mb": None,
    }

    asyncio.create_task(async_process_download(job_id))
    logger.info(f"[{job_id}] Job created for: {request.url}")
    return JobInfo(**jobs[job_id])


@app.get("/status/{job_id}", response_model=JobInfo)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobInfo(**jobs[job_id])


@app.get("/audio/{job_id}")
async def get_audio(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if job["status"] != JobStatus.completed:
        raise HTTPException(status_code=409, detail=f"Job not complete. Status: {job['status']}")
    if not job.get("audio_file"):
        raise HTTPException(status_code=404, detail="No audio file available.")
    audio_path = DOWNLOAD_DIR / job_id / job["audio_file"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found on disk.")
    return FileResponse(path=str(audio_path), media_type="audio/mpeg", filename=f"parliament_{job_id}.mp3")


@app.get("/subtitles/{job_id}")
async def get_subtitles(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if not job.get("subtitle_file"):
        raise HTTPException(status_code=404, detail="No subtitles available.")
    sub_path = DOWNLOAD_DIR / job_id / job["subtitle_file"]
    if not sub_path.exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found on disk.")
    return FileResponse(path=str(sub_path), media_type="text/vtt", filename=f"parliament_{job_id}.vtt")


@app.get("/jobs")
async def list_jobs():
    sorted_jobs = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return [JobInfo(**j) for j in sorted_jobs]


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job_dir = DOWNLOAD_DIR / job_id
    if job_dir.exists():
        import shutil as sh
        sh.rmtree(job_dir)
    del jobs[job_id]
    return {"message": f"Job {job_id} deleted."}


@app.get("/health")
async def health_check():
    checks = {}

    for name, path in [("yt-dlp", YTDLP_PATH), ("ffmpeg", FFMPEG_PATH), ("ffprobe", FFPROBE_PATH)]:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
            checks[name] = {
                "available": result.returncode == 0,
                "version": result.stdout.strip().split("\n")[0][:100],
                "path": path,
            }
        except Exception as e:
            checks[name] = {"available": False, "error": str(e), "path": path}

    all_ok = all(c.get("available", False) for c in checks.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "tools": checks,
        "active_jobs": len([j for j in jobs.values() if j["status"] in ("queued", "downloading", "extracting_audio")]),
        "total_jobs": len(jobs),
    }
