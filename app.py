"""
Parliament TV Downloader API v5

A FastAPI service that downloads video from parliamentlive.tv,
extracts the audio, and makes it available for transcription services.
Now includes speaker context extraction - scrapes witness names from
the Parliament TV event page and fetches committee members from the
Parliament Committees API.

Designed to sit behind an n8n workflow:
  1. n8n POSTs a Parliament TV URL here
  2. This service downloads the video + extracts audio in the background
  3. n8n polls for completion
  4. n8n calls /speakers to get witness + committee member names
  5. n8n fetches the audio file and sends it to AssemblyAI
  6. n8n passes transcript + speaker context to Claude for drafting
"""

import os
import re
import uuid
import asyncio
import subprocess
import shutil
import logging
import httpx
from datetime import datetime
from pathlib import Path
from enum import Enum
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Find ffmpeg - check multiple locations
# ---------------------------------------------------------------------------

def find_tool(name: str) -> str:
    """Find a tool on PATH or in known locations."""
    found = shutil.which(name)
    if found:
        return found
    for path in [
        f"/usr/local/bin/{name}",
        f"/usr/bin/{name}",
        f"/opt/{name}",
    ]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        found = shutil.which(name)
        if found:
            return found
    except ImportError:
        pass
    return name

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


class SpeakerContextRequest(BaseModel):
    url: str


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
    description="Downloads and processes parliamentlive.tv sessions for transcription. Includes speaker context extraction.",
    version="5.0.0",
)

# ---------------------------------------------------------------------------
# Speaker context extraction
# ---------------------------------------------------------------------------

async def scrape_parliament_tv_page(url: str) -> dict:
    """
    Scrape a parliamentlive.tv event page for session metadata.

    Parliament TV event pages contain:
    - The committee name (in the page title and body)
    - Witness names and roles (listed under the video player)
    - The session date and subject

    The HTML structure typically has:
    - <h1> or <h2> with the committee name
    - A section listing witnesses with their roles/organisations
    - Metadata about the session date and type
    """
    result = {
        "committee_name": None,
        "session_title": None,
        "session_date": None,
        "witnesses": [],
        "raw_text": None,
    }

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ParliamentReadoutBot/1.0)"}
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the page title - usually contains committee name
        title_tag = soup.find("title")
        if title_tag:
            result["session_title"] = title_tag.get_text(strip=True)

        # Look for committee name in various places
        # Parliament TV pages use different structures, so we try several
        for selector in [
            "h1", "h2",
            ".event-title", ".session-title",
            "[class*='committee']", "[class*='title']",
        ]:
            tag = soup.select_one(selector)
            if tag:
                text = tag.get_text(strip=True)
                if text and len(text) > 5:
                    result["committee_name"] = text
                    break

        # Extract witness/panel information
        # Parliament TV pages list witnesses in a structured section
        # Look for common patterns in the page
        page_text = soup.get_text(separator="\n", strip=True)
        result["raw_text"] = page_text[:5000]  # First 5000 chars for context

        # Try to find witness sections by looking for common headings
        witness_patterns = [
            r"(?:Witnesses?|Panell?ists?|Oral evidence)[\s:]+(.+?)(?=\n\n|\Z)",
            r"(?:giving evidence|appeared before)[\s:]+(.+?)(?=\n\n|\Z)",
        ]

        for pattern in witness_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split on common delimiters (commas, semicolons, "and")
                names = re.split(r"[;,]|\band\b", match)
                for name in names:
                    name = name.strip()
                    if name and len(name) > 2 and len(name) < 200:
                        result["witnesses"].append(name)

        # Also look for structured elements that might list witnesses
        for element in soup.select("li, dd, .witness, .panel-member, [class*='witness']"):
            text = element.get_text(strip=True)
            if text and len(text) > 3 and len(text) < 200:
                # Check if it looks like a person's name (contains title or capitalised words)
                if re.match(r"^(?:(?:Sir|Dame|Dr|Professor|Prof|Mr|Mrs|Ms|Lord|Baroness|Rt Hon)\s+)?\w+\s+\w+", text):
                    if text not in result["witnesses"]:
                        result["witnesses"].append(text)

        logger.info(f"Scraped Parliament TV page: committee={result['committee_name']}, witnesses={len(result['witnesses'])}")

    except Exception as e:
        logger.error(f"Failed to scrape Parliament TV page: {e}")
        result["error"] = str(e)

    return result


async def fetch_committee_members(committee_name: str) -> dict:
    """
    Fetch committee members from the Parliament Committees API.

    The API at committees-api.parliament.uk provides:
    - GET /api/Committees - list/search committees
    - GET /api/Committees/{id}/Members - get members of a committee

    We search by name, then fetch the membership list.
    """
    result = {
        "committee_id": None,
        "committee_full_name": None,
        "members": [],
        "chair": None,
    }

    if not committee_name:
        return result

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"Accept": "application/json"}
        ) as client:

            # Step 1: Search for the committee by name
            # Clean up the name - remove common suffixes
            search_name = committee_name
            for suffix in [" Committee", " Commission", " Sub-committee", " Subcommittee"]:
                search_name = search_name.replace(suffix, "")
            search_name = search_name.strip()

            search_url = "https://committees-api.parliament.uk/api/Committees"
            params = {
                "SearchTerm": search_name,
                "House": "1",  # Commons
                "IsCurrentMember": "true",
            }

            response = await client.get(search_url, params=params)

            if response.status_code != 200:
                # Try Lords too
                params["House"] = "2"
                response = await client.get(search_url, params=params)

            if response.status_code != 200:
                # Try without House filter
                del params["House"]
                response = await client.get(search_url, params=params)

            if response.status_code != 200:
                logger.error(f"Committees API search failed: {response.status_code}")
                result["error"] = f"API returned {response.status_code}"
                return result

            data = response.json()

            # The API returns results in various formats - try common ones
            committees = []
            if isinstance(data, list):
                committees = data
            elif isinstance(data, dict):
                committees = data.get("items", data.get("value", data.get("results", [])))

            if not committees:
                logger.info(f"No committee found for: {search_name}")
                result["error"] = f"No committee found matching '{search_name}'"
                return result

            # Take the first match
            committee = committees[0]
            committee_id = committee.get("id") or committee.get("committeeId")
            result["committee_id"] = committee_id
            result["committee_full_name"] = committee.get("name") or committee.get("committeeName")

            # Step 2: Fetch committee members
            if committee_id:
                members_url = f"https://committees-api.parliament.uk/api/Committees/{committee_id}/Members"
                members_response = await client.get(members_url)

                if members_response.status_code == 200:
                    members_data = members_response.json()

                    member_list = []
                    if isinstance(members_data, list):
                        member_list = members_data
                    elif isinstance(members_data, dict):
                        member_list = members_data.get("items", members_data.get("value", members_data.get("results", [])))

                    for member in member_list:
                        # Extract member info - API structure varies
                        member_info = {}

                        if isinstance(member, dict):
                            # Try nested structure first
                            m = member.get("member", member)
                            member_info["name"] = m.get("nameDisplayAs") or m.get("nameFullTitle") or m.get("name", "")
                            member_info["party"] = m.get("latestParty", {}).get("name", "") if isinstance(m.get("latestParty"), dict) else m.get("party", "")
                            member_info["constituency"] = m.get("latestHouseMembership", {}).get("membershipFrom", "") if isinstance(m.get("latestHouseMembership"), dict) else ""

                            # Check if they're the chair
                            role = member.get("memberRole", "") or member.get("role", "")
                            if "chair" in str(role).lower():
                                member_info["role"] = "Chair"
                                result["chair"] = member_info["name"]
                            else:
                                member_info["role"] = "Member"

                        if member_info.get("name"):
                            result["members"].append(member_info)

                    logger.info(f"Found {len(result['members'])} members for committee {result['committee_full_name']}")
                else:
                    logger.error(f"Failed to fetch members: {members_response.status_code}")

    except Exception as e:
        logger.error(f"Failed to fetch committee members: {e}")
        result["error"] = str(e)

    return result


async def get_witness_info_from_committees_site(committee_name: str, session_date: str = None) -> list[str]:
    """
    Try to get witness names from the committees.parliament.uk oral evidence page.
    This is a backup source - the Parliament TV page itself is the primary one.
    """
    witnesses = []

    if not committee_name:
        return witnesses

    try:
        # The committees site URL pattern is:
        # https://committees.parliament.uk/committee/{id}/publications/oral-evidence/
        # But we'd need the committee ID first, which we get from the API above.
        # For now, this is a placeholder for future enhancement.
        pass
    except Exception as e:
        logger.error(f"Failed to get witnesses from committees site: {e}")

    return witnesses


@app.post("/speakers")
async def get_speaker_context(request: SpeakerContextRequest):
    """
    Given a Parliament TV URL, returns the speaker context:
    - Witnesses (from the event page)
    - Committee members (from the Committees API)
    - Session metadata

    This information should be passed to the Claude drafting step
    so it can map AssemblyAI's generic speaker labels (Speaker A, Speaker B)
    to actual names.

    Usage pattern in n8n:
      1. Call POST /speakers with the Parliament TV URL
      2. Get back witnesses + committee members
      3. Include this in the Claude prompt alongside the transcript
      4. Claude uses the opening of the session (where the chair usually
         introduces witnesses) to map speakers to names
    """
    url = request.url

    # Validate URL
    parsed = urlparse(url)
    if "parliamentlive.tv" not in parsed.netloc:
        raise HTTPException(status_code=400, detail="URL must be from parliamentlive.tv")

    # Step 1: Scrape the Parliament TV page
    page_data = await scrape_parliament_tv_page(url)

    # Step 2: If we found a committee name, fetch its members
    committee_data = {}
    if page_data.get("committee_name"):
        committee_data = await fetch_committee_members(page_data["committee_name"])

    # Step 3: Build the speaker context
    speaker_context = {
        "source_url": url,
        "session": {
            "title": page_data.get("session_title"),
            "committee": page_data.get("committee_name"),
            "date": page_data.get("session_date"),
        },
        "witnesses": page_data.get("witnesses", []),
        "committee_members": committee_data.get("members", []),
        "committee_chair": committee_data.get("chair"),
        "committee_full_name": committee_data.get("committee_full_name"),
        "notes": (
            "In UK select committee hearings, the pattern is predictable: "
            "committee members (MPs) ask questions, witnesses answer. "
            "The chair opens the session and usually introduces witnesses by name. "
            "Use the opening remarks to map speaker labels to names. "
            "If a speaker asks probing questions, they are likely a committee member. "
            "If a speaker gives detailed answers about their organisation or expertise, "
            "they are likely a witness."
        ),
    }

    # Add page text excerpt if we got it (useful for Claude to parse)
    if page_data.get("raw_text"):
        speaker_context["page_text_excerpt"] = page_data["raw_text"][:3000]

    logger.info(
        f"Speaker context: {len(speaker_context['witnesses'])} witnesses, "
        f"{len(speaker_context['committee_members'])} committee members"
    )

    return speaker_context


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
            result = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=10)
            output = result.stdout.strip() or result.stderr.strip()
            version_line = output.split("\n")[0][:100] if output else ""
            is_available = "version" in version_line.lower()
            checks[name] = {
                "available": is_available,
                "version": version_line,
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
