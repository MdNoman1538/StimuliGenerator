"""
FastAPI routes for the Design Stimulus Generator.
"""

import asyncio
import io
import json
import os
import zipfile
from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from config import EXPORT_DIR, BASE_DIR, EXPECTED_SENTENCES, GENERATED_IMAGES_DIR, UPLOADED_IMAGES_DIR, task_slug, next_available_name
from system_prompt import SYSTEM_PROMPT
import gemini_client
from parser import extract_three_stimuli, parse_sentences
from image_generator import generate_paired_images
from logger import session_scope, current_session

LOG_PATH = BASE_DIR / "stimuli_log.jsonl"
# Hard cap on Gemini wait time so a slow primary doesn't blow the browser 3-min timeout.
GEMINI_HARD_TIMEOUT_S = 150.0

router = APIRouter()


# ── Request / Response models ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    task: str
    requirements: list[str]
    format: str = "text"

    @field_validator("task")
    @classmethod
    def task_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Task description cannot be empty")
        return v.strip()

    @field_validator("requirements")
    @classmethod
    def requirements_valid(cls, v: list[str]) -> list[str]:
        cleaned = [r.strip() for r in v if r.strip()]
        if not cleaned:
            raise ValueError("At least one requirement is required")
        return cleaned


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_user_prompt(task: str, requirements: list[str]) -> str:
    req_lines = "\n".join(f"{i+1}. {r}" for i, r in enumerate(requirements))
    return f"""Design task:
{task}

Requirements:
{req_lines}

Follow the system prompt exactly."""


def _save_report(task: str, content: str) -> str:
    slug = task_slug(task)
    filename = next_available_name(EXPORT_DIR, slug, ".txt")
    (EXPORT_DIR / filename).write_text(content, encoding="utf-8")
    return filename


def _build_report(task: str, requirements: list[str], stimuli: dict[str, str],
                   fmt: str, level: str) -> str:
    lines = [
        "Design Stimuli Report",
        "=" * 60,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Model: {gemini_client.last_model_used}",
        f"Condition: {fmt} + {level.upper()}",
        f"Task: {task}",
        "Requirements:",
        *[f"  - {r}" for r in requirements],
        "",
        "NC Stimulus:",
        stimuli.get("nc", "(missing)") or "(missing)",
        "",
        "MC Stimulus:",
        stimuli.get("mc", "(missing)") or "(missing)",
        "",
        "VC Stimulus:",
        stimuli.get("vc", "(missing)") or "(missing)",
        "",
    ]
    return "\n".join(lines)


def _append_log(task: str, requirements: list[str], stimuli: dict[str, str],
                fmt: str, level: str) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": gemini_client.last_model_used,
        "condition": f"{fmt}+{level}",
        "format": fmt,
        "level": level,
        "task": task,
        "requirements": requirements,
        "nc": stimuli.get("nc", ""),
        "mc": stimuli.get("mc", ""),
        "vc": stimuli.get("vc", ""),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_analyze_url(task: str, nc_text: str, vc_text: str) -> str:
    """Build semantic analyzer URL with NC vs VC pre-filled."""
    label = task_slug(task) or "stimulus"
    base_url = "http://localhost:8001"
    return f"{base_url}/?ta={quote(nc_text)}&tb={quote(vc_text)}&label={quote(label)}"


async def _generate_stimuli(task, requirements, fmt):
    """Single Gemini call, no validation, no retry. Return whatever comes back."""
    prompt = _build_user_prompt(task, requirements)

    try:
        raw = await asyncio.wait_for(
            gemini_client.generate(prompt, SYSTEM_PROMPT, purpose="stimuli_generation"),
            timeout=GEMINI_HARD_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return None, f"Gemini exceeded {int(GEMINI_HARD_TIMEOUT_S)}s hard timeout"
    except Exception as e:
        return None, str(e)[:800]

    stimuli = extract_three_stimuli(raw)
    _append_log(task, requirements, stimuli, fmt, "both")
    report_text = _build_report(task, requirements, stimuli, fmt, "both")
    report_file = _save_report(task, report_text)

    sess = current_session()
    if sess is not None:
        sess.set_stimuli(stimuli, report_file=report_file)
    return stimuli, report_file


# ── Routes ───────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@router.get("/model")
async def get_model():
    return {"model": gemini_client.last_model_used}


@router.post("/generate")
async def generate(req: GenerateRequest):
    with session_scope(endpoint="/generate", request_payload=req.model_dump()) as sess:
        stimuli, report_file = await _generate_stimuli(req.task, req.requirements, req.format)
        if stimuli is None:
            sess.note("stimuli_generation_failed", error=report_file)
            return {"error": report_file}

        nc_text = stimuli.get("nc", "")
        vc_text = stimuli.get("vc", "")
        analyze_url = _build_analyze_url(req.task, nc_text, vc_text)

        return {
            "nc": {"stimulus": nc_text, "level": "nc", "level_label": "NC — Not Concrete"},
            "vc": {"stimulus": vc_text, "level": "vc", "level_label": "VC — Very Concrete"},
            "format": req.format,
            "model": gemini_client.last_model_used,
            "task": req.task,
            "requirements": req.requirements,
            "report_file": report_file,
            "report_url": f"/exports/{report_file}",
            "analyze_url": analyze_url,
        }


@router.post("/generate-images")
async def generate_images(req: GenerateRequest):
    with session_scope(endpoint="/generate-images", request_payload=req.model_dump()) as sess:
        stimuli, report_file = await _generate_stimuli(req.task, req.requirements, req.format)
        if stimuli is None:
            sess.note("stimuli_generation_failed", error=report_file)
            return {"error": report_file}

        nc_text = stimuli.get("nc", "")
        vc_text = stimuli.get("vc", "")
        nc_sentences = parse_sentences(nc_text)
        vc_sentences = parse_sentences(vc_text)
        if len(nc_sentences) != EXPECTED_SENTENCES:
            err = f"NC: expected {EXPECTED_SENTENCES} sentences, got {len(nc_sentences)}"
            sess.note("sentence_count_mismatch", error=err)
            return {"error": err}
        if len(vc_sentences) != EXPECTED_SENTENCES:
            err = f"VC: expected {EXPECTED_SENTENCES} sentences, got {len(vc_sentences)}"
            sess.note("sentence_count_mismatch", error=err)
            return {"error": err}

        try:
            paired = await generate_paired_images(
                nc_text=nc_text,
                vc_text=vc_text,
                nc_sentences=nc_sentences,
                vc_sentences=vc_sentences,
                task=req.task,
                requirements=req.requirements,
            )
        except Exception as e:
            err = f"Paired image generation failed: {str(e)[:300]}"
            sess.note("paired_image_gen_failed", error=err)
            return {"error": err}

        sess.set_images({"nc": paired["nc"], "vc": paired["vc"]})

        analyze_url = _build_analyze_url(req.task, nc_text, vc_text)

        return {
            "nc": {
                "stimulus": nc_text,
                "sentences": nc_sentences,
                "images": paired["nc"],
                "level": "nc",
                "level_label": "NC — Not Concrete",
            },
            "vc": {
                "stimulus": vc_text,
                "sentences": vc_sentences,
                "images": paired["vc"],
                "level": "vc",
                "level_label": "VC — Very Concrete",
            },
            "format": req.format,
            "model": gemini_client.last_model_used,
            "task": req.task,
            "requirements": req.requirements,
            "report_file": report_file,
            "report_url": f"/exports/{report_file}",
            "analyze_url": analyze_url,
        }


@router.get("/images/{filename}")
async def serve_image(filename: str):
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = GENERATED_IMAGES_DIR / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png", filename=safe_name)


@router.post("/download-zip")
async def download_zip(payload: dict):
    """Create a zip of NC + VC images + prompts with subfolders."""
    task = payload.get("task", "")
    nc = payload.get("nc", {})
    vc = payload.get("vc", {})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for level_key, level_data in [("nc", nc), ("vc", vc)]:
            images = level_data.get("images", [])
            stimulus = level_data.get("stimulus", "")

            for img in images:
                filename = img.get("filename", "")
                path = GENERATED_IMAGES_DIR / filename
                if path.is_file():
                    zf.write(path, f"{level_key}/{filename}")

            prompts_txt = []
            for img in images:
                prompts_txt.append(f"=== {img.get('position', '')} ===")
                prompts_txt.append(f"Sentence: {img.get('sentence', '')}")
                prompts_txt.append(f"Prompt: {img.get('imagen_prompt', '')}")
                prompts_txt.append("")
            zf.writestr(f"{level_key}/prompts.txt", "\n".join(prompts_txt))
            zf.writestr(f"{level_key}/stimulus.txt", stimulus)

        meta = {"task": task, "nc": nc, "vc": vc}
        zf.writestr("metadata.json", json.dumps(meta, indent=2))

    buf.seek(0)
    slug = task_slug(task) if task else "stimuli"
    zip_name = f"{slug}_nc_vc.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@router.get("/exports/{filename}")
async def download_export(filename: str):
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = EXPORT_DIR / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=safe_name)


# ── Image description feature ────────────────────────────────────────────────

_IMG_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}


@router.post("/describe-images")
async def describe_images(files: list[UploadFile] = File(...)):
    """Accept images, save as s1..sN, stream one-line descriptions
    back via Server-Sent Events as each finishes (calls run in parallel)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    session_name = next_available_name(UPLOADED_IMAGES_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    session_dir = UPLOADED_IMAGES_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Read + save all files up front (can't hold UploadFile across generator)
    saved: list[dict] = []
    for i, upload in enumerate(files, start=1):
        data = await upload.read()
        if not data:
            continue
        mime = upload.content_type or "image/jpeg"
        ext = _IMG_EXT.get(mime, ".jpg")
        filename = f"s{i}{ext}"
        (session_dir / filename).write_bytes(data)
        saved.append({
            "position": f"s{i}",
            "filename": filename,
            "image_url": f"/uploaded-images/{session_name}/{filename}",
            "data": data,
            "mime": mime,
        })

    async def describe_one(slot: dict) -> dict:
        try:
            desc = await gemini_client.describe_image(slot["data"], mime_type=slot["mime"])
        except Exception as e:
            desc = f"[Error: {str(e)[:200]}]"
        return {
            "position": slot["position"],
            "filename": slot["filename"],
            "image_url": slot["image_url"],
            "description": desc,
        }

    async def event_stream():
        payload = {"file_count": len(saved), "session_dir": session_name}
        with session_scope(endpoint="/describe-images", request_payload=payload) as sess:
            yield f"data: {json.dumps({'type': 'session', 'session': session_name, 'total': len(saved)})}\n\n"

            tasks = [asyncio.create_task(describe_one(s)) for s in saved]
            results: list[dict] = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                sess.log_description(
                    position=result["position"],
                    filename=result["filename"],
                    description=result["description"],
                )
                yield f"data: {json.dumps({'type': 'result', 'result': result})}\n\n"

            results.sort(key=lambda r: r["position"])
            (session_dir / "descriptions.txt").write_text(
                "\n".join(f"{r['position']}: {r['description']}" for r in results),
                encoding="utf-8",
            )
            yield f"data: {json.dumps({'type': 'done', 'results': results})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/uploaded-images/{session}/{filename}")
async def serve_uploaded_image(session: str, filename: str):
    safe_session = os.path.basename(session)
    safe_name = os.path.basename(filename)
    if safe_session != session or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    path = UPLOADED_IMAGES_DIR / safe_session / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@router.post("/download-descriptions")
async def download_descriptions(payload: dict):
    """Return all descriptions as a single txt file."""
    results = payload.get("results", [])
    content = "\n".join(f"{r.get('position', '')}: {r.get('description', '')}" for r in results)
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="descriptions.txt"'},
    )
