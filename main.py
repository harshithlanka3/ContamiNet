import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from numpy.typing import NDArray
from ultralytics import YOLO

from vlm_providers import init_google_backend, parse_vlm_provider, vlm_generate_json_text

log = logging.getLogger(__name__)


def _debug_timing_enabled() -> bool:
    return os.environ.get("CONTAMINET_DEBUG", "").lower() in ("1", "true", "yes")


def _phase(req_id: str, label: str, t_prev: float | None) -> float:
    """Log wall-clock step timing when CONTAMINET_DEBUG is set; always return monotonic time."""
    now = time.monotonic()
    if not _debug_timing_enabled():
        return now
    if t_prev is None:
        log.info("contaminet [%s] %s (t=%.3f)", req_id, label, now)
    else:
        log.info(
            "contaminet [%s] %s (dt=%.3fs, t=%.3f)",
            req_id,
            label,
            now - t_prev,
            now,
        )
    return now


YOLO_WEIGHTS = os.environ.get("YOLO_WORLD_WEIGHTS", "yolov8s-world.pt")
YOLO_CONF = float(os.environ.get("YOLO_WORLD_CONF", "0.05"))
CROP_PADDING = float(os.environ.get("YOLO_CROP_PADDING", "0.15"))

# Matches js/VLM-implementation yolo-world-test.py open-vocabulary classes
YOLO_CLASSES = [
    "plastic",
    "clear plastic container",
    "transparent plastic box",
    "empty plastic tray",
    "plastic packaging",
    "plastic container",
    "liquid in plastic container",
    "plastic water bottle",
    "plastic film",
    "plastic bag",
    "",
]

CONTAMINATION_PROMPT = """Analyze this plastic cup/container image for recycling contamination.

Step 1 (image description): Describe what is visibly inside the cup/container/plastic container/bag/plastic waste interior. Include:
- whether there is any pooled liquid at the bottom
- whether a meniscus/clear liquid line is visible
- the liquid color (if any) and whether there are bubbles/film on the walls/bottom

Step 2 (contamination decision): Decide if the plastic is contaminated based ONLY on visible residue inside or on it.

IMPORTANT LIQUID RULE (to reduce missed detections):
- If you can see any liquid inside the plastic waste (even if it is clear, dark, translucent, or appears as a meniscus/pooled layer), count that as visible residue.
- Also count any film/stains/residue on the interior walls or bottom (food, oil, organic stains).
- Ignore reflections/lighting artifacts on the outside and ignore physical damage/crushing.

Respond ONLY in JSON with this exact structure (valid JSON).
{
  "image_description": "complete description of what you see inside/on the container",
  "contaminated": true or false,
  "reason": "complete explanation tied to visible evidence"
}"""


def _vlm_mime_for_upload(upload_content_type: str | None) -> str:
    if upload_content_type and upload_content_type.startswith("image/"):
        return upload_content_type
    return "image/jpeg"


def _decode_bgr(image_bytes: bytes) -> NDArray[np.uint8]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes")
    return bgr


def _encode_jpeg_rgb(image_rgb: NDArray[np.uint8], quality: int = 95) -> bytes:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode image to JPEG")
    return buf.tobytes()


def select_vlm_image_bytes(
    image_bytes: bytes,
    model: YOLO,
    req_id: str = "",
    upload_content_type: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """
    Run YOLO-World, take the highest-confidence box, apply padding, crop, and JPEG-encode.
    If no detection or crop is invalid, return original bytes for the VLM.
    """
    t = _phase(req_id, "yolo_worker: enter select_vlm_image_bytes", None)

    meta: dict[str, Any] = {
        "crop_source": "full_image",
        "yolo": None,
    }

    bgr = _decode_bgr(image_bytes)
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = _phase(req_id, "yolo_worker: decode + bgr->rgb done", t)
    img_h, img_w = image_rgb.shape[:2]
    t = _phase(
        req_id,
        f"yolo_worker: calling model.predict (h={img_h} w={img_w} conf={YOLO_CONF})",
        t,
    )

    results = model.predict(image_rgb, conf=YOLO_CONF, verbose=False)

    t = _phase(req_id, "yolo_worker: model.predict returned", t)

    best_box = None
    highest_conf = 0.0
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > highest_conf:
                highest_conf = conf
                best_box = box

    if best_box is None:
        _phase(req_id, "yolo_worker: no box above conf -> full image for VLM", t)
        meta["vlm_mime"] = _vlm_mime_for_upload(upload_content_type)
        return image_bytes, meta

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    class_id = int(best_box.cls[0])
    label = (
        YOLO_CLASSES[class_id]
        if 0 <= class_id < len(YOLO_CLASSES)
        else "unknown"
    )

    width, height = (x2 - x1), (y2 - y1)
    pad_x = int(width * CROP_PADDING)
    pad_y = int(height * CROP_PADDING)

    px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    px2, py2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)

    if px2 <= px1 or py2 <= py1:
        _phase(req_id, "yolo_worker: invalid padded box -> full image for VLM", t)
        meta["vlm_mime"] = _vlm_mime_for_upload(upload_content_type)
        meta["yolo"] = {
            "label": label,
            "confidence": highest_conf,
            "class_id": class_id,
            "note": "invalid_box_after_padding",
        }
        return image_bytes, meta

    crop_rgb = image_rgb[py1:py2, px1:px2]
    if crop_rgb.size == 0:
        _phase(req_id, "yolo_worker: empty crop -> full image for VLM", t)
        meta["vlm_mime"] = _vlm_mime_for_upload(upload_content_type)
        meta["yolo"] = {
            "label": label,
            "confidence": highest_conf,
            "class_id": class_id,
            "note": "empty_crop",
        }
        return image_bytes, meta

    t = _phase(req_id, "yolo_worker: encoding crop jpeg", t)
    crop_bytes = _encode_jpeg_rgb(crop_rgb)
    _phase(
        req_id,
        f"yolo_worker: done crop label={label!r} conf={highest_conf:.3f} vlm_bytes={len(crop_bytes)}",
        t,
    )
    meta["crop_source"] = "yolo_best_detection"
    meta["vlm_mime"] = "image/jpeg"
    meta["yolo"] = {
        "label": label,
        "confidence": highest_conf,
        "class_id": class_id,
        "bbox_xyxy": [x1, y1, x2, y2],
        "crop_xyxy": [px1, py1, px2, py2],
    }
    return crop_bytes, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()
    log.info("ContamiNet: loading YOLO-World (%s)…", YOLO_WEIGHTS)
    model = YOLO(YOLO_WEIGHTS)
    log.info(
        "ContamiNet: YOLO() loaded in %.2fs; calling set_classes…",
        time.monotonic() - t0,
    )
    t1 = time.monotonic()
    model.set_classes(YOLO_CLASSES)
    log.info(
        "ContamiNet: YOLO-World ready (set_classes %.2fs, total %.2fs). "
        "Set CONTAMINET_DEBUG=1 for per-request phase logs.",
        time.monotonic() - t1,
        time.monotonic() - t0,
    )
    app.state.yolo = model

    provider = parse_vlm_provider()
    app.state.vlm_provider = provider
    app.state.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    app.state.ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    app.state.ollama_vlm_model = os.environ.get("OLLAMA_VLM_MODEL", "qwen2.5vl:3b")

    log.info("ContamiNet: VLM provider=%s", provider)
    if provider == "google":
        init_google_backend()
        log.info(
            "ContamiNet: Google Gemini model=%s (API key from GEMINI_API_KEY or GOOGLE_API_KEY)",
            app.state.gemini_model,
        )
    else:
        log.info(
            "ContamiNet: Ollama host=%s model=%s",
            app.state.ollama_host,
            app.state.ollama_vlm_model,
        )

    yield


app = FastAPI(title="ContamiNet VLM API", lifespan=lifespan)


@app.post("/analyze")
async def check_contamination(file: UploadFile = File(...)):
    """
    Accepts an image upload, runs YOLO-World on the best plastic-related detection,
    crops that region when possible, and sends the crop (or full image) to the VLM.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        req_id = uuid.uuid4().hex[:12]
        t = time.monotonic()
        t = _phase(req_id, "analyze: request started", None)

        image_bytes = await file.read()
        t = _phase(req_id, f"analyze: read upload ({len(image_bytes)} bytes)", t)

        model: YOLO = app.state.yolo

        t = _phase(req_id, "analyze: scheduling YOLO in thread pool", t)
        vlm_bytes, yolo_meta = await asyncio.to_thread(
            select_vlm_image_bytes,
            image_bytes,
            model,
            req_id,
            file.content_type,
        )
        t = _phase(
            req_id,
            f"analyze: YOLO thread done crop_source={yolo_meta['crop_source']} vlm_input_bytes={len(vlm_bytes)}",
            t,
        )

        provider = app.state.vlm_provider
        vlm_mime = yolo_meta.get("vlm_mime", "image/jpeg")
        t = _phase(
            req_id,
            f"analyze: calling VLM ({provider}, mime={vlm_mime})…",
            t,
        )
        content = await vlm_generate_json_text(
            provider,
            prompt=CONTAMINATION_PROMPT,
            image_bytes=vlm_bytes,
            image_mime=vlm_mime,
            ollama_host=app.state.ollama_host,
            ollama_model=app.state.ollama_vlm_model,
            gemini_model=app.state.gemini_model,
        )
        t = _phase(req_id, "analyze: VLM returned", t)

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {
                "error": "Model returned invalid JSON",
                "raw_content": content,
                "vlm_provider": provider,
                "crop_source": yolo_meta["crop_source"],
                "yolo": yolo_meta["yolo"],
            }
            return payload

        if isinstance(payload, dict):
            payload["vlm_provider"] = provider
            payload["crop_source"] = yolo_meta["crop_source"]
            payload["yolo"] = yolo_meta["yolo"]

        _phase(req_id, "analyze: response ready", t)
        return payload

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM Error: {str(e)}") from e
