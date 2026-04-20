import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from numpy.typing import NDArray
from ultralytics import YOLO

from vlm_providers import (
    GeminiQuotaOrRateLimit,
    init_google_backend,
    parse_vlm_provider,
    vlm_generate_json_text,
)

log = logging.getLogger(__name__)

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
    "plastic bottle",
    "plastic film",
    "plastic bag",
    "",
]

CONTAMINATION_PROMPT = """Analyze this plastic cup/container image for recycling contamination.

Step 1 (image description): Describe what is visibly inside the cup/container/plastic container/bag/plastic waste interior. Include:
- whether there is any pooled liquid at the bottom
- whether a meniscus/clear liquid line is visible
- the liquid color (if any) and whether there are bubbles/film on the walls/bottom
- explicitly note if the ONLY moisture you see is small **clear** beads/droplets or light **condensation** (no tint, no pool, no film)

Step 2 (contamination decision): Set **contaminated** using the rules below. Base it ONLY on the interior/on-container evidence you described.

**ACCEPTABLE (set contaminated to false):**
- **Trace clear water only:** scattered **tiny clear** droplets clinging to walls/bottom, and/or **light condensation**, with **NO** pooled liquid, **NO** continuous meniscus/liquid line, **NO** color or cloudiness in the liquid, **NO** oily sheen, **NO** food/drink film, **NO** suds/bubbles from soap or beverage.
- A water bottle or cup that is **essentially empty** with only this kind of cling moisture is **NOT** contaminated for sorting purposes.

**NOT acceptable (set contaminated to true):**
- Any **pooled** layer, **meniscus**, or **connected wet patch** that acts like a liquid surface (even if clear).
- **Tinted, cloudy, or colored** liquid or film; **bubbles/suds**; **oil/grease**; **milk/juice/soda/food** residue or stains; **slime** or **sticky** film.
- A **continuous hazy or frosted wet film** over a large area (not the same as a few separate tiny droplets).

**Critical instruction (reduces false positives on water bottles):**
- **Do NOT set contaminated=true only because you see small clear water droplets or condensation.** If your description lists **only** trace clear water / condensation and none of the “NOT acceptable” items above, you **must** set **contaminated** to **false** and say so in **reason**.
- If something might be water vs sugary drink residue: if it is **clear, not sticky-looking, and not forming a pool or film**, treat it as acceptable trace water.

Ignore reflections/lighting artifacts on the outside and ignore physical damage/crushing.

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
    upload_content_type: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """
    Run YOLO-World, take the highest-confidence box, apply padding, crop, and JPEG-encode.
    If no detection or crop is invalid, return original bytes for the VLM.
    """
    meta: dict[str, Any] = {
        "crop_source": "full_image",
        "yolo": None,
    }

    bgr = _decode_bgr(image_bytes)
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    results = model.predict(image_rgb, conf=YOLO_CONF, verbose=False)

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
        meta["vlm_mime"] = _vlm_mime_for_upload(upload_content_type)
        meta["yolo"] = {
            "label": label,
            "confidence": highest_conf,
            "class_id": class_id,
            "note": "empty_crop",
        }
        return image_bytes, meta

    crop_bytes = _encode_jpeg_rgb(crop_rgb)
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
    log.info("ContamiNet: loading YOLO-World (%s)…", YOLO_WEIGHTS)
    model = YOLO(YOLO_WEIGHTS)
    model.set_classes(YOLO_CLASSES)
    log.info("ContamiNet: YOLO-World ready")
    app.state.yolo = model

    provider = parse_vlm_provider()
    app.state.vlm_provider = provider
    # Default favors AI Studio free tier; 2.0-flash often shows 0 quota on free tier.
    app.state.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
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
        image_bytes = await file.read()

        model: YOLO = app.state.yolo

        vlm_bytes, yolo_meta = await asyncio.to_thread(
            select_vlm_image_bytes,
            image_bytes,
            model,
            file.content_type,
        )

        provider = app.state.vlm_provider
        vlm_mime = yolo_meta.get("vlm_mime", "image/jpeg")
        content = await vlm_generate_json_text(
            provider,
            prompt=CONTAMINATION_PROMPT,
            image_bytes=vlm_bytes,
            image_mime=vlm_mime,
            ollama_host=app.state.ollama_host,
            ollama_model=app.state.ollama_vlm_model,
            gemini_model=app.state.gemini_model,
        )

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

        return payload

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except GeminiQuotaOrRateLimit as e:
        raise HTTPException(
            status_code=429,
            detail=(
                "Gemini rate limit or quota exceeded (common on the free tier). "
                "Wait and retry, try GEMINI_MODEL=gemini-2.5-flash-lite or gemini-2.5-flash, "
                "or enable billing for higher limits. Raw: "
                + str(e)
            ),
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM Error: {str(e)}") from e
