import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from numpy.typing import NDArray
from ollama import AsyncClient
from ultralytics import YOLO

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


def select_vlm_image_bytes(image_bytes: bytes, model: YOLO) -> tuple[bytes, dict[str, Any]]:
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
        meta["yolo"] = {
            "label": label,
            "confidence": highest_conf,
            "class_id": class_id,
            "note": "invalid_box_after_padding",
        }
        return image_bytes, meta

    crop_rgb = image_rgb[py1:py2, px1:px2]
    if crop_rgb.size == 0:
        meta["yolo"] = {
            "label": label,
            "confidence": highest_conf,
            "class_id": class_id,
            "note": "empty_crop",
        }
        return image_bytes, meta

    crop_bytes = _encode_jpeg_rgb(crop_rgb)
    meta["crop_source"] = "yolo_best_detection"
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
    model = YOLO(YOLO_WEIGHTS)
    model.set_classes(YOLO_CLASSES)
    app.state.yolo = model
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
            select_vlm_image_bytes, image_bytes, model
        )

        client = AsyncClient(host="http://127.0.0.1:11434")

        response = await client.chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": CONTAMINATION_PROMPT,
                    "images": [vlm_bytes],
                }
            ],
            format="json",
        )

        content = response["message"]["content"]

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {
                "error": "Model returned invalid JSON",
                "raw_content": content,
                "crop_source": yolo_meta["crop_source"],
                "yolo": yolo_meta["yolo"],
            }
            return payload

        if isinstance(payload, dict):
            payload["crop_source"] = yolo_meta["crop_source"]
            payload["yolo"] = yolo_meta["yolo"]

        return payload

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM Error: {str(e)}") from e
