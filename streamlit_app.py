"""
ContamiNet UI: upload an image, call the FastAPI /analyze endpoint, show VLM results
and YOLO box/crop overlays when the API returns coordinates.
"""

from __future__ import annotations

import io
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

DEFAULT_API = os.environ.get("CONTAMINET_API_URL", "http://127.0.0.1:8000")


def _annotate_detections(img: Image.Image, yolo: dict[str, Any] | None) -> Image.Image:
    """Draw YOLO tight box (red) and padded crop region (teal) on a copy of the image."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    if not yolo:
        return out

    label = yolo.get("label", "")
    conf = yolo.get("confidence")
    conf_s = f" {conf:.0%}" if isinstance(conf, (int, float)) else ""

    bbox = yolo.get("bbox_xyxy")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = (int(b) for b in bbox)
        draw.rectangle([x1, y1, x2, y2], outline="#e74c3c", width=4)
        text = f"{label}{conf_s}".strip() or "detection"
        ty = max(0, y1 - 22)
        draw.rectangle([x1, ty, x1 + 8 * len(text), ty + 18], fill="#e74c3c")
        draw.text((x1 + 4, ty + 2), text, fill="white")

    crop = yolo.get("crop_xyxy")
    if crop and len(crop) == 4:
        cx1, cy1, cx2, cy2 = (int(c) for c in crop)
        draw.rectangle([cx1, cy1, cx2, cy2], outline="#00cec9", width=3)

    return out


def _extract_crop(img: Image.Image, crop_xyxy: list[Any] | None) -> Image.Image | None:
    if not crop_xyxy or len(crop_xyxy) != 4:
        return None
    x1, y1, x2, y2 = (int(c) for c in crop_xyxy)
    w, h = img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))


def _post_analyze(api_base: str, file_bytes: bytes, filename: str, content_type: str | None) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/analyze"
    ct = content_type if content_type and content_type.startswith("image/") else "image/jpeg"
    files = {"file": (filename or "upload.jpg", file_bytes, ct)}
    r = requests.post(url, files=files, timeout=600)
    if r.status_code != 200:
        try:
            body = r.json()
            detail = body.get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"HTTP {r.status_code}: {detail}")
    return r.json()


st.set_page_config(page_title="ContamiNet", layout="wide")

st.title("ContamiNet")
st.caption("Upload a photo of plastic waste; the API runs YOLO-World + a VLM and returns a contamination assessment.")

with st.sidebar:
    api_base = st.text_input("API base URL", value=DEFAULT_API, help="FastAPI server root, e.g. http://127.0.0.1:8000")
    st.markdown("Start the API with: `fastapi run main.py`")

uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "gif", "bmp"])

if st.button("Analyze", type="primary", disabled=uploaded is None):
    assert uploaded is not None
    raw = uploaded.getvalue()
    name = uploaded.name or "upload.jpg"
    ctype = uploaded.type

    with st.spinner("Calling /analyze…"):
        try:
            data = _post_analyze(api_base, raw, name, ctype)
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    if "error" in data:
        st.error(data.get("error", "Unknown error"))
        if data.get("raw_content"):
            st.code(data["raw_content"], language="text")
        st.json(data)
        st.stop()

    contaminated = data.get("contaminated")
    desc = data.get("image_description", "")
    reason = data.get("reason", "")
    crop_source = data.get("crop_source", "")
    yolo = data.get("yolo")
    provider = data.get("vlm_provider", "")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if contaminated is True:
            st.error("**Contaminated**")
        elif contaminated is False:
            st.success("**Not contaminated**")
        else:
            st.warning("**Classification unclear**")
    with c2:
        st.metric("VLM provider", provider or "—")
    with c3:
        st.metric("YOLO crop", crop_source.replace("_", " ") if crop_source else "—")

    st.subheader("Image description")
    st.write(desc or "—")

    st.subheader("Reasoning")
    st.write(reason or "—")

    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        st.warning(f"Could not open image for preview: {e}")
        st.stop()

    st.subheader("Image & detection")
    st.caption("Red box: YOLO detection (tight). Teal box: padded region sent to the VLM (when available).")

    if yolo and yolo.get("bbox_xyxy") and yolo.get("crop_xyxy"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(_annotate_detections(img, yolo), caption="Original with boxes", use_container_width=True)
        with col_b:
            crop_img = _extract_crop(img, yolo.get("crop_xyxy"))
            if crop_img:
                st.image(crop_img, caption="Region sent to VLM (padded crop)", use_container_width=True)
            else:
                st.info("Could not extract crop preview.")
    elif yolo and (yolo.get("note") or yolo.get("label")):
        st.image(img, caption="Original (YOLO had a detection but no usable crop coordinates)", use_container_width=True)
        st.info(f"YOLO: **{yolo.get('label', '?')}** — {yolo.get('note', 'no box drawn')}")
    else:
        st.image(img, caption="Original (full image used; no single YOLO crop)", use_container_width=True)
        if crop_source == "full_image":
            st.info("No YOLO box above threshold (or crop invalid); the VLM saw the full upload.")
