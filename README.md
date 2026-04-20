# ContamiNet API

HTTP API for upstream plastic recycling contamination detection using a vision-language model (VLM) hosted in [Ollama](https://ollama.com/).

## Prerequisites

- Python 3.10+ recommended
- [Ollama](https://ollama.com/) running locally (default: `http://127.0.0.1:11434`)
- Pull the model used by the API (must match the name in `main.py`):

```bash
ollama pull qwen2.5vl:3b
```

- **YOLO-World weights**: on first run, Ultralytics can download `yolov8s-world.pt` (or set `YOLO_WORLD_WEIGHTS` to a local path).

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `fastapi[standard]` (CLI and server), `ollama`, `ultralytics`, `opencv-python-headless`, and `numpy`.

## Run the server

From the repository root:

```bash
fastapi run main.py
```

This loads the `app` object from `main.py` and serves the API (default bind is shown in the CLI output; use `--host` / `--port` as needed).

For local development with auto-reload:

```bash
fastapi dev main.py
```

Equivalent with Uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## How the API works

1. The client sends a **multipart form** POST to `/analyze` with a single file field named `file` containing an image.
2. The server checks that the upload’s `Content-Type` is an image (`image/*`). If not, it responds with **400**.
3. **YOLO-World (startup)**: a `YOLO(YOLO_WORLD_WEIGHTS)` model is loaded once and configured with the same open-vocabulary class list used on `js/VLM-implementation` in `yolo-world-test.py` (plastic containers, bags, liquid-in-plastic, etc.).
4. **Per request**: the upload is decoded, YOLO-World runs at confidence `YOLO_WORLD_CONF` (default `0.05`, matching the script’s low threshold). Among all boxes, the **highest-confidence** detection is kept. That box is expanded by **`YOLO_CROP_PADDING`** (default `0.15` of box width/height), clipped to image bounds, cropped, and re-encoded as JPEG for the VLM.
5. **Fallback**: if there is **no** detection above the threshold, the **box is degenerate** after padding/clipping, or the **crop is empty**, the API sends the **original full image bytes** to the VLM instead (same behavior as “no useful crop”).
6. YOLO runs in a **worker thread** (`asyncio.to_thread`) so the async server stays responsive while inference runs.
7. The crop (or full image) is sent to Ollama with **`qwen2.5vl:3b`** and `format="json"`. The contamination prompt is unchanged (`CONTAMINATION_PROMPT` in `main.py`).
8. The JSON response includes the VLM fields when parsing succeeds, plus pipeline metadata:
   - **`crop_source`**: `"yolo_best_detection"` if a crop was sent to the VLM, or `"full_image"` if the whole upload was used.
   - **`yolo`**: when a best box was chosen, an object with `label`, `confidence`, `class_id`, and box coordinates (`bbox_xyxy`, `crop_xyxy`); otherwise `null` (or includes a `note` when a box existed but cropping failed).

If the model returns non-JSON text, the handler may return `error` and `raw_content` (and still attach `crop_source` / `yolo` when the payload is a dict). Malformed image bytes yield **400**. Ollama/network errors yield **500** with a `detail` string.

### Environment variables (optional)

| Variable | Default | Meaning |
|----------|---------|---------|
| `YOLO_WORLD_WEIGHTS` | `yolov8s-world.pt` | Weights path or model name for Ultralytics |
| `YOLO_WORLD_CONF` | `0.05` | Minimum confidence for detections |
| `YOLO_CROP_PADDING` | `0.15` | Fractional pad applied to each side of the box before cropping |

Ollama must be reachable at **`http://127.0.0.1:11434`** unless you change `AsyncClient(host=...)` in `main.py`.

## Endpoint reference

### `POST /analyze`

| Item | Value |
|------|--------|
| Content-Type | `multipart/form-data` |
| Field name | `file` (image file) |
| Success body | JSON object: VLM fields plus `crop_source` and `yolo` (see above) |
| 400 | Not an image, undecodable image, or bad input (`detail` message) |
| 500 | VLM/Ollama or unexpected server error (`detail` message) |

### Example with `curl`

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@test_images/24oz_blueberry_72.jpg"
```

Replace host/port with wherever your FastAPI process is listening.

### OpenAPI

Interactive docs are served by FastAPI at **`/docs`** (Swagger UI) and **`/redoc`** when the app is running.

## `test_images/`

Sample images for manual testing or demos. They are not required at runtime except for your own requests.
