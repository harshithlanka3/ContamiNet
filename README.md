# ContamiNet API

HTTP API for upstream plastic recycling contamination detection using a **vision-language model (VLM)**. The backend is selected at **process startup** with `CONTAMINET_VLM_PROVIDER`:

- **`ollama`** (default): local [Ollama](https://ollama.com/) with a vision model (e.g. `qwen2.5vl:3b`).
- **`google`**: [Google AI Studio](https://aistudio.google.com/) / Gemini over HTTPS using an API key (no local Ollama required for inference).

YOLO-World still runs locally for detection/cropping in both modes.

## Prerequisites

- Python 3.10+ recommended
- **If `CONTAMINET_VLM_PROVIDER=ollama` (default)**  
  - Ollama running (default host `http://127.0.0.1:11434`)  
  - Pull the vision model (must match `OLLAMA_VLM_MODEL`, default `qwen2.5vl:3b`):

```bash
ollama pull qwen2.5vl:3b
```

- **If `CONTAMINET_VLM_PROVIDER=google`**  
  - Create an API key in Google AI Studio and export it as **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`**.  
  - **`GEMINI_MODEL`** defaults to **`gemini-2.5-flash-lite`** (better fit for the **free tier** than `gemini-2.0-flash`, which often reports **zero quota**). Override with `gemini-2.5-flash` or another [supported model](https://ai.google.dev/gemini-api/docs/models) if needed.  
  - Free tier has **strict RPM/RPD limits**; if you hit **429**, wait for the suggested retry window or switch to a lighter model.

- **YOLO-World weights**: on first run, Ultralytics can download `yolov8s-world.pt` (or set `YOLO_WORLD_WEIGHTS` to a local path).

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `fastapi[standard]`, `ollama`, `google-generativeai`, `ultralytics`, `opencv-python-headless`, and `numpy`.

## Run the server

From the repository root.

**Ollama (default):**

```bash
fastapi run main.py
```

**Google AI Studio (Gemini):**

```bash
export CONTAMINET_VLM_PROVIDER=google
export GEMINI_API_KEY="your-key"   # or GOOGLE_API_KEY
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
7. The crop (or full image) is sent to the configured VLM:
   - **Ollama**: `AsyncClient.chat(..., format="json")` with `OLLAMA_VLM_MODEL` (default `qwen2.5vl:3b`).
   - **Google**: Gemini `generate_content` with `response_mime_type="application/json"` and `GEMINI_MODEL` (default **`gemini-2.5-flash-lite`**). The image is sent with MIME type `image/jpeg` for YOLO crops, or the upload’s `Content-Type` when the full image is used.
8. The JSON response includes the VLM fields when parsing succeeds, plus pipeline metadata:
   - **`vlm_provider`**: `"ollama"` or `"google"`.
   - **`crop_source`**: `"yolo_best_detection"` if a crop was sent to the VLM, or `"full_image"` if the whole upload was used.
   - **`yolo`**: when a best box was chosen, an object with `label`, `confidence`, `class_id`, and box coordinates (`bbox_xyxy`, `crop_xyxy`); otherwise `null` (or includes a `note` when a box existed but cropping failed).

If the model returns non-JSON text, the handler may return `error` and `raw_content` (and still attach `vlm_provider` / `crop_source` / `yolo` when the payload is a dict). Malformed image bytes yield **400**. Gemini **quota / rate limit** responses yield **429**. Other VLM errors yield **500** with a `detail` string.

### Environment variables (optional)

| Variable | Default | Meaning |
|----------|---------|---------|
| `CONTAMINET_VLM_PROVIDER` | `ollama` | `ollama` or `google` (aliases treated as Google: `gemini`, `google_ai`, `aistudio`, `google-ai-studio`). |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL (Ollama mode only). |
| `OLLAMA_VLM_MODEL` | `qwen2.5vl:3b` | Ollama vision model name. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | unset | **Required** for `CONTAMINET_VLM_PROVIDER=google` (either variable is accepted). |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model id for Google mode (free-tier friendly). |
| `YOLO_WORLD_WEIGHTS` | `yolov8s-world.pt` | Weights path or model name for Ultralytics |
| `YOLO_WORLD_CONF` | `0.05` | Minimum confidence for detections |
| `YOLO_CROP_PADDING` | `0.15` | Fractional pad applied to each side of the box before cropping |
| `CONTAMINET_DEBUG` | unset | Set to `1` / `true` / `yes` to log **per-request phase timings** (upload read → YOLO → VLM). Helps find stalls (e.g. long gap before `model.predict returned` vs before `VLM returned`). |

**Startup logs** (always on, `INFO`): time to load `YOLO()` and `set_classes`, plus which **VLM provider** is active. First boot may spend a long time **downloading YOLO weights**.

In **Ollama** mode, Ollama must be reachable at **`OLLAMA_HOST`**. In **Google** mode, outbound HTTPS to Google’s API is required; Ollama does not need to be running.

## Endpoint reference

### `POST /analyze`

| Item | Value |
|------|--------|
| Content-Type | `multipart/form-data` |
| Field name | `file` (image file) |
| Success body | JSON object: VLM fields plus `vlm_provider`, `crop_source`, and `yolo` (see above) |
| 400 | Not an image, undecodable image, or bad input (`detail` message) |
| 429 | Gemini rate limit or quota exceeded (`detail` includes retry hints) |
| 500 | VLM provider error or unexpected server error (`detail` message) |

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
