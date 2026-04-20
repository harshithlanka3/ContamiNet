# ContamiNet API

HTTP API for upstream plastic recycling contamination detection using a vision-language model (VLM) hosted in [Ollama](https://ollama.com/).

## Prerequisites

- Python 3.10+ recommended
- [Ollama](https://ollama.com/) running locally (default: `http://127.0.0.1:11434`)
- Pull the model used by the API (must match the name in `main.py`):

```bash
ollama pull qwen2.5vl:3b
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `fastapi[standard]`, which includes the FastAPI CLI (`fastapi run` / `fastapi dev`) and a production-capable server stack.

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
3. The image bytes are read in memory and sent to Ollama’s chat API using the **async** Python client (`ollama.AsyncClient`), model **`qwen2.5vl:3b`**, with `format="json"` so the model is steered toward JSON output.
4. The prompt asks the model to (a) describe visible contents/residue inside or on the plastic container and (b) decide if the item is **contaminated** for recycling, with strict rules about liquids and interior film/stains (see `CONTAMINATION_PROMPT` in `main.py`).
5. On success, the handler returns parsed JSON (typically `image_description`, `contaminated`, `reason`). If the model returns non-JSON text, the API may return a small object with `error` and `raw_content`. On upstream failures (Ollama/network/model errors), the API responds with **500** and a `detail` string.

Ollama must be reachable at **`http://127.0.0.1:11434`** unless you change `AsyncClient(host=...)` in `main.py`.

## Endpoint reference

### `POST /analyze`

| Item | Value |
|------|--------|
| Content-Type | `multipart/form-data` |
| Field name | `file` (image file) |
| Success body | JSON object (see above) |
| 400 | Not an image (`Content-Type` not `image/*`) |
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
