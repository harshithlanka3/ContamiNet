import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from ollama import AsyncClient

app = FastAPI(title="ContamiNet VLM API")

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


@app.post("/analyze")
async def check_contamination(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns a contamination assessment from the VLM.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        client = AsyncClient(host="http://127.0.0.1:11434")

        response = await client.chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": CONTAMINATION_PROMPT,
                    "images": [image_bytes],
                }
            ],
            format="json",
        )

        content = response["message"]["content"]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Model returned invalid JSON", "raw_content": content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM Error: {str(e)}")
