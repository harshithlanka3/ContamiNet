import json

import ollama


def check_contamination(image_path):
    """
    Analyzes an image to detect if a plastic container is contaminated.
    """
    prompt = """Analyze this plastic cup/container image for recycling contamination.

Step 1 (image description): Describe what is visibly inside the cup/container interior. Include:
- whether there is any pooled liquid at the bottom
- whether a meniscus/clear liquid line is visible
- the liquid color (if any) and whether there are bubbles/film on the walls/bottom

Step 2 (contamination decision): Decide if the plastic is contaminated based ONLY on visible residue inside or on it.

IMPORTANT LIQUID RULE (to reduce missed detections):
- If you can see any liquid inside the cup (even if it is clear, dark, translucent, or appears as a meniscus/pooled layer), count that as visible residue.
- Also count any film/stains/residue on the interior walls or bottom (food, oil, organic stains).
- Ignore reflections/lighting artifacts on the outside and ignore physical damage/crushing.

Respond ONLY in JSON with this exact structure (valid JSON). Use the full text
needed—do not truncate image_description or reason.
{
  "image_description": "complete description of what you see inside/on the container",
  "contaminated": true or false,
  "reason": "complete explanation tied to visible evidence"
}"""

    try:
        response = ollama.chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ],
            format="json",
        )

        content = response["message"]["content"]
        try:
            return json.loads(content)
        except Exception:
            return content

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    image_file = "d26mbvplp2p61.jpg"
    result = check_contamination(image_file)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result)
