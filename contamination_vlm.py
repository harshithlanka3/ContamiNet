import ollama

def check_contamination(image_path):
    """
    Analyzes an image to detect if a plastic container is contaminated.
    """
    prompt = """Analyze this plastic item for recycling. Look ONLY for visual evidence of residue (food, oil, liquid, or organic stains).

CRITICAL RULE: Plastic is NOT contaminated unless you see visible residue inside or on it.

Respond ONLY in JSON with this exact structure: {"contaminated": true or false, "reason": "short explanation"}"""

    try:
        response = ollama.chat(
            model='moondream',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path] # The library handles path-to-base64 for you
            }],
            format='json' # Forces the model to output valid JSON
        )
        
        return response['message']['content']
    
    except Exception as e:
        return f"Error: {str(e)}"

# Example Usage
if __name__ == "__main__":
    result = check_contamination('istockphoto-157616664-612x612.jpg')
    print(result)