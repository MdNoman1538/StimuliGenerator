"""
Quick standalone test: generate a single image of a cow using the same
Gemini image-generation setup as the main app.
"""

from google import genai
from config import get_gemini_key, IMAGEN_MODEL

def main():
    key = get_gemini_key()
    if not key:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return

    print(f"Using model: {IMAGEN_MODEL}")
    print(f"API key loaded: {key[:8]}...")

    client = genai.Client(api_key=key)

    prompt = "A simple photograph of a cow standing in a green field under blue sky."
    print(f"Prompt: {prompt}")
    print("Sending request...")

    try:
        response = client.models.generate_content(
            model=IMAGEN_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
    except Exception as e:
        print(f"\nREQUEST FAILED with exception:\n  Type: {type(e).__name__}\n  Message: {e}")
        return

    print(f"\nResponse received.")
    print(f"  candidates: {len(response.candidates) if response.candidates else 0}")

    if not response.candidates:
        print("  No candidates returned.")
        print(f"  Full response: {response}")
        return

    candidate = response.candidates[0]
    print(f"  finish_reason: {getattr(candidate, 'finish_reason', 'N/A')}")

    if not candidate.content or not candidate.content.parts:
        print("  No content parts.")
        print(f"  candidate: {candidate}")
        return

    part = candidate.content.parts[0]
    if hasattr(part, "inline_data") and part.inline_data is not None:
        data = part.inline_data.data
        out_path = "test_cow.png"
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"  SUCCESS — image saved to {out_path} ({len(data)} bytes)")
    else:
        print(f"  Part has no inline_data. Part type: {type(part)}")
        print(f"  Part: {part}")


if __name__ == "__main__":
    main()
