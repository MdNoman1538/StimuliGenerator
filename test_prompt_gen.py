"""
Test the Gemini prompt-translation step (Step 1) for S3,
to see if the paired JSON comes back correctly.
"""

import asyncio
import json
from image_system_prompt import IMAGE_SYSTEM_PROMPT
import gemini_client

# Example S3 sentences (fabricated for testing)
NC_S3 = (
    "To ensure practical manufacturing, structural elements like load-bearing "
    "frameworks, connective interfaces, and reinforcement members can be "
    "configured to guarantee constant dimensional integrity and remarkable "
    "surface uniformity."
)
VC_S3 = (
    "To ensure practical manufacturing, aluminum extrusions like T-slot "
    "framing rails, corner gusset brackets, and carbon-fiber stiffener rods "
    "can be configured to guarantee constant flatness tolerance and remarkable "
    "anodized finish quality."
)


async def main():
    position = "S3"
    user_msg = (
        f"Sentence position: {position}\n"
        f"NC sentence: {NC_S3}\n"
        f"VC sentence: {VC_S3}\n\n"
        f"Return ONE line of JSON: "
        f'{{"nc_prompt": "...", "vc_prompt": "..."}}. '
        f"Both prompts must share identical composition, lighting, framing, "
        f"and supporting elements, and differ ONLY in the specificity of the "
        f"primary objects."
    )

    print("Sending prompt-translation request for S3...")
    print(f"NC: {NC_S3[:80]}...")
    print(f"VC: {VC_S3[:80]}...")
    print()

    try:
        raw = await gemini_client.generate(
            prompt=user_msg,
            system_prompt=IMAGE_SYSTEM_PROMPT,
        )
        print(f"Raw response ({len(raw)} chars):")
        print(raw[:2000])
        print()

        # Try to parse
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

        obj = json.loads(text)
        print("Parsed JSON successfully!")
        print(f"  nc_prompt length: {len(obj.get('nc_prompt', ''))}")
        print(f"  vc_prompt length: {len(obj.get('vc_prompt', ''))}")
        print()
        print("NC prompt preview:")
        print(f"  {obj['nc_prompt'][:300]}...")
        print()
        print("VC prompt preview:")
        print(f"  {obj['vc_prompt'][:300]}...")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
