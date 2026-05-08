"""
Quick noun extraction test for NC vs VC parity
"""

import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("ERROR: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    exit(1)

nc_text = """This personal transit system can thoughtfully integrate metabolic output and frictional interface to create inventive energy transfer that effectively facilitates lateral transit without compromising the safety threshold. Designed to handle continuous energetic input, these dynamic configurations seamlessly blend force application and directional bias to optimize uphill spatial displacement and ensure rapid velocity reduction. To ensure practical manufacturing, fundamental properties like propulsion, conversion, and leverage can be configured to guarantee constant temporal reliability and remarkable structural integrity. A highly adaptable interaction boundary allows users to translate physical intent into predictable regulated vectors, actively maximizing user confidence while minimizing safety threshold. Ultimately, by exploring distinct physical principles and accessible spatial resources, the design delivers a reliable, inventive, and truly affordable transport solution."""

vc_text = """This personal transit system can thoughtfully integrate leg drive and snow contact to create inventive propulsive geometry that effectively facilitates icy travel without compromising the balance point. Designed to handle continuous pedal strokes, these articulated assemblies seamlessly blend chain tension and steering angle to optimize uphill vertical ascent and ensure rapid disc braking. To ensure practical manufacturing, structural parts like treads, cranks, and levers can be configured to guarantee constant traction levels and remarkable frame durability. A highly adaptable handlebar assembly allows users to translate arm pressure into predictable tight turns, actively maximizing quick mastery while minimizing balance point. Ultimately, by exploring distinct hardware designs and accessible material stocks, the design delivers a reliable, inventive, and truly affordable winter vehicle."""

def extract_noun_slots(text, label=""):
    """Extract noun phrases from text"""
    doc = nlp(text)
    slots = []

    for chunk in doc.noun_chunks:
        slots.append({
            "phrase": chunk.text.strip(),
            "nouns": [tok.text for tok in chunk if tok.pos_ in {"NOUN", "PROPN"}],
            "lemmas": [tok.lemma_.lower() for tok in chunk if tok.pos_ in {"NOUN", "PROPN"}],
        })

    return slots

# Extract
nc_slots = extract_noun_slots(nc_text, "NC")
vc_slots = extract_noun_slots(vc_text, "VC")

print("=" * 80)
print(f"NC noun phrases: {len(nc_slots)}")
print("=" * 80)
for i, slot in enumerate(nc_slots, 1):
    print(f"{i:2}. {slot['phrase']:40} → {', '.join(slot['nouns'])}")

print("\n" + "=" * 80)
print(f"VC noun phrases: {len(vc_slots)}")
print("=" * 80)
for i, slot in enumerate(vc_slots, 1):
    print(f"{i:2}. {slot['phrase']:40} → {', '.join(slot['nouns'])}")

print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)

# Create comparison table
rows = []
max_slots = max(len(nc_slots), len(vc_slots))

for i in range(max_slots):
    nc = nc_slots[i] if i < len(nc_slots) else None
    vc = vc_slots[i] if i < len(vc_slots) else None

    nc_phrase = nc["phrase"] if nc else "—"
    vc_phrase = vc["phrase"] if vc else "—"
    nc_nouns = len(nc["nouns"]) if nc else 0
    vc_nouns = len(vc["nouns"]) if vc else 0
    match = "✓" if nc_nouns == vc_nouns else "✗"

    rows.append([i+1, nc_phrase, nc_nouns, vc_phrase, vc_nouns, match])

headers = ["Slot", "NC Phrase", "NC#", "VC Phrase", "VC#", "Match"]
# Simple table format
col_widths = [6, 35, 5, 35, 5, 7]
header_line = " | ".join(f"{h:{col_widths[i]}}" for i, h in enumerate(headers))
print(header_line)
print("-" * len(header_line))
for row in rows:
    print(" | ".join(f"{str(val):{col_widths[i]}}" for i, val in enumerate(row)))

print("\n" + "=" * 80)
print(f"SUMMARY: NC has {len(nc_slots)} phrases | VC has {len(vc_slots)} phrases")
matches = sum(1 for i in range(min(len(nc_slots), len(vc_slots)))
              if len(nc_slots[i]["nouns"]) == len(vc_slots[i]["nouns"]))
print(f"Matching noun counts: {matches}/{min(len(nc_slots), len(vc_slots))}")
print("=" * 80)
