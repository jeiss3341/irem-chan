"""One-time-ish helper: turn a folder of NNN_Name.png character art into a
compact name -> appearance roster, by asking Gemini to describe each one.

The roster is what lets her identify characters the model has never heard of
(most of the Eternal Return cast). Rebuilding is cheap and only needs doing
when new characters are added, so it deliberately skips anything already in
the output file rather than re-describing the whole set every run.

    python build_roster.py ~/Downloads
"""
import json
import os
import re
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ROSTER_PATH = os.path.join(os.path.dirname(__file__), "characters.json")
CHARACTER_FILE_RE = re.compile(r"^(\d{3})_(.+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)

# Lite first: describing a picture is mechanical work that doesn't need a
# smart model, and the Lite pools are the big ones.
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash"]

PROMPT = (
    "Describe this character's appearance in ONE sentence, under 25 words, listing only "
    "the details that would let someone pick them out of a lineup: hair colour and style, "
    "eye colour, outfit, and any signature item or weapon. No name, no lore, no commentary."
)

_keys = [os.environ["GEMINI_API_KEY"]]
_i = 2
while os.environ.get(f"GEMINI_API_KEY_{_i}"):
    _keys.append(os.environ[f"GEMINI_API_KEY_{_i}"])
    _i += 1
_clients = [genai.Client(api_key=k) for k in _keys]


def describe(image_bytes, mime):
    parts = [{"text": PROMPT}, {"inline_data": {"mime_type": mime, "data": image_bytes}}]
    for model in MODELS:
        for client in _clients:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[{"role": "user", "parts": parts}],
                    config=types.GenerateContentConfig(max_output_tokens=120),
                )
                text = (resp.text or "").strip()
                if text:
                    return text
            except genai_errors.APIError:
                continue
    return None


def main(folder):
    roster = {}
    if os.path.exists(ROSTER_PATH):
        with open(ROSTER_PATH) as f:
            roster = json.load(f)

    for filename in sorted(os.listdir(folder)):
        match = CHARACTER_FILE_RE.match(filename)
        if not match:
            continue
        name = match.group(2).replace("_", " ")
        if name in roster:
            print(f"  skip {name} (already described)")
            continue
        path = os.path.join(folder, filename)
        with open(path, "rb") as f:
            data = f.read()
        mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        description = describe(data, mime)
        if description is None:
            print(f"  FAILED {name} (no capacity)")
            continue
        roster[name] = description
        print(f"  {name}: {description}")

    with open(ROSTER_PATH, "w") as f:
        json.dump(roster, f, indent=2, sort_keys=True)
    print(f"\n{len(roster)} characters -> {ROSTER_PATH}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads"))
