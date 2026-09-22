"""Is Gemini up right now, and which model should she be using?

    venv/bin/python check_gemini.py

Sends ONE tiny request per model across the first few keys and prints a grid.
Deliberately light: a couple of dozen calls, not the 70-slot sweep a real
reply can do, so running it never eats meaningful quota or makes congestion
worse. Read it as:

    all OK            -> Gemini is fine, any slowness is elsewhere
    503               -> that model is overloaded right now (Google's side)
    504 / slow OK     -> congested; she'll route around it but replies drag
    429               -> quota, not an outage -- per key, per model, per day

She walks models top to bottom, so the first row that's mostly OK is the one
she'll settle on.
"""
import os, sys, time
from dotenv import load_dotenv
from google import genai
from google.genai import types as gt, errors as ge

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
          "gemini-3.5-flash", "gemini-3-flash-preview",
          "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
KEYS_TO_TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 3

keys = [os.environ["GEMINI_API_KEY"]]
i = 2
while os.environ.get(f"GEMINI_API_KEY_{i}"):
    keys.append(os.environ[f"GEMINI_API_KEY_{i}"])
    i += 1
keys = keys[:KEYS_TO_TEST]

print(f"probing {len(MODELS)} models x {len(keys)} keys, one small call each\n")
healthy = []
for model in MODELS:
    cells = []
    ok_count = 0
    for n, key in enumerate(keys, 1):
        client = genai.Client(api_key=key, http_options=gt.HttpOptions(
            timeout=20_000, retry_options=gt.HttpRetryOptions(attempts=1)))
        started = time.monotonic()
        try:
            client.models.generate_content(
                model=model, contents="hi",
                config=gt.GenerateContentConfig(max_output_tokens=10))
            took = time.monotonic() - started
            cells.append(f"OK/{took:.0f}s")
            ok_count += 1
        except ge.APIError as e:
            cells.append(f"{e.code}/{time.monotonic() - started:.0f}s")
        except Exception as e:
            cells.append(type(e).__name__[:9])
        time.sleep(0.3)
    if ok_count == len(keys):
        healthy.append(model)
    print(f"  {model:24} {'  '.join(f'{c:>8}' for c in cells)}")

print()
if healthy:
    print(f"  healthy: {', '.join(healthy)}")
    print(f"  she should be answering on {healthy[0]}")
else:
    print("  nothing fully healthy -- expect slow replies or the sleepy line")
