# Irem-chan — to do

Small, scoped follow-ups that aren't urgent enough to build right now.

## Link understanding

She can already see images, GIFs, and videos people post (implemented —
see `extract_image_parts` in `irem.py`; GIFs are resampled into a few PNG
frames since Gemini has no native GIF support, and Tenor/Giphy GIFs shared
via Discord's picker are pulled from the message embed, not an attachment).
Links are a separate, harder piece: having her actually fetch and react to
what's behind a URL someone posts, not just the raw link text.

Why it's harder than images: it means the bot fetching the page itself,
and several real sites already turned out to block or time out automated
fetches during testing (namu.wiki needed a browser workaround instead of a
plain fetch, Twitter/X is fully login-walled, Best Buy timed out). Any
implementation needs to fail gracefully — react to the link normally
without content, or just say something in-character — rather than error
out when a site blocks it.

Not started yet.

## Standing orders from jeiss/neotep

**Built 2026-09-11.** jeiss/neotep can give her instructions that stick, and
she follows them until told otherwise instead of agreeing sweetly and
carrying on. Measured before: told "stop saying meow" she said "Okay, I won't
say it anymore!" and then said Meow three times running; told "no more food
from shingai" she argued back and thanked him for a cookie a minute later.

How it works:

- `ORDER_HINT_RE` is a cheap gate so ordinary chat never costs an API call.
  Only messages that look like an instruction reach `_classify_order`, a
  small JSON pass that decides order vs. comment and rewrites it as one short
  rule. Keyword matching alone can't separate "stop saying meow" from "haha
  you never stop saying meow", and a false positive is the bad direction.
- Rules live in `data/standing_orders.json` and are injected into the system
  prompt only when the list is non-empty.
- "what are your rules" lists them; "forget rule 2" and "forget all your
  rules" remove them. Cap of 10, oldest dropped.
- She may protest — pout, ask why, call it unfair — but not disobey. Grumbling
  while obeying is fine; agreeing and then not doing it is not.
- The cruelty and genuine-concern guardrails from `IREM_SYSTEM_PROMPT` are
  restated inside the orders block so an order can't quietly erase them.

This also needed `author_name` in `ask_irem`. The model never learned who was
speaking, so a rule naming a person could not fire at all — with the shingai
rule loaded she still answered "Yay, thank you so much!" to his cookie.

**Remaining:** Railway wipes the container filesystem on redeploy, so rules
survive restarts but not deploys. Mounting a volume at `IREM_DATA_DIR` makes
them permanent — the same storage the memory system below will want.

## Character roster so she can name what she sees

She can see and describe images accurately now, but can't name most
Eternal Return characters — Gemini simply wasn't trained on them (asked
directly, with an explicit "reply UNKNOWN if you don't know" escape hatch,
it answers UNKNOWN for Lumi). No amount of prompt wording fixes that; the
knowledge has to be supplied.

Groundwork is done: [build_roster.py](../build_roster.py) describes each
`NNN_Name.png` character art file into
[characters.json](../characters.json) as one sentence of distinguishing
visual features, skipping any already described. On three test images the
roster took identification from 1/3 to 3/3 (Coraline and Henry both went
from invented names to correct ones) at ~210 tokens for six characters.

Remaining work: get the rest of the character art into `~/Downloads` and
rerun the script, then wire the roster into a separate identification call
— image + roster + "which of these is this, or unknown", with no
personality in it — and feed just the resulting name into her normal reply.
Keeping it in its own call is the point: her own prompt never grows, and
the mechanical matching can run on a Lite model where the quota is cheap.

Storage is a file today; moving it into the database is a ten-line change
if that's preferred, since the data shape is the same either way.

## Database + real memory of people

The bigger piece — full design already written up in
[memory-system-design.md](memory-system-design.md): persistent per-person
impressions (SQLite on a Railway volume, or their Postgres addon — not
decided), a periodic reflection LLM pass that forms them, and using them so
she actually remembers people/relationships instead of the current static
`DEEP_CONNECTIONS` stand-in ([irem.py](../irem.py)) which only kicks in
when jeiss/neotep are the one talking or being talked about, with nothing
persisted and nothing for anyone else.

You said you'd set up the actual database yourself. Nothing built yet.
