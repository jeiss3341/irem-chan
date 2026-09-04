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

Let jeiss and neotep give her instructions that actually stick — "stop
saying meow", "be quieter in #general", "talk more" — and have her follow
them **permanently**, until one of them says otherwise. Not just for the
next reply, and not just until the channel history rolls over.

That "permanently" is what makes this a real feature rather than a prompt
tweak: it needs persistence, so it wants the same storage as the memory
system below (a Railway redeploy currently wipes everything). Worth doing
both at once rather than standing up storage twice.

Things to decide when building it:

- **Recognising a command.** Distinguishing "stop saying meow" (an order)
  from "haha you say meow a lot" (a comment). Probably model judgement in a
  structured JSON pass rather than keyword matching, similar to the
  reflection pass below. Getting this wrong in the false-positive direction
  is worse — she'd silently adopt rules nobody meant to give her.
- **Scope.** Global, or per-channel? "Be quieter in #general" implies
  per-channel is at least sometimes wanted.
- **Listing and revoking.** There has to be a way to see what's currently
  in force and remove one, or they'll accumulate invisibly and she'll drift
  for reasons nobody can trace. A cap plus a "what are your rules right
  now" answer would cover it.
- **Conflicts.** Later orders should presumably override earlier
  contradictory ones rather than both sitting in the prompt fighting.
- **Interaction with the existing guardrail.** `IREM_SYSTEM_PROMPT` already
  says closeness with deep connections is "not blind agreement" and that
  she should respond like a caring friend if something seems genuinely
  worrying, no matter who said it. Standing orders shouldn't quietly erase
  that.

Only jeiss/neotep (`DEEP_CONNECTIONS`) should be able to set these.

Not started yet.

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
