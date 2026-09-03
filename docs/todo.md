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
