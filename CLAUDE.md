# irem-chan

Discord bot that roleplays **Irem**, a cat-girl from *Eternal Return*, in
jeiss and neotep's server ("na norms"). Python + discord.py + Gemini,
deployed on Railway from GitHub (auto-deploys on push to `main`).

## Running things

```bash
cd /Users/jeiss/irem-chan
venv/bin/python …              # ALWAYS the venv, never bare python3
venv/bin/python check_gemini.py    # is Gemini up today, and on which model?
venv/bin/python test_sleep_sim.py  # 200-day sleep schedule simulation
```

Tests that touch her data should redirect it so they don't clobber real state:

```bash
IREM_DATA_DIR=/tmp/scratch venv/bin/python …
```

To exercise her without Discord, exec the module with the bot startup cut off:

```python
src = open("irem.py").read().split("client.run(")[0]
mod = types.ModuleType("m"); mod.__file__ = os.path.abspath("irem.py")
exec(compile(src, "irem.py", "exec"), mod.__dict__)
await mod.ask_irem(channel_id, "hi irem", author_id, "awake", author_name="Lizaqua")
```

`mod.__file__` matters — module-level code uses it to resolve `DATA_DIR`.

## Layout

- `irem.py` (~2k lines) — everything: personality prompt, Gemini routing,
  media pipeline, commands, Discord handlers
- `sleepy.py` — the sleep cycle state machine and its timezone
- `check_gemini.py` — health probe
- `build_roster.py` / `characters.json` — character-appearance roster
  (**built but NOT wired in**; she still invents names like "Chloe")
- `docs/todo.md`, `docs/memory-system-design.md` — designs not yet built
- `data/` — gitignored; standing orders live here

## Hard-won facts — do not re-derive

**Gemini free-tier quota is PER KEY**, not account-wide. Measured directly:
one key drained on 3.8-flash while nine others answered on it. An earlier
note claimed account-wide; that was inferred from search grounding 429ing
everywhere, and grounding has its own separate quota. Ten keys → ~11,000
requests/day, not ~1,100.

**A 503 is also per key.** Same model, same second: four keys OK, six 503.
"High demand" sounds global and isn't.

**Search grounding does not work on the free tier.** All ten keys 429 with no
quota detail while plain requests on the same key succeed. Removed; Tavily
handles lookups instead. Re-enable only on a paid tier.

**Local tests do not exercise the fetch path.** Everything media-related that
loads files from disk skips `fetch_media_bytes` entirely — which is where the
worst bug of the project lived (`resp.content.read(n)` silently truncating
89% of every image). Test the network path explicitly.

**Check what's actually deployed before believing a fix failed.** Railway has
lagged by a full day. The ACTIVE deployment card names the commit by its
message. Deploy Logs answer questions that guessing doesn't.

## Design principles that came from real bugs

- **Never let a real state look like a crash.** Silence, an empty reply, and
  a canned "i'm sleepy" line are indistinguishable from outside — and three
  separate bugs hid behind that line. If something fails, say something true.
- **Enforce behaviour in code when the prompt won't hold it.** Kaomoji rates,
  the ignore command, standing orders. She will happily *agree* to a
  constraint and then ignore it.
- **You can't scold a model into knowing something.** A 490-token
  anti-hallucination block failed where a 210-token roster of facts worked.
- **Guard the output, not just the input.** `looks_like_reasoning`,
  `MALFORMED_KAOMOJI_RE`, `strip_pingable_syntax` all catch things the prompt
  was already told not to do.
- **Measure before concluding.** Most wrong turns here came from a plausible
  mechanism that was never tested.

## Conventions

- Commit messages explain *why*, with the measured evidence that motivated
  the change (before/after numbers, the log line, the failing case).
- Sizeable edits go through a Python patch script in the scratchpad using
  exact-match `assert s.count(old) == 1` replacements — heredoc quoting has
  caused real breakage; prefer raw triple-quoted strings.
- Comments say what was tried and failed, so it isn't retried.

## People

`DEEP_CONNECTIONS` is exactly **jeiss** and **neotep** — the only two who can
command her (ignore/unignore, standing orders). **neotep is she/her.** Never
infer anyone's gender from a Discord name or avatar; `DEEP_CONNECTION_PRONOUNS`
holds only what jeiss has actually stated, defaulting to they/them.

Regulars: Shingai, Lizaqua, Shiori, Chata, Squortle, CamperOnDuty, Ms Luci.

Her cat form is a cream-and-ginger tabby with a **red collar and a gold
ornament**, and **heterochromia** (one amber eye, one blue) in every form —
that's the discriminator. **Not every cat is her.**

## Known open items

- **No persistence across redeploys.** Standing orders write to `data/`, but
  Railway wipes the filesystem on deploy. Needs a mounted volume at
  `IREM_DATA_DIR`. Same storage the memory system will want.
- **Roster not wired in** — she invents character names.
- **Gemini free tier is heavily congested** (ongoing through late Sept 2026).
  Routing now degrades gracefully, but can't manufacture capacity. Measured
  cost of a paid key at her real volume: ~$2–7/month.
- Lottie stickers (Discord's default packs) are still invisible to her.
