# Irem-chan long-term memory design

Design notes from an ongoing conversation about giving Irem persistent,
per-person impressions of the people who talk to her — so she remembers
people across sessions and treats them differently based on history,
instead of starting fresh every conversation. **Nothing here is built
yet** — this is the design as discussed so far, to pick back up from.

## Goal

Right now nothing persists at all (a Railway redeploy wipes everything).
The idea: give her a small, curated set of private "impressions" per
person, formed slowly and rarely, that quietly shape her tone toward
them — without ever turning into a list she recites, and without ever
becoming a vector for someone to be treated badly through her.

## Data model (sketch)

- `impressions` table: `id`, `user_id`, `text`, `created_at`. One row
  per impression, not one table per person.
- `user_stats` table: `user_id`, lifetime message count, messages since
  last reflection, `last_interacted_at`.
- Deep connections are **not** a database table — see below.

This is also the first feature that needs real persistence (SQLite on a
volume, or Railway's Postgres addon — not decided yet).

## How impressions form: the reflection pass

- Every ~50 messages a given person has exchanged with her (tracked per
  person, not per channel — a relationship isn't scoped to one channel),
  fire one extra Gemini call.
- Input: her current impression list for that person, plus the actual
  last 50 messages between them (this needs a new per-*person* rolling
  log — her existing conversation history is per-*channel* and mixes
  multiple people together, so it can't be reused directly for this).
- Output: structured JSON, not free text (this is writing to a
  database, so it needs to be reliable). Schema should force an
  explicit `anything_notable: true/false` gate *before* any add/replace
  fields — this measurably reduces the model's tendency to invent
  something just because it was asked.
- **Default outcome is "no change."** The prompt should say this
  explicitly: most 50-message windows won't have anything worth
  recording, and that's normal, not a failure.

### What counts as worth remembering

- Concrete bar for the model: "would you still remember this a year
  from now?" / "does this tell you something about who they are, not
  just what they said?"
- Explicitly excluded: generic pleasantries, one-off topical chat
  (game strategy, memes, etc.) unless it reveals character, and
  anything that just restates an existing impression (check against
  the current list to avoid near-duplicates).
- Impressions should read like her own private, in-character thoughts
  ("always asks about fish, it's kind of sweet") — not clinical
  summaries ("user frequently discusses fish-related topics").
- The cap (see tiers below) is a **ceiling, not a target**. Most people
  should land at maybe 3-8 impressions even after a lot of chatting —
  that's realistic and it's what makes a maxed-out list feel earned
  rather than automatic.
- She never recites this list back verbatim. It only ever surfaces as
  tone — warmer, more familiar, an occasional organic callback in her
  own words. The moment she sounds like she's "reading notes," the
  illusion breaks.

## Two tiers: casual vs. deep connections

- **Casual (default, everyone)**: smaller impression cap (~20). Higher
  bar to change an existing impression of *them* — she should rarely
  change her mind about someone, only when something genuinely
  significant happened.
- **Deep connections**: bigger cap (~40-50). Lower bar to shift her
  view of *them* specifically — closer to how real trust works, where
  one real conversation with someone close can shift your view, versus
  a stranger who has to prove it repeatedly.
- Deep connections are a **manually curated list**, not earned through
  message volume — a hardcoded constant in the source (e.g. a Python
  set of Discord user IDs), edited only by pushing a code change. This
  was a deliberate choice: no Discord command for it, no admin-editable
  DB table — it changes only "from my side," i.e. only when the owner
  asks for a code change.
- Originally discussed as two separate tiers ("deep connections" and
  "parents" as an even-higher tier) — settled on **just one tier,
  called "deep connections."** ("Parents" as a label was deemed weird.)

## Forgetting

- If 2+ weeks pass with no interaction, wipe a person's impressions —
  **unless** they're a deep connection, who's exempt regardless of gap
  length.
- Checked **lazily**, not on a schedule: compare `now` vs.
  `last_interacted_at` the moment they message her again, and wipe
  before generating a reply if the threshold's crossed. No background
  sweep needed — it only ever computes for the one person actually back
  in front of her.
- Open question, not resolved: if someone gets forgotten and later
  comes back and talks a lot, should their lifetime message-count
  progress (irrelevant now that tiers aren't earned by volume, but
  potentially relevant to future tier ideas) reset to zero, or keep
  climbing quietly through the gap? Leaning toward: doesn't matter much
  anymore since tiers are manual now, but worth revisiting if that
  changes.

## Third-party testimony (deep connections shaping opinions of *others*)

- If a deep connection tells her something about a third person Y
  ("Y was really sweet to me" / "Y was kind of rude"), that can shape
  her impression of Y — not just her impression of the deep connection.
  This mirrors how kids' early opinions of unfamiliar people are shaped
  by trusted adults, which fits her established childlike character.
- Weighted by how much *direct* experience she already has with Y:
  - Little/no firsthand history with Y → secondhand testimony can form
    the bulk of her initial impression.
  - Substantial firsthand history with Y → testimony is just one more
    input into the next reflection review, not an automatic override.
    Her own accumulated direct experience outweighs being told
    something, once she has enough of it. (This is the "still its own
    being" property — real influence without erasing her own agency.)
- This is real scope beyond the base system: the reflection pass now
  also needs to watch conversations *with deep connections* for
  mentions of other people, not just self-referential content.

## Safety: not a vector for bullying (non-negotiable)

This was flagged explicitly as the one thing to never compromise on:

- **Hard ceiling on hostility.** No matter how sour an impression gets,
  her actual behavior can only range from genuinely warm down to
  polite-and-less-effusive — never hostile, dismissive, or exclusionary.
  This is already implied by her core character (never mean, sarcastic,
  or crude) — this system must never be allowed to override that floor.
- **Positive and negative testimony are not trusted equally.** Wrong
  positive testimony about a stranger costs little (worst case: she's
  a bit too friendly to someone). Wrong *negative* testimony is the
  actual bullying vector — it could shape how she treats an innocent
  person off one person's claim. So negative claims about a third
  party need a much higher bar than positive ones: ideally corroborated
  across multiple mentions, or backed by her own direct experience, not
  adopted from a single claim.
- **Build skepticism into the reflection prompt itself** — trusting
  someone's character overall shouldn't mean uncritically believing
  everything they say about someone else. The prompt should lean
  toward not updating anything when a claim reads one-sided,
  exaggerated, or like it's trying to turn her against someone, rather
  than just relaying something that happened.
- **The small, hand-picked deep-connections list is itself a
  safeguard**, not just a design choice — this level of influence is
  only ever available to a few explicitly trusted people, which rules
  out the obvious abuse case of a random person racking up interactions
  specifically to gain sway over her opinions.

## Open / not yet decided

- Exact prompt wording for the reflection call.
- Exact JSON schema for reflection output.
- SQLite+volume vs. Postgres for persistence.
- Whether/how to let someone ask her to "forget" them (a privacy
  courtesy mentioned once, not fleshed out).
- This entire feature is still at the design-discussion stage — nothing
  has been implemented, and it hasn't been scoped into an actual build
  plan (file paths, function signatures, migration steps) yet.
