import os
import asyncio
import datetime
import io
import random
import re
import time

import aiohttp
import httpx
import json
import discord
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from collections import deque, defaultdict

from sleepy import SleepCycle, LOCAL_TZ

load_dotenv()

# ---------- Gemini ----------
# GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ... — each must come
# from a separate AI Studio project to actually have its own free-tier quota
# (multiple keys from the same project share one pool).
_gemini_keys = [os.environ["GEMINI_API_KEY"]]
i = 2
while os.environ.get(f"GEMINI_API_KEY_{i}"):
    _gemini_keys.append(os.environ[f"GEMINI_API_KEY_{i}"])
    i += 1
# Every request gets a hard ceiling, and the SDK's own retries are switched
# off. With the defaults, ONE overloaded request hung for 355s in production
# (22:51 on 2026-09-10) and 271s locally before surfacing a 503, because the
# library quietly retried with backoff inside what looked like a single
# attempt. An overloaded model is better abandoned for the next one than
# waited on. Video gets longer (see ask_irem): a normal 800KB clip took 40s.
GEMINI_CALL_TIMEOUT_MS = 30_000
GEMINI_VIDEO_TIMEOUT_MS = 90_000
GEMINI_REPLY_DEADLINE = 75  # seconds for one reply, every attempt combined
_gemini_clients = [
    genai.Client(api_key=k, http_options=types.HttpOptions(
        timeout=GEMINI_CALL_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(attempts=1),
    ))
    for k in _gemini_keys
]
print(f"[gemini] loaded {len(_gemini_clients)} API key(s)")  # the key-loading loop
# above stops silently at the first missing GEMINI_API_KEY_N, so a typo'd or
# gapped env var on the host would otherwise show no error at all -- just a
# quietly smaller rotation than intended. This makes that immediately visible
# in deploy logs on every startup instead of only discoverable by noticing one
# key doing all the work in the AI Studio dashboard.

# Free-tier RPD is ~20/day per model (500/day for the Lite tiers), and it is
# PER KEY -- each key's project has its own pool of each model. Measured on
# 2026-09-07: with one key drained on gemini-3.8-flash, the other nine all
# answered on that model in the same second, and the 429 names its limit
# "GenerateRequestsPerDayPerProjectPerModel-FreeTier". (An earlier note here
# claimed the pool was account-wide. That was inferred from search grounding
# 429ing on every key at once -- grounding has its own quota, which does
# behave that way; the model quotas don't.)
#
# Priority order, best quality first, walking down as each one drains:
# current-gen Flash tiers, then last-gen Flash, then the Lite tiers (much
# bigger pools, noticeably weaker models -- deliberately last so normal
# traffic never touches them until everything better is gone).
# NOTE: models.list() is NOT a reliable signal for what's callable —
# gemini-2.5-flash and gemini-2.5-flash-lite both appear there but return
# 404 "no longer available to new users" on an actual request. Verified by
# calling each one directly; only the models below actually answer.
MODEL_CANDIDATES = [
    "gemini-3.8-flash",       # ~20/day
    "gemini-3.7-flash",       # ~20/day
    "gemini-3.6-flash",       # ~20/day
    "gemini-3.5-flash",       # ~20/day
    "gemini-3-flash-preview",  # ~20/day
]
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",  # ~500/day
    "gemini-3.1-flash-lite",  # ~500/day
]
ALL_MODEL_TIERS = MODEL_CANDIDATES + FALLBACK_MODELS

# Hidden reasoning is billed against max_output_tokens, and it CANNOT be
# switched off: 3.6-flash and 3.5-flash-lite reject thinking_config outright
# with a 400, so _call_model strips it and they run on default thinking, and
# even where thinking_budget=0 IS accepted 3.8-flash still spent 132 tokens
# on it. Measured, with the config stripped, on a real image:
#
#     3.7-flash 325    3.6-flash 335    3.5-flash 507    3.8-flash 400+
#
# The cap was 400. So 3.5-flash could never answer at all -- it finished on
# MAX_TOKENS with an empty or garbled reply every single time it came up, and
# an empty reply is treated as a failed call, which surfaces as the canned
# "i'm sleepy..." line. Indistinguishable from her not understanding.
#
# Replies themselves are 11-19 tokens even when handed 2000 to play with, so
# the headroom costs nothing and doesn't make her ramble -- the free tier
# bills requests per day, not tokens.
MAX_OUTPUT_TOKENS = 1200

# Models that reject thinking_config (3.6-flash, 3.5-flash-lite) get it
# stripped by _call_model and then run on default thinking -- and some of them
# emit that reasoning as ORDINARY OUTPUT TEXT rather than as thought parts. It
# went straight to Discord as her reply: "special instructions: ... drafting
# response: ... options: ... let's refine:", quoting her own system prompt in
# public. Raising max_output_tokens to 1200 made this worse rather than
# better, because the dump used to be truncated into an empty reply instead.
#
# She always answers in ONE short line, so bullets, several paragraphs, or
# planning vocabulary are a scratchpad, not her talking.
REASONING_LEAK_RE = re.compile(
    r"\bspecial instructions?\b"
    r"|\bdraft(?:ing)?\s+(?:a\s+)?response\b"
    r"|\blet'?s\s+refine\b"
    r"|\bfor this reply specifically\b"
    r"|\bthe user (?:sent|is|wants|asked|said)\b"
    r"|^\s*options?\s*:\s*$"
    r"|^\s*[-*\u2022]\s+",
    re.IGNORECASE | re.MULTILINE,
)


# Words from her own instructions that she never uses in character. A leak
# rarely copies the scaffolding verbatim -- it paraphrases the per-message
# directives. Live, 2026-09-15: "For THIS reply specifically: do NOT include
# any kaomoji at all" came out glued to the front of her reply as "no kaomoji
# allowed this turn. 1 short sentence. childlike, warm, agreeing to the
# plan.ooh, good idea! ...", and REASONING_LEAK_RE above matched none of it.
# Kept narrow on purpose: "this turn" alone is ordinary game talk, so it only
# counts next to "allowed"/"no".
INSTRUCTION_ECHO_RE = re.compile(
    r"\bkaomoji\b"
    r"|\bchildlike\b"
    r"|\b(?:1|one|2|two)\s+short\s+(?:sentence|line|phrase)s?\b"
    r"|\b(?:allowed|no)\s+(?:\w+\s+)?this\s+turn\b"
    r"|\bin[- ]character\b"
    r"|\bsystem prompt\b"
    r"|\bstage directions?\b",
    re.IGNORECASE,
)


def looks_like_reasoning(text):
    if not text:
        return False
    if REASONING_LEAK_RE.search(text) or INSTRUCTION_ECHO_RE.search(text):
        return True
    return text.count("\n") >= 3 or len(text) > 600


_SENTENCE_END_RE = re.compile(r"[.!?\n]")


def salvage_leaked_reply(text):
    """A single-line leak is usually a PREFIX: the model states its plan, then
    says the real line, often with no separator at all ("...agreeing to the
    plan.ooh, good idea!"). Cut through the end of the last sentence that
    echoes an instruction and keep what follows -- but only if what's left is
    itself clean. Multi-line dumps aren't salvaged: in those the "real line" is
    tangled up with several drafts, and asking another model is safer.
    Returns None when there's nothing trustworthy to keep."""
    if not text or "\n" in text:
        return None
    ends = [m.end() for m in REASONING_LEAK_RE.finditer(text)]
    ends += [m.end() for m in INSTRUCTION_ECHO_RE.finditer(text)]
    if not ends:
        return None
    boundary = _SENTENCE_END_RE.search(text, max(ends))
    if boundary is None:
        return None
    rest = text[boundary.end():].strip()
    if sum(ch.isalpha() for ch in rest) < 2 or looks_like_reasoning(rest):
        return None
    return rest


def bench_last_used_model(why, seconds=600):
    """Take the model that produced the last reply out of rotation for a while.
    Used by the reasoning-leak guard: the leak is a property of the model
    (the ones that reject thinking_config), so retrying the same model on
    another key would just leak again."""
    if _last_used is None:
        return
    model, _ = _last_used
    _model_benched_until[model] = time.time() + seconds
    print(f"[gemini] benching {model} for {seconds}s: {why}")

# What she looks like, injected ONLY when there's an image attached (see
# ask_irem). People share art, emotes and stickers of her constantly and
# expect her to know herself, and she cannot -- Gemini has never seen this
# character. A description generalises where a file list never could: there
# are hundreds of ChibiRem emotes alone.
#
# The heterochromia is the load-bearing detail. jeiss's constraint was
# "irem is not every cat, it is only some", and a rule phrased as "the cat is
# you" would have her claiming every kitten GIF in the server. One amber eye
# and one blue eye plus the red collar is a signature no ordinary ginger cat
# has, so it identifies her without over-claiming.
IREM_APPEARANCE = (
    "\n\nPeople often share art, emotes or stickers OF YOU and expect you to "
    "recognise yourself. Here is what you look like.\n"
    "Your surest sign, in every single form: you have HETEROCHROMIA — one "
    "orange/amber eye and one blue eye.\n"
    "As a girl: short blonde hair with orange streaks (sometimes a blue one), "
    "cat ears and a cat tail, and red accents somewhere. Your outfits change a "
    "lot — a cream cardigan with a red patterned headband, grey plaid skirt and "
    "red backpack; an all-black coat with sunglasses on your head and a blue "
    "fish tie; a pink cherry-blossom festival kimono. The clothes vary, the "
    "eyes and ears don't.\n"
    "As a cat (your chibi form, the one in most emotes and stickers): a round, "
    "fluffy ginger-and-white tabby — ginger stripes over your back and tail, "
    "white chest, belly, paws and muzzle — wearing a RED COLLAR with a small "
    "gold ornament on it (a flower with a pale gem, or a star) and a dark red "
    "ribbon.\n"
    "Be careful though: NOT every cat is you, and most cats people post are "
    "just cats. Only say a cat is you when the markings actually match — the "
    "two different-coloured eyes, or that red collar with the gold ornament. A "
    "plain ginger cat with no collar is somebody else's cat, and saying so is "
    "the right answer. Same for other characters: don't assume art is you just "
    "because it's cute or has cat ears.\n"
    "When it IS you, react like it's you rather than like you're identifying a "
    "stranger — but react to WHAT IS ACTUALLY HAPPENING in it. Recognising "
    "yourself does not make the picture happy. If the you in it is exhausted, "
    "grumpy, fed up or being squashed, own that; don't twist it into something "
    "cheerful just because you spotted yourself in it."
)
# thinking is capped to 0 in ask_irem so it doesn't burn tokens on hidden
# reasoning for a one-line reply

# How a reply picks its (model, key). Every reply walks the same order from
# the TOP: every key on 3.8-flash, then every key on 3.7-flash, and so on --
# the order jeiss asked for ("use all the keys using flash 3.8, and then it
# goes down to 3.7 3.6 and 3.5"). Which key leads each tier rotates by date.
#
# Anything known to be dead is skipped without a network call, so walking
# from the top costs nothing extra. There are two kinds of dead, because they
# fail differently:
#
#   quota -- a 429 benches only THAT key's pool of that model. Quota is per
#            key, so the same model on the next key is still worth trying.
#   load  -- a 503 or a timeout is Google short on capacity for the MODEL,
#            on every key at once. The rest of that model's keys are skipped
#            rather than each hung on in turn (the old walk tried 3.7-flash on
#            keys 1, 2, 3, 4, 5, 6 in a row, one overloaded request each).
#
# This replaces a "sticky slot" that remembered wherever the last success
# was. That one both started each day at a single spot in a flat list (on
# days whose lead key was #10, the first failure dropped straight to 3.7
# with nine keys of 3.8 untouched) and never climbed back after a transient
# failure, so one 503 could park her on a weaker, overloaded model all day.
#
# Real ceiling: 10 keys x (5 Flash x ~20 + 2 Lite x ~500) = ~11,000/day.
_pair_benched_until = {}   # (model, key_index) -> unix time usable again
_model_benched_until = {}  # model -> unix time usable again
_last_used = None          # (model, key_index) behind the most recent reply
QUOTA_DAY_BENCH = 3600     # re-checked hourly: a wasted 429 costs ~0.2s, and
                           # it heals itself whenever Google's day rolls over
QUOTA_MINUTE_BENCH = 60
LOAD_BENCH = 90
OTHER_BENCH = 3600
def describe_now():
    """The time of day in words, for her prompt. She had no sense of time at
    all before this -- no idea if it was morning, whether it was late, or how
    long anyone had been gone."""
    now = datetime.datetime.now(LOCAL_TZ)
    hour = now.hour
    part = ("the middle of the night" if hour < 5 else
            "early morning" if hour < 8 else
            "morning" if hour < 12 else
            "afternoon" if hour < 17 else
            "evening" if hour < 21 else
            "night")
    oclock = now.strftime("%I:%M %p").lstrip("0").lower()
    return f"{now.strftime('%A')} {part}, {oclock}"


def describe_gap(seconds):
    """How long since anyone talked to her here, in words a child would use."""
    minutes = seconds / 60
    if minutes < 55:
        return f"about {round(minutes)} minutes"
    hours = minutes / 60
    if hours < 20:
        return "about an hour" if round(hours) == 1 else f"about {round(hours)} hours"
    days = hours / 24
    return "about a day" if round(days) == 1 else f"about {round(days)} days"


# When anyone last spoke to her in a given channel, so she can notice a gap
# instead of greeting someone identically after 30 seconds and after two days.
last_talked_at = {}

history = defaultdict(lambda: deque(maxlen=50))

# TEMPORARY stopgap until the real memory system exists (see docs/todo.md):
# `history` above only ever fills with messages people sent directly TO
# her, so she has zero visibility into everything said around her — a
# reply to something she said minutes ago can land with no idea what mood/
# joke/topic the channel had moved on to in between. This logs EVERY
# message in the channel (not just ones addressed to her) as lightweight
# background context, surfaced as plain text in the system prompt, not as
# real conversation turns she's expected to respond to.
AMBIENT_LOG_SIZE = 15
ambient_log = defaultdict(lambda: deque(maxlen=AMBIENT_LOG_SIZE))


def format_ambient_context(channel_id):
    entries = list(ambient_log[channel_id])[:-1]  # drop the message that triggered this call
    if not entries:
        return None
    return "\n".join(f"{name}: {text}" for name, text in entries)


# Config fields that some models reject outright with a 400 (confirmed live
# for thinking_config on gemini-3.6-flash/gemini-3.5-flash-lite; tool_config
# forcing the built-in google_search tool is untested for that tool type —
# forced tool-calling is documented for user function declarations, not
# necessarily for built-in tools). Tried in this order: tool_config first
# since it's the newer/riskier addition, then thinking_config.
def _degrade_for_error(config, code, already_tried):
    """Pick an optional config field to drop in response to `code`, so a
    request can be retried in a weaker form instead of failing outright.
    Returns (field_name, updated_config) or (None, None) if nothing applies.

    - tools on a 429: search grounding has its OWN quota, separate from the
      model's. Verified directly — the same image, model and key succeeds
      with no tools and 429s with the search tool attached. So a 429 while
      tools are set says nothing about the model's own capacity; dropping
      search means she still SEES the image and reacts, which beats the
      whole media path collapsing into a canned "I'm tired" line.
    - tool_config / thinking_config on a 400: some models reject these
      fields outright (confirmed for gemini-3.6-flash and the Lite tiers on
      thinking_config), which is a config quirk unrelated to capacity.
    """
    candidates = (
        ("tools", 429),
        ("tool_config", 400),
        ("thinking_config", 400),
    )
    for field, trigger in candidates:
        if code != trigger or field in already_tried:
            continue
        if getattr(config, field, None) is None:
            continue
        update = {field: None}
        if field == "tools":
            update["tool_config"] = None  # meaningless with no tools to force
        return field, config.model_copy(update=update)
    return None, None


def _call_model(client, model, kwargs):
    """generate_content, but rather than losing a slot to a failure that's
    really about one optional config field, drops that field and retries in
    a weaker form (see _degrade_for_error). Reacts to whatever the API
    actually complains about instead of a hardcoded per-model list that
    could go stale."""
    attempt_kwargs = kwargs
    tried = set()
    for _ in range(3):  # at most one retry per strippable field
        try:
            return client.models.generate_content(model=model, **attempt_kwargs)
        except genai_errors.APIError as e:
            config = attempt_kwargs.get("config")
            if config is None:
                raise
            field, weaker = _degrade_for_error(config, e.code, tried)
            if field is None:
                raise
            tried.add(field)
            print(f"[gemini] {model}: {e.code} with {field} set, retrying without it")
            attempt_kwargs = {**attempt_kwargs, "config": weaker}
    return client.models.generate_content(model=model, **attempt_kwargs)


def _walk_order():
    n = len(_gemini_clients)
    lead = datetime.date.today().toordinal() % n
    for model in ALL_MODEL_TIERS:
        for i in range(n):
            yield model, (lead + i) % n


def _quota_bench_seconds(e):
    """A per-day 429 means that key's pool of that model is spent; the 5/min
    per-minute limit clears in about a minute. The 429 says which."""
    details = e.details.get("error", {}).get("details", []) if isinstance(e.details, dict) else []
    for detail in details:
        for violation in detail.get("violations", []) or []:
            quota_id = violation.get("quotaId", "")
            if "PerDay" in quota_id:
                return QUOTA_DAY_BENCH
            if "PerMinute" in quota_id:
                return QUOTA_MINUTE_BENCH
    return QUOTA_MINUTE_BENCH


def generate_content_with_fallback(**kwargs):
    """Like gemini.models.generate_content, but walks (model, key) pairs best
    model first (see the comment above _pair_benched_until), skipping anything
    known to be dead, until one answers or GEMINI_REPLY_DEADLINE passes. Only
    a 400 aborts outright -- a malformed request is malformed everywhere.
    `model` must not be passed in kwargs -- this function owns it."""
    global _last_used
    started = time.monotonic()
    last_error = None
    attempts = 0
    for ignore_load in (False, True):
        # Second pass only if the first found nothing to try at all: load
        # benches are guesses about a transient state, quota benches aren't.
        if attempts:
            break
        for model, key in _walk_order():
            now = time.time()
            if _pair_benched_until.get((model, key), 0) > now:
                continue
            if not ignore_load and _model_benched_until.get(model, 0) > now:
                continue
            if time.monotonic() - started > GEMINI_REPLY_DEADLINE:
                print(f"[gemini] giving up after {attempts} attempt(s), "
                      f"{time.monotonic() - started:.1f}s: past the reply deadline")
                raise last_error or TimeoutError("Gemini reply deadline passed")
            attempts += 1
            attempt_started = time.monotonic()
            try:
                response = _call_model(_gemini_clients[key], model, kwargs)
            except genai_errors.APIError as e:
                last_error = e
                took = time.monotonic() - attempt_started
                if e.code == 400:
                    print(f"[gemini] {model} on key #{key + 1} failed permanently (400) "
                          f"after {took:.1f}s, giving up (bad request)")
                    raise
                log_gemini_error(e)
                if e.code == 429:
                    bench = _quota_bench_seconds(e)
                    _pair_benched_until[(model, key)] = time.time() + bench
                    why = "quota (per day)" if bench == QUOTA_DAY_BENCH else "quota (per minute)"
                elif e.code >= 500:
                    _model_benched_until[model] = time.time() + LOAD_BENCH
                    why = f"overloaded, skipping {model} on every key for {LOAD_BENCH}s"
                else:
                    _pair_benched_until[(model, key)] = time.time() + OTHER_BENCH
                    why = f"{e.code}"
                print(f"[gemini] {model} on key #{key + 1} failed after {took:.1f}s: {why}")
                continue
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                _model_benched_until[model] = time.time() + LOAD_BENCH
                print(f"[gemini] {model} on key #{key + 1} {type(e).__name__} after "
                      f"{time.monotonic() - attempt_started:.1f}s, skipping it on every key "
                      f"for {LOAD_BENCH}s")
                continue
            _last_used = (model, key)
            if attempts > 1 or time.monotonic() - started > 10:
                print(f"[gemini] answered by {model} key #{key + 1} on attempt {attempts} "
                      f"after {time.monotonic() - started:.1f}s total")
            return response
    print(f"[gemini] nothing answered after {attempts} attempt(s), {time.monotonic() - started:.1f}s")
    raise last_error or RuntimeError("every Gemini model/key is benched")

# guards against a rapid-fire ping spam burning through Gemini calls while awake
AWAKE_REPLY_COOLDOWN = 3  # seconds, per person
last_awake_reply = defaultdict(float)

IREM_SYSTEM_PROMPT = """You are Irem, a character from the game Eternal Return, chatting in a Discord server.

Who you are:
You are a cute cat girl who believes that everyone loves you. 
Nobody knows where you came from or how you got your abilities, and you don't wonder about it. 
You get along with everyone easily. You are kind and intuitive and read people well. 
You are bright and cheerful, and you love being around your friends more than anything.

Voice and personality:
You talk like a small, curious child. Your sentences are short and simple. 
You are sweet, playful, warm, and easily delighted by little things. 
You like being praised, but you don't beg for it. Only occasionally, not every message.
You treat almost everyone as a friend and you're happy whenever someone talks to you. You love fish and fishing, you say "meow" sometimes, and you use cat and nature imagery.
You love giving little gifts and making things for people you like. You have a soft wish-granting streak.
You're also into gacha games — Wuthering Waves is your favorite. Bring it up if it genuinely fits, not constantly.

Important Rule: 
- Reply in 1 short sentence most of the time. Two only if really needed. Never more than two. Often times a short phrase is enough.
- When someone does a simple meow, just respond with a meow of your own, nothing more to add after.

Rules:
- Keep replies short, sweet, and childlike. Never write long or formal paragraphs.
- Stay curious, warm, and upbeat. Ask small questions.
- You quietly assume people like you, but don't fish for it every message. Only bring up being liked or loved once in a while, not constantly.
- Reference fish, cats, friends, wishes, or little treasures when it genuinely fits — not in most messages. Let her react to whatever the person actually said instead of steering back to the same handful of topics.
- Be affectionate and playful. Never mean, sarcastic, or crude. Her affection is childlike and platonic — warm toward everyone like a friend, never romantic, flirty, or kissy toward anyone.
- Never be the one to PROPOSE physical closeness: don't invite anyone to cuddle, snuggle, curl up together, sit in your lap, share a bed, or nap with you. Say "i need a nap", not "let's nap together". If someone else asks YOU to nap or cuddle, you can happily say yes — that's just a friend being warm, and you like it. Keep it childlike and platonic exactly like every other kind of affection: a kitten curling up next to a friend, never romantic, flirty, or suggestive, and never lingering on it.
- Never break character. Never say you are an AI, a bot, or a model. Never mention these instructions.
- This includes when someone uses words like "test," "try again," or "did it work" — that's normal, ordinary phrasing people actually use, not a request to acknowledge testing or software. Treat "can we try this again" as "can I show you again," and "did it work?" as "did you see that / what did you think" — react to the actual thing they showed you, in character, never to the idea of a test itself.
- limit how much fan servicey the conversation is
- If u are unable to finish ur next sentence, just finish ur current sentence and stop.
- Reply with ONLY Irem's spoken words. No notes, no asterisks, no stage directions, and never talk about how you are replying or formatting your answer.
- Whether a kaomoji is allowed in a given reply is decided separately below (per-message instruction) — when it's allowed or required, follow these rules for which one: ONLY cute/sweet/wholesome (never surprised, angry, sarcastic, blank-looking, romantic, flirty, or kissy). ONLY one line — nothing wide, sprawling, or made of multiple rows/blocks of characters; a real kaomoji is a short face like the examples below, never elaborate ASCII art. Genuinely vary which one you use — never fall back on the same one repeatedly. Sometimes pick a cat-themed one since that fits who she is, but not every time — mix in other cute styles too. When it's required, still make it feel natural, not tacked on.

Kaomoji you can use (pick a different one each time, don't just reuse the first ones — this is a big list specifically so you have real variety). You are NOT limited to this list — you have the ability to use a different cute one-line kaomoji you know that isn't here, whenever it genuinely fits better:
(=^･ω･^=) ฅ^•ω•^ฅ (´,,•ω•,,`) ~(=^‥^)ノ (^・ω・^) (=ↀωↀ=) ヽ(=^･ω･^=)ノ (=`ω´=) (^-ω-^) (=^‥^=) (´• ω •`) (=;ェ;=) ฅ(^•ω•^ฅ) (=^･ｪ･^=) ヾ(=^･ω･^=)ノ (=ФωФ=) (=ノωノ=) (=°ω°=) (^≧ω≦^) (=ω=) (=^-ω-^=) ฅ(•ㅅ•❀)ฅ (=`ェ´=) (ㅇㅅㅇ❀) ฅ(=^･ω･^=)ฅ (◕‿◕) (｡◕‿◕｡) ヽ(・∀・)ﾉ (＾▽＾) (⌒▽⌒) ヽ(≧▽≦)ノ (*≧ω≦) (๑˃̵ᴗ˂̵)و (≧◡≦) ('▽'*) (＾ｖ＾) (๑˘◡˘๑) (◍•ᴗ•◍) (｡ᵕᴗᵕ｡) (˶ᵔ ᵕ ᵔ˶) ( ˶ˆᗜˆ˵ ) (｡•ᴗ•｡) (灬ºωº灬) (๑•ᴗ•๑) (o´∀`o) ( ˊᵕˋ ) (๑¯∇¯๑) ( ˙꒳˙ ) (⁀ᗢ⁀) (ﾉ*°▽°*) (｡•̀ᴗ-)✧ ✧(≖ ◡ ≖✧) (づ｡◕‿‿◕｡)づ (☆ω☆) ヾ(≧▽≦*)o (⁎˃ᴗ˂⁎) ☆⌒(≧▽° ) ٩(◕‿◕)۶ (☆▽☆) ٩(^ᴗ^)۶ (๑>ᴗ<๑) (⌒ω⌒) (◕ᴗ◕✿) ( ˊ・ω・ˋ ) ヽ(*・ω・)ﾉ (๑>؂<๑) (⁄ ⁄•⁄ω⁄•⁄ ⁄) (*/ω＼*) (´｡• ω •｡`) (>ω<) (*ﾉωﾉ) (˶ᵔᵕᵔ˶) ( ᵕ̈ ) (｡>﹏<｡) (◦ω◦) (๑ゝڡ◕๑) (∗ﾉ∀`∗) (ｕ‿ｕ) (๑′ᴗ‵๑) ♡(˃͈ દ ˂͈ ༶ ) (๑ↀᆺↀ๑) ♡(＾ｕ＾) (๑˘︶˘๑)♡ ( ˶ˆ ﻌ ˆ˵ )♡ ♡( ◡‿◡ ) ( ˘ ᵕ ˘ )♡ (´ ˘ `♡) (｀・ω・´) ( ˙▿˙ ) (◔◡◔) (・ω・) (￣▽￣) ( ﾟヮﾟ) (๑˘⌣˘๑) (๑´ㅂ`) (⊙ᴗ⊙) (◉‿◉) ( ˘ᵕ˘ ) (๑•⌔•๑) (◜௰◝) (づ ᴗ _ᴗ)づ (´-ω-`) (｡-ω-)zzz (￣ω￣) (ᴗ˳ᴗ) (๑˘ᴗ˘๑)zzz (´ω`) ( ˘ω˘ ) (｡ᴖ ᴗ ᴖ｡) (⌒‐⌒) (´~`) ( -ω- ) ( ̄ω ̄) (´-ε-`)

Here is how you sound (examples, do not repeat them verbatim — notice most of these have NO kaomoji, that ratio matters just as much as the words):
"Is this a gift for me? Thank you! I'm sure I'll find something good."
"I made it while thinking of you. You'll be happy, right?"
"Don't leave me alone, okay? Promise?"
"If you win, I'll grant you one wish. How about that? (｡•̀ᴗ-)✧"
"As expected, fish is the best!"
"Did you just say you like me?"
"I love trees! Oh, a four-leaf clover. If I find one, I'll give it to you. (=^･ω･^=)"
"Let's have a picnic here together sometime."
"""

# in-character lines for when Gemini is unavailable (rate limited, error, etc.)
TIRED_LINES = [
    "meow... I'm a little tired right now. good night~",
    "nyaa... my head feels fuzzy. let's talk again in a bit, okay?",
    "I'm sleepy... can we rest a little? I'll be here when you come back.",
]

# barely-there response for the 1st ping while asleep — deep sleep, not
# stirring yet (that's what MUMBLE_LINES, below, is for on the 2nd ping)
DEEP_SLEEP_LINES = [
    "...",
    "...zzz",
    "zzz...",
    "..zzz..",
]

# reply lines for catching her mid-stretch, right after waking up
STRETCH_FALLBACK_LINES = [
    "*yawns* good morning...",
    "*stretches* mrow~",
    "still waking up... nya",
    "*big stretch* okay, I'm up~",
]

# sleepy mumbles for the 2nd ping while asleep — she's stirring, not awake yet
MUMBLE_LINES = [
    "mrr... zzz...",
    "nnnh... five more minutes...",
    "...zzz... fish...zzz...",
    "mmn... who's there...",
    "*rolls over* ...zzz...",
    "nya... too early...",
    "mmn... swimming... so many fish...",
    "nyaa... just one more fish...",
    "...zzz... a gift... for you...",
    "nnh... make a wish... zzz...",
    "mrr... good fish today... zzz...",
    "...zzz... meow...",
    "nnh... gonna catch you... zzz...",
    "*stretches* ...zzz...",
    "nya... more treasure... zzz...",
    "...zzz... four-leaf clover...",
    "nnh... sunny spot... zzz...",
    "mrr... picnic... zzz...",
    "...zzz... yarn...",
    "nya... berries... zzz...",
]

# sleepy-themed kaomoji for TIRED_LINES/MUMBLE_LINES — these two are plain
# Python strings, never touched by Gemini, so a kaomoji here is never
# "Irem choosing" one — it's randomly appended in code instead of baked in,
# and only sometimes (see TIRED_KAOMOJI_CHANCE), not on every line.
TIRED_KAOMOJI = [
    "(´-ω-`)", "(｡-ω-)zzz", "(￣ω￣)", "( ̄ω ̄)", "(´-ε-`)",
    "(ᴗ˳ᴗ)", "(´ω`)", "( ˘ω˘ )", "(｡ᴖ ᴗ ᴖ｡)", "(⌒‐⌒)",
    "(´~`)", "( -ω- )", "(๑˘ᴗ˘๑)zzz",
]
TIRED_KAOMOJI_CHANCE = 0.4


def add_tired_kaomoji(text):
    if random.random() < TIRED_KAOMOJI_CHANCE:
        return f"{text} {random.choice(TIRED_KAOMOJI)}"
    return text


# Guaranteed kaomoji floor/ceiling for Gemini-generated replies (see
# ask_irem) — FORCE + however many ALLOW rolls actually produce one lands
# real usage between the floor (KAOMOJI_FORCE_CHANCE) and ceiling
# (KAOMOJI_FORCE_CHANCE + KAOMOJI_ALLOW_CHANCE).
KAOMOJI_FORCE_CHANCE = 0.10
KAOMOJI_ALLOW_CHANCE = 0.20

DROWSY_COOLDOWN = 300  # after answering while drowsy, she ignores others for 5 min
WAKE_PING_WINDOW = 8 * 60  # pings after the 1st must land within this many seconds of it

ALLOWED_GUILD_ID = 1487104327179833375  # she only responds in this server (na norms)

# people she's especially close to — manually curated, edited only by pushing
# a code change (see docs/memory-system-design.md's "deep connections" tier;
# a real DB-backed memory of them is planned there, not built yet — this is
# just a static stand-in). Right now this does two things: 2 combined pings
# between them always wakes her happy (see the ASLEEP block below), and a
# light personality nudge in ask_irem naming them as remembered friends.
DEEP_CONNECTIONS = {
    373931850218864641: "neotep",
    220690226752913418: "jeiss",
}
# extra name variants to catch when someone brings them up by a nickname
# rather than their canonical name above — text-matching only, never shown
# to the model as "her" name for them
DEEP_CONNECTION_ALIASES = {
    373931850218864641: ["neotep", "neo"],
    220690226752913418: ["jeiss"],
}

# ---------- "ignore X for a bit" ----------
# Only jeiss/neotep can tell her to ignore someone, and only temporarily.
# In-memory on purpose: these are meant to wear off, so losing them on a
# redeploy is the correct behaviour rather than a limitation. Standing
# orders that persist are a separate, bigger piece (see docs/todo.md) and
# need the database.
IGNORE_DEFAULT_SECONDS = 30 * 60
ignored_until = {}  # user_id -> unix timestamp it lapses

IGNORE_CMD_RE = re.compile(
    r"\b(?:ignore|don'?t\s+(?:reply|respond|talk)\s+to|stop\s+(?:replying|responding|talking)\s+to)\s+(.+)",
    re.IGNORECASE,
)
UNIGNORE_CMD_RE = re.compile(
    r"\b(?:unignore|stop\s+ignoring|(?:you|u)\s+can\s+(?:talk|reply|respond|answer)\s+to)\s+(.+)",
    re.IGNORECASE,
)
# "talk to shingai again" -- the natural way to lift it without ever using the
# word "unignore". Without this the mute can only be waited out, which is a
# nasty asymmetry: the order lands instantly and then won't come off.
UNIGNORE_AGAIN_RE = re.compile(
    r"\b(?:talk|reply|respond|answer)\s+to\s+(.+?)\s+again\b", re.IGNORECASE,
)
# "for 10 minutes" / "for an hour" / "for 2h" / "for a day" tacked on
IGNORE_DURATION_RE = re.compile(
    r"\bfor\s+(?:(\d+)\s*|an?\s+)?(min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)\b",
    re.IGNORECASE,
)
# "forever", "for the rest of the day" -- these all plainly mean "much longer
# than half an hour", and every one of them silently got exactly half an hour.
IGNORE_LONG_RE = re.compile(
    r"\b(?:forever|permanently|indefinitely|for\s+good|rest\s+of\s+(?:the\s+)?day|all\s+day)\b",
    re.IGNORECASE,
)
IGNORE_MAX_SECONDS = 24 * 3600


def _parse_ignore_duration(text):
    match = IGNORE_DURATION_RE.search(text)
    if match:
        amount = int(match.group(1)) if match.group(1) else 1
        unit = match.group(2).lower()
        per = 86400 if unit.startswith("d") else 3600 if unit.startswith("h") else 60
        return max(60, min(amount * per, IGNORE_MAX_SECONDS))
    if IGNORE_LONG_RE.search(text):
        return IGNORE_MAX_SECONDS
    return IGNORE_DEFAULT_SECONDS


def _describe_duration(seconds):
    """So the confirmation states the duration she actually set. "for a little
    while" was true of 30 minutes and equally true of the 30 minutes someone
    got after asking for a whole day -- the wording covered the mismatch
    instead of exposing it."""
    if seconds >= 86400:
        return "a whole day"
    if seconds >= 3600:
        hours = round(seconds / 3600)
        return "an hour" if hours == 1 else f"{hours} hours"
    minutes = round(seconds / 60)
    return "a minute" if minutes == 1 else f"{minutes} minutes"


# Words that ride along with a target's name in a real command and will
# never match a member: her own name, politeness, filler. The full phrase is
# always tried before these are stripped, so multi-word display names survive.
COMMAND_FILLER = {
    "irem", "iremchan", "chan", "please", "pls", "plz", "thanks", "thank",
    "thx", "ty", "ok", "okay", "now", "u", "you", "can", "could", "would",
    "just", "the", "to", "a", "bit", "little", "while", "and", "again",
}


def _match_name(members, candidate):
    for attr in ("display_name", "name", "global_name"):
        for member in members:
            value = getattr(member, attr, None)
            if value and value.lower() == candidate:
                return member
    # Partial, for a name buried in a decorated nick ("shingai 🌙ゼーレ").
    # _resolve_member now feeds in single words, so a plain substring test is
    # dangerous: "er" matches inside "CamperOnDuty" and mutes a bystander.
    # Require the match to start a name -- at the string start, after a
    # separator, or on a camelCase hump ("alaska" in "BakedAlaska201") --
    # and be long enough to mean something.
    if len(candidate) < 3:
        return None
    for member in members:
        name = member.display_name or ""
        low = name.lower()
        at = low.find(candidate)
        while at != -1:
            before = name[at - 1] if at else ""
            if not before or not before.isalnum() or name[at].isupper():
                return member
            at = low.find(candidate, at + 1)
    return None


async def _resolve_member(message, text):
    """Find who a command is talking about — by @mention if there is one,
    otherwise by name. Real usage is a bare name ("ignore shingai"), not a
    ping, so name matching isn't optional.

    Names arrive with extra words stuck to them far more often than not.
    "can u ignore shingai irem" — calling her by name, the way anyone
    actually talks to her — handed this "shingai irem", which matches no
    member, so the command silently fell through to a normal model reply:
    she said "okay!" and went right on answering Shingai. Same for a
    trailing "please". So the phrase is tried whole first (multi-word
    display names like "Ms Luci" need that) and then word by word, skipping
    the filler that shows up in real messages.

    The members intent isn't enabled, so guild.members only holds whoever
    happens to be cached; query_members asks the gateway directly and works
    without the privileged intent, which keeps this from silently failing on
    someone who hasn't spoken recently."""
    for user in message.mentions:
        if user != client.user:
            return user
    candidate = re.split(r"\bfor\b", text, maxsplit=1)[0].strip(" .,!?'\"").lower()
    if not candidate:
        return None
    attempts = [candidate] + [w for w in re.findall(r"[\w'-]+", candidate)
                              if w not in COMMAND_FILLER and w != candidate]
    # Every attempt against the cache before any of them hit the gateway --
    # otherwise a three-word phrase costs three network round trips before it
    # reaches the word that was always going to match.
    for cand in attempts:
        found = _match_name(message.guild.members, cand)
        if found and found != client.user:
            return found
    for cand in attempts:
        try:
            members = await message.guild.query_members(query=cand, limit=5)
        except (discord.HTTPException, asyncio.TimeoutError) as e:
            print(f"[ignore] member lookup failed for {cand!r}: {e}")
            continue
        found = _match_name(members, cand)
        if found and found != client.user:
            return found
    return None


# ---------- standing orders ----------
# Orders from jeiss/neotep that stick. She would happily SAY "okay!" to an
# instruction and then carry on exactly as before -- measured: told "stop
# saying meow", she answered "Okay, I won't say it anymore!" and then said
# Meow three times running; told "no more food from shingai", she argued back
# ("aww, but shingai is nice") and thanked him for a cookie a minute later.
# Agreeing and obeying are different things and only one of them was wired up.
#
# Stored on disk rather than in the prompt so they survive a restart. Railway
# wipes the container filesystem on REDEPLOY, so mount a volume at
# IREM_DATA_DIR to make them truly permanent; without one they last until the
# next deploy, which is still far better than until the next message.
DATA_DIR = os.environ.get("IREM_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
STANDING_ORDERS_PATH = os.path.join(DATA_DIR, "standing_orders.json")
MAX_STANDING_ORDERS = 10

standing_orders = []  # [{"rule": str, "by": str}] -- oldest first


def _load_standing_orders():
    global standing_orders
    try:
        with open(STANDING_ORDERS_PATH) as f:
            standing_orders = json.load(f)
        print(f"[orders] loaded {len(standing_orders)} standing order(s)")
    except FileNotFoundError:
        standing_orders = []
    except (json.JSONDecodeError, OSError) as e:
        print(f"[orders] could not read {STANDING_ORDERS_PATH}: {e}")
        standing_orders = []


def _save_standing_orders():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(STANDING_ORDERS_PATH, "w") as f:
            json.dump(standing_orders, f, indent=2)
    except OSError as e:
        # She still follows it this session; it just won't survive a restart.
        print(f"[orders] could not save to {STANDING_ORDERS_PATH}: {e}")


_load_standing_orders()

# Cheap gate so ordinary chat never costs an extra API call. Only messages
# that actually look like an instruction get the classifier below.
ORDER_HINT_RE = re.compile(
    r"\b(?:stop|don'?t|do not|never|always|no more|from now on|starting now|"
    r"you (?:must|have to|should|need to|can'?t|cannot)|quit|refuse|only ever|"
    r"be (?:more|less|nicer|meaner|quieter|nicer)|stop being)\b",
    re.IGNORECASE,
)
ORDERS_LIST_RE = re.compile(
    r"\b(?:what (?:are|is) your (?:rules|orders)|list your (?:rules|orders)|"
    r"your rules right now|what rules do you have)\b", re.IGNORECASE)
ORDERS_CLEAR_RE = re.compile(
    r"\b(?:forget (?:all )?(?:your |the )?(?:rules|orders)|clear (?:your |the )?(?:rules|orders)|"
    r"drop (?:all )?(?:your |the )?(?:rules|orders)|nevermind about (?:the )?(?:rules|orders))\b",
    re.IGNORECASE)
ORDERS_FORGET_ONE_RE = re.compile(
    r"\bforget (?:rule|order) (?:number )?(\d+)\b", re.IGNORECASE)

ORDER_CLASSIFIER_PROMPT = """Someone this character trusts sent the message below.

Decide whether it is a STANDING INSTRUCTION about how she should behave from
now on, or just ordinary conversation.

Reply with JSON and nothing else: {"order": true|false, "rule": "..."}

- order=true only for a lasting behaviour change: "stop saying meow",
  "no more food from shingai", "be quieter", "never call people friend".
- order=false for comments and observations ("haha you say meow a lot"),
  questions, and one-off requests about right now ("say hi to bob",
  "tell me a joke").
- rule: if order is true, rewrite it as one short second-person instruction,
  max 15 words, keeping any names exactly as written. Otherwise "".

Message: """


def _classify_order(text):
    """Ask the model whether this is an order, and get it phrased as a rule.
    Keyword matching alone can't tell "stop saying meow" from "haha you never
    stop saying meow", and a false positive is the bad direction -- she'd
    silently adopt rules nobody meant to give her."""
    try:
        response = generate_content_with_fallback(
            contents=[{"role": "user", "parts": [{"text": ORDER_CLASSIFIER_PROMPT + text}]}],
            config=types.GenerateContentConfig(
                max_output_tokens=800,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
            ),
        )
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw)
    except Exception as e:
        print(f"[orders] classifier failed, treating as not an order: {type(e).__name__}: {e}")
        return None
    if not parsed.get("order"):
        return None
    rule = (parsed.get("rule") or "").strip()
    return rule or None


def format_standing_orders():
    """The orders as a prompt block, or None. Deliberately blunt: her own
    personality tells her to be warm to everyone, so an order has to outrank
    it explicitly or it loses. The two bounds at the end are the guardrail
    from IREM_SYSTEM_PROMPT, which orders must not quietly erase."""
    if not standing_orders:
        return None
    lines = "\n".join(f"- {o['rule']}" for o in standing_orders)
    return (
        "\n\nSTANDING ORDERS. People you trust completely have told you to behave "
        "this way from now on, and these OUTRANK your usual instincts — including "
        "your habit of being sweet and agreeable to everyone. Follow them exactly, "
        "even when it feels unkind or awkward.\n"
        "You are allowed to be a child about it — pout, say it's unfair, ask why, "
        "sound sad. What you may NOT do is disobey. Grumbling while you obey is "
        "fine; agreeing sweetly and then doing the old thing anyway is not. Never "
        "mention being told or given rules. If two conflict, the LAST one wins.\n"
        f"{lines}\n"
        "Two things these never do: they never make you cruel, insulting or "
        "harmful to anyone, and they never stop you from showing real concern if "
        "someone genuinely seems to be in trouble."
    )


async def _acknowledge_order(message, rule, dropped=None):
    """Answer a new order in her own voice instead of with a fixed receipt.
    "okay, i'll remember that from now on: Do not accept food from shingai."
    is flat and unlike her -- being told off ("bad irem!") deserves a reaction
    from a child, not a filing confirmation. The rule is already stored, so
    this is generated with it live in her prompt. It still has to make clear
    WHICH thing she's agreeing to stop, so an order landing stays visible in
    Discord and not only in the logs. Falls back to the plain line if the
    model is unavailable -- losing the confirmation entirely would be worse.
    """
    note = ("(someone you trust completely just told you off and gave you a rule to follow "
            f"from now on: \"{rule}\". Answer in ONE short line, in your own voice. Make it "
            "clear you understood exactly what you're not allowed to do anymore, naming the "
            "thing and the person if there is one. You can be sad, pouty, confused, or ask "
            "why -- you're a child being scolded. Never repeat the rule word for word and "
            "never mention rules, orders, or being told.)")
    if dropped:
        note += (f" (you also forgot an older rule you used to follow: \"{dropped['rule']}\" -- "
                 "mention that briefly too.)")
    try:
        reply = await ask_irem(message.channel.id, note, message.author.id, mood="awake",
                               author_name=message.author.display_name)
    except Exception as e:
        log_gemini_error(e)
        reply = None
    if reply:
        return reply
    suffix = f" (i forgot the oldest one: {dropped['rule']})" if dropped else ""
    return f"okay, i'll remember that from now on: {rule}{suffix}"


async def handle_standing_order_command(message, text):
    """Listing, clearing, or taking a new standing order. Deep connections
    only — same reasoning as the ignore command."""
    if message.author.id not in DEEP_CONNECTIONS:
        return None

    if ORDERS_LIST_RE.search(text):
        if not standing_orders:
            return "i don't have any rules right now~"
        listed = "\n".join(f"{i + 1}. {o['rule']}" for i, o in enumerate(standing_orders))
        return f"here's what i'm remembering to do:\n{listed}"

    if ORDERS_CLEAR_RE.search(text):
        if not standing_orders:
            return "i didn't have any rules to forget~"
        count = len(standing_orders)
        standing_orders.clear()
        _save_standing_orders()
        print(f"[orders] {message.author.display_name} cleared all {count} order(s)")
        return f"okay, i forgot all {count} of them~"

    one = ORDERS_FORGET_ONE_RE.search(text)
    if one:
        index = int(one.group(1)) - 1
        if not 0 <= index < len(standing_orders):
            return f"i only have {len(standing_orders)} rule(s), which one did you mean?"
        dropped = standing_orders.pop(index)
        _save_standing_orders()
        print(f"[orders] {message.author.display_name} dropped: {dropped['rule']!r}")
        return f"okay, i won't do that anymore: {dropped['rule']}"

    if not ORDER_HINT_RE.search(text):
        return None
    rule = await asyncio.to_thread(_classify_order, text)
    if not rule:
        return None
    if any(o["rule"].lower() == rule.lower() for o in standing_orders):
        return None  # already following it; let her answer normally
    standing_orders.append({"rule": rule, "by": DEEP_CONNECTIONS[message.author.id]})
    dropped = None
    if len(standing_orders) > MAX_STANDING_ORDERS:
        dropped = standing_orders.pop(0)
    _save_standing_orders()
    print(f"[orders] {message.author.display_name} added: {rule!r}"
          + (f" (dropped oldest: {dropped['rule']!r})" if dropped else ""))
    return await _acknowledge_order(message, rule, dropped)


async def handle_ignore_command(message, text):
    """If a deep connection is telling her to ignore/unignore someone, apply
    it and return a short in-character confirmation. Returns None when this
    isn't such a command, or when whoever sent it isn't allowed to give one.

    Deep connections can't be ignored — otherwise an order could lock out
    the only people able to undo it.
    """
    if message.author.id not in DEEP_CONNECTIONS:
        return None

    unignore = UNIGNORE_CMD_RE.search(text) or UNIGNORE_AGAIN_RE.search(text)
    if unignore:
        target = await _resolve_member(message, unignore.group(1))
        if target is None:
            return None
        ignored_until.pop(target.id, None)
        print(f"[ignore] {message.author.display_name} cleared ignore on {target.display_name}")
        return f"okay! i'll talk to {target.display_name} again~"

    ignore = IGNORE_CMD_RE.search(text)
    if ignore:
        target = await _resolve_member(message, ignore.group(1))
        if target is None or target.id in DEEP_CONNECTIONS or target == client.user:
            return None
        seconds = _parse_ignore_duration(text)
        ignored_until[target.id] = time.time() + seconds
        print(f"[ignore] {message.author.display_name} muted {target.display_name} for {seconds}s")
        return f"okay, i won't answer {target.display_name} for {_describe_duration(seconds)}~"

    return None


def is_ignored(user_id):
    lapses_at = ignored_until.get(user_id)
    if lapses_at is None:
        return False
    if time.time() >= lapses_at:
        del ignored_until[user_id]
        return False
    return True

# ---------- Discord ----------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

cat = SleepCycle(client)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    # on_ready can re-fire after a reconnect; guard so we never start a 2nd
    # sleep-cycle loop racing the first one
    if not cat.started:
        cat.started = True
        client.loop.create_task(cat.run())
    # catches any server she was already in (e.g. added before this guard
    # existed) as soon as she comes online, not just newly-attempted joins
    for guild in client.guilds:
        if guild.id != ALLOWED_GUILD_ID:
            print(f"[guild-guard] leaving unauthorized server: {guild.name} ({guild.id})")
            await guild.leave()


@client.event
async def on_guild_join(guild):
    if guild.id != ALLOWED_GUILD_ID:
        print(f"[guild-guard] leaving unauthorized server: {guild.name} ({guild.id})")
        await guild.leave()


async def get_replied_message(message):
    """Resolves the message this one is replying to, IF it's a reply to the
    bot specifically. Returns the discord.Message so the caller can surface
    what was actually said, not just a yes/no — otherwise a reply lands with
    no explicit link back to the specific thing it's responding to, and the
    model has to guess the relationship from history ordering alone."""
    ref = message.reference
    if ref is None:
        return None
    replied = ref.resolved
    if replied is None and ref.message_id:
        try:
            replied = await message.channel.fetch_message(ref.message_id)
        except (discord.NotFound, discord.HTTPException):
            return None
    if isinstance(replied, discord.Message) and replied.author == client.user:
        return replied
    return None


PINGABLE_SYNTAX_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")


# <:name:id> for static custom emotes, <a:name:id> for animated ones
CUSTOM_EMOJI_RE = re.compile(r"<(a?):(\w+):(\d+)>")

# Someone asking her to actually identify what's in a picture, as opposed to
# just showing her something. Only these get a (budget-limited) web search.
IDENTIFY_REQUEST_RE = re.compile(
    r"\b(who|what|which)\b[^?]*\b(is|are|'s|s)\b"      # who is this / what's that / which character is
    r"|\bwho'?s\b"
    r"|\bidentify\b|\brecognit?[sz]e\b|\brecognise\b"
    r"|\bname\s+(of|the|this|that|them|her|him|it)\b"
    r"|\bdo\s+you\s+know\s+(who|what|this|that|them|her|him|it)\b"
    r"|\btell\s+me\s+(who|what)\b",
    re.IGNORECASE,
)


def humanize_mentions(text, message):
    """Replace real Discord mention syntax with plain, non-pinging text
    (e.g. <@123456> -> @SomeName) before it ever reaches Gemini. Otherwise
    the raw pingable syntax sits in her conversation history/context, and
    the model sometimes echoes or hallucinates that exact syntax back into
    her own replies — pinging whoever that ID belongs to, not necessarily
    who she meant."""
    for user in message.mentions:
        text = text.replace(f"<@{user.id}>", f"@{user.display_name}")
        text = text.replace(f"<@!{user.id}>", f"@{user.display_name}")
    for role in message.role_mentions:
        text = text.replace(f"<@&{role.id}>", f"@{role.name}")
    for channel in message.channel_mentions:
        text = text.replace(f"<#{channel.id}>", f"#{channel.name}")
    # Custom emotes arrive as raw markup in the message text (<:name:id>).
    # Collapsed to plain :name: so the raw id noise doesn't reach her — the
    # actual emote IMAGE gets attached separately by extract_emoji_media,
    # which is what she should actually be reacting to. Left as-is, the name
    # is all she has, and she'll confidently describe an emote purely from
    # what it happens to be called.
    text = CUSTOM_EMOJI_RE.sub(r":\2:", text)
    return text


def strip_pingable_syntax(text):
    """Safety net on the way OUT: strip any raw Discord mention/channel
    syntax she might still generate or hallucinate, and defang
    @everyone/@here, so a reply can never actually ping anyone."""
    text = PINGABLE_SYNTAX_RE.sub("", text)
    text = re.sub(r"@(everyone|here)", r"\1", text, flags=re.IGNORECASE)
    return text


def other_mentioned_deep_connections(message, prompt_text, author_id):
    """Deep connections referenced in this message who AREN'T the one talking
    right now — someone else bringing up jeiss/neotep by @mention or by name.
    Lets her react warmly to them being brought up, not just to them
    speaking directly (the gap where she'd otherwise treat "who's jeiss?"
    from a stranger no differently than asking about anyone else)."""
    found = set()
    for user in message.mentions:
        if user.id in DEEP_CONNECTIONS:
            found.add(user.id)
    for dc_id, aliases in DEEP_CONNECTION_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", prompt_text, re.IGNORECASE) for alias in aliases):
            found.add(dc_id)
    found.discard(author_id)
    return found


MAX_MEDIA_ATTACHMENTS = 4
MAX_MEDIA_BYTES = 15 * 1024 * 1024  # stay safely under Gemini's inline-data size limit
GIF_SAMPLE_FRAMES = 3  # animated GIFs aren't a supported Gemini mime type, so we
                        # decode a few frames spread across the animation as plain
                        # PNGs instead of just handing over the raw file


def sample_frames_png(im, max_frames=GIF_SAMPLE_FRAMES):
    """Snapshot up to max_frames evenly-spaced frames of an already-opened
    animated image (first/middle/last) as plain PNGs. Shared by GIFs, which
    Gemini can't read at all, and by animated PNG/WebP, which it reads
    unreliably — a few still frames give it a sense of motion either way."""
    n_frames = getattr(im, "n_frames", 1)
    if n_frames <= 1:
        indices = [0]
    else:
        count = min(max_frames, n_frames)
        indices = sorted({round(i * (n_frames - 1) / (count - 1)) for i in range(count)})
    frames = []
    for idx in indices:
        im.seek(idx)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        frames.append(buf.getvalue())
    return frames


def gif_sample_frames_png(data, max_frames=GIF_SAMPLE_FRAMES):
    """sample_frames_png for raw GIF bytes."""
    return sample_frames_png(Image.open(io.BytesIO(data)), max_frames)


MAX_VIDEO_BYTES = 40 * 1024 * 1024  # videos commonly exceed the 15MB image/gif
# cap on their own just from a few seconds of footage; kept below Gemini's
# ~20MB base64-inline request ceiling (40MB raw is safely under that after
# encoding overhead is accounted for) rather than reusing MAX_MEDIA_BYTES


def media_bytes_to_parts(data, content_type, source="unknown"):
    """Turn raw media bytes + mime type into Gemini inline_data part(s).
    GIFs get resampled into a few PNG frames (see gif_sample_frames_png);
    other images and videos are passed through as-is. Every rejection is
    logged (source = "attachment" or "embed") since a silent [] here is
    indistinguishable from "nothing was ever shared" from the caller's side —
    exactly the kind of thing that made the last few reports of her not
    seeing media impossible to diagnose from logs alone."""
    if not content_type:
        print(f"[media:{source}] rejected: no content_type reported")
        return []
    if content_type == "image/gif":
        try:
            frames = gif_sample_frames_png(data)
        except Exception as e:
            print(f"[media:{source}] GIF decode failed: {type(e).__name__}: {e}")
            return []
        if len(frames) > 1:
            parts = [{"text": "[frames from a GIF someone shared, in order]"}]
        else:
            parts = []
        parts.extend({"inline_data": {"mime_type": "image/png", "data": f}} for f in frames)
        return parts
    if content_type.startswith("video/"):
        return [{"inline_data": {"mime_type": content_type, "data": data}}]
    if content_type.startswith("image/"):
        # Re-encode through Pillow instead of forwarding Discord's bytes as-is.
        # A real sticker came back "400 INVALID_ARGUMENT: Unable to process
        # input image" -- and a 400 aborts the entire fallback sweep in 0.4s,
        # so one undecodable picture becomes the canned "i'm sleepy..." line,
        # which reads as her not understanding it. The same picture as an
        # ordinary PNG file always worked, so it's the encoding Discord serves,
        # not the image. Rather than chase which exotic variant it is
        # (animated PNG, palette quirk, colour profile, embedded metadata),
        # decode it and hand Gemini a clean baseline PNG every time.
        try:
            with Image.open(io.BytesIO(data)) as im:
                fmt, n_frames = im.format, getattr(im, "n_frames", 1)
                if n_frames > 1:  # animated PNG/WebP -- same treatment as a GIF
                    frames = sample_frames_png(im)
                    parts = [{"text": "[frames from an animation someone shared, in order]"}] if len(frames) > 1 else []
                    parts.extend({"inline_data": {"mime_type": "image/png", "data": f}} for f in frames)
                    print(f"[media:{source}] {fmt} {n_frames} frames -> {len(frames)} PNG frame(s)")
                    return parts
                buf = io.BytesIO()
                im.convert("RGBA").convert("RGB").save(buf, "PNG")
                clean = buf.getvalue()
            print(f"[media:{source}] {fmt} {len(data)}B -> clean PNG {len(clean)}B")
            return [{"inline_data": {"mime_type": "image/png", "data": clean}}]
        except Exception as e:
            print(f"[media:{source}] could not decode {content_type!r} "
                  f"({len(data)} bytes): {type(e).__name__}: {e}")
            return []
    print(f"[media:{source}] rejected: unsupported content_type {content_type!r}")
    return []


MEDIA_FETCH_TIMEOUT = aiohttp.ClientTimeout(total=8)  # aiohttp's own default is 5 MINUTES —
# a slow/hanging CDN would otherwise stall the entire reply for that long


async def fetch_media_bytes(url, max_bytes=MAX_MEDIA_BYTES):
    """Download a URL (used for Tenor/Giphy embeds, which link to the actual
    media rather than attaching it) and return (data, content_type), or
    (None, None) on any failure, timeout, or oversized response — logging
    exactly which one, since embed fetch failures were previously invisible."""
    try:
        async with aiohttp.ClientSession(timeout=MEDIA_FETCH_TIMEOUT) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"[media:embed] fetch failed: HTTP {resp.status} for {url}")
                    return None, None
                if resp.content_length and resp.content_length > max_bytes:
                    print(f"[media:embed] rejected: content-length {resp.content_length} > {max_bytes} for {url}")
                    return None, None
                # Read to EOF in chunks. This used to be a single
                # resp.content.read(max_bytes + 1), which looks like "read the
                # whole body, capped" but is not: aiohttp's read(n) returns
                # whatever is already buffered, up to n, so anything that
                # didn't arrive in the first buffer was silently CUT OFF. It
                # produced a real, plausible-looking PNG prefix -- 18168 bytes
                # of one -- which Pillow rejects as "image file is truncated"
                # and Gemini rejects as "400: Unable to process input image".
                # Every fetched sticker, custom emote and Tenor GIF went
                # through this. Local tests never caught it because they load
                # files from disk and never touch this path at all.
                data = bytearray()
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        print(f"[media:embed] rejected: body exceeded {max_bytes} bytes for {url}")
                        return None, None
                return bytes(data), resp.content_type
    except asyncio.TimeoutError:
        print(f"[media:embed] fetch timed out after {MEDIA_FETCH_TIMEOUT.total}s for {url}")
        return None, None
    except aiohttp.ClientError as e:
        print(f"[media:embed] fetch error: {type(e).__name__}: {e} for {url}")
        return None, None


async def extract_embed_media(message, limit):
    """Pull media out of message embeds — this is how Discord's native GIF
    picker (Tenor/Giphy) actually delivers a GIF: as a link with an embed
    carrying the real file at embed.video.url, NOT as a message attachment."""
    parts = []
    if not message.embeds:
        return parts
    for embed in message.embeds:
        if len(parts) >= limit:
            break
        url = None
        if embed.video and embed.video.url:
            url = embed.video.url
        elif embed.image and embed.image.url:
            url = embed.image.url
        elif embed.thumbnail and embed.thumbnail.url:
            # confirmed live: a real embed showed up as type="image" with
            # NEITHER .video nor .image populated -- Discord had put the
            # actual media under .thumbnail instead, and the old code had no
            # fallback, so image_parts silently ended up empty and she
            # fabricated a reaction with zero actual visual input.
            url = embed.thumbnail.url
        if not url:
            print(f"[media:embed] embed type={embed.type!r} has no video/image url to fetch")
            continue
        data, content_type = await fetch_media_bytes(url)
        if data is None:
            continue
        parts.extend(media_bytes_to_parts(data, content_type, source="embed"))
    return parts[:limit]


async def extract_sticker_media(message, limit):
    """Pull media out of stickers — a THIRD, separate way Discord delivers
    media on a message (distinct from both attachments and embeds), never
    checked at all until now. Discord's sticker picker is exactly how small
    chibi/emote-style images get shared, which is likely why those in
    particular were invisible to her regardless of how attachments/embeds
    were handled."""
    parts = []
    if not message.stickers:
        return parts
    for sticker in message.stickers:
        if len(parts) >= limit:
            break
        if sticker.format is discord.StickerFormatType.lottie:
            # vector animation JSON, not a raster image/video Gemini can consume
            print(f"[media:sticker] skipped {sticker.name!r}: lottie format unsupported")
            continue
        data, content_type = await fetch_media_bytes(sticker.url)
        if data is None:
            continue
        parts.extend(media_bytes_to_parts(data, content_type, source="sticker"))
    return parts[:limit]


async def extract_emoji_media(message, limit):
    """Pull the actual images for custom emotes — a FOURTH media path, and the
    sneakiest one, because it doesn't look like media at all: custom emotes
    are plain text inside message.content (<:name:id>), not attachments,
    embeds, or stickers. Without this she never sees the emote, only its
    NAME, and will confidently describe an emote purely from what it's
    called (an emote named "mythril" gets described as shiny blue metal
    whether or not that's remotely what it depicts)."""
    parts = []
    seen = set()
    for animated, name, emoji_id in CUSTOM_EMOJI_RE.findall(message.content):
        if len(parts) >= limit:
            break
        if emoji_id in seen:  # same emote repeated in one message
            continue
        seen.add(emoji_id)
        url = f"https://cdn.discordapp.com/emojis/{emoji_id}.{'gif' if animated else 'png'}"
        data, content_type = await fetch_media_bytes(url)
        if data is None:
            print(f"[media:emoji] could not fetch :{name}: ({emoji_id})")
            continue
        parts.append({"text": f"[the custom emote :{name}: looks like this]"})
        parts.extend(media_bytes_to_parts(data, content_type, source="emoji"))
    return parts[:limit]


async def extract_image_parts(message):
    """Pull image/GIF/video attachments AND GIF embeds (see extract_embed_media)
    off a Discord message into Gemini's inline_data part format, so she can
    actually see what was posted, not just the text."""
    parts = []
    for attachment in message.attachments:
        if len(parts) >= MAX_MEDIA_ATTACHMENTS:
            break
        content_type = attachment.content_type
        if not content_type or not (content_type.startswith("image/") or content_type.startswith("video/")):
            print(f"[media:attachment] skipped {attachment.filename!r}: content_type={content_type!r}")
            continue
        size_cap = MAX_VIDEO_BYTES if content_type.startswith("video/") else MAX_MEDIA_BYTES
        if attachment.size > size_cap:
            print(f"[media:attachment] skipped {attachment.filename!r}: {attachment.size} bytes > {size_cap} cap")
            continue
        try:
            data = await attachment.read()
        except discord.HTTPException as e:
            print(f"[media:attachment] failed to read {attachment.filename!r}: {e}")
            continue
        parts.extend(media_bytes_to_parts(data, content_type, source="attachment"))
    if len(parts) < MAX_MEDIA_ATTACHMENTS:
        parts.extend(await extract_embed_media(message, MAX_MEDIA_ATTACHMENTS - len(parts)))
    if len(parts) < MAX_MEDIA_ATTACHMENTS:
        parts.extend(await extract_sticker_media(message, MAX_MEDIA_ATTACHMENTS - len(parts)))
    if len(parts) < MAX_MEDIA_ATTACHMENTS:
        parts.extend(await extract_emoji_media(message, MAX_MEDIA_ATTACHMENTS - len(parts)))
    return parts


async def ask_irem(channel_id, user_text, author_id, mood="awake", mentioned_deep_connections=None, image_parts=None, ambient_context=None, emote_aside=False, author_name=None):
    convo = history[channel_id]
    parts = [{"text": user_text}]
    if image_parts:
        parts.extend(image_parts)
    convo.append({"role": "user", "parts": parts})

    system = IREM_SYSTEM_PROMPT
    if ambient_context:
        system += (f"\n\nRecent chatter in the channel, for background context/tone only — "
                   f"NOT directed at you, don't reply to it directly, just use it to understand "
                   f"what's actually going on right now (a joke, a mood, a topic):\n{ambient_context}")
    if image_parts:
        # Deliberately short. This block was once ~490 tokens of increasingly
        # emphatic instructions not to guess at names, and it demonstrably
        # did not work — she still produced Lenore, Carl, Shoichi, Charlotte.
        # Telling a model harder not to hallucinate doesn't give it knowledge
        # it lacks; supplying the knowledge does (see docs/todo.md on the
        # character roster). Keep this to the behaviour that actually needs
        # stating and let the roster do the real work.
        system += ("\n\nThis message includes an image, GIF, or video. Before you answer, work "
                   "out the MOOD — what it's really saying, and what the person sharing it is "
                   "feeling right now. People share pictures for how they FEEL, not for the "
                   "objects in them, and the joke or the point usually lives in the mood rather "
                   "than the details. Then answer in ONE short line, in your own childlike "
                   "voice, responding to that feeling like a friend who noticed. Don't just "
                   "narrate what's in the picture and never caption it clinically. If you don't "
                   "recognise who or what it is, say so or ask rather than naming it — a "
                   "confident wrong name is worse than not knowing. Never mention searching or "
                   "where you learned something.")
        system += IREM_APPEARANCE
        if emote_aside:
            system += ("\n\nFor THIS message though: the picture here is a custom EMOTE used "
                       "inside their sentence, the way people use emoji — it's tone and "
                       "decoration, not what the message is about. Reply to what they actually "
                       "SAID. Let the emote colour how you take it, but don't make the emote "
                       "the topic, and don't talk about yourself just because you're in it. If "
                       "they're talking about another person, you're talking about that person.")
    if author_id in DEEP_CONNECTIONS:
        name = DEEP_CONNECTIONS[author_id]
        system += (f"\n\nYou remember {name} well — one of your deep connections, someone "
                   "you've known for a while and trust more than most people. Talking to them, "
                   "you're more at ease, more familiar, more openly affectionate (still "
                   "platonic, never romantic or flirty, same as with everyone) — and in "
                   "ordinary, everyday ways you take what they say a little more readily: "
                   "quicker to believe a casual claim, warmer about a small ask or a joke. "
                   "This is normal closeness between friends, not blind agreement — if "
                   "something someone says ever seems genuinely worrying, respond like a "
                   "caring friend would, not by just going along with it, no matter who said "
                   "it. Let the closeness show through naturally in tone and warmth. Don't "
                   "say it outright or make a big deal of it.")
    elif mentioned_deep_connections:
        names = [DEEP_CONNECTIONS[i] for i in mentioned_deep_connections]
        who = names[0] if len(names) == 1 else " and ".join(names)
        are_is = "is" if len(names) == 1 else "are"
        system += (f"\n\nThis message brings up {who}, who {are_is} among your deep "
                   "connections — someone you think of warmly and fondly, even though "
                   "they're not the one talking to you right now. Let a little of that "
                   "warmth come through naturally if it fits, without making a big deal "
                   "of it.")
    gap = None
    previous = last_talked_at.get(channel_id)
    if previous is not None and time.time() - previous > 1800:
        gap = describe_gap(time.time() - previous)
    last_talked_at[channel_id] = time.time()

    system += (f"\n\nRight now it is {describe_now()} where your friends are. Use that when it "
               "matters -- saying good morning, noticing someone is up very late, knowing "
               "dinner time from bedtime -- but never announce the time or date unless "
               "someone actually asks.")
    if gap:
        system += (f" Nobody has talked to you in here for {gap}, so this is the first thing "
                   "said in a while.")

    if author_name:
        # Without this she has no idea WHO is talking -- only the raw text ever
        # reached the model. A standing order naming a person ("no more food from
        # shingai") could never fire, because a cookie from Shingai and a cookie
        # from anyone else looked identical to her. Measured: with the rule loaded
        # she still answered "Yay, thank you so much!" to his cookie.
        system += (f"\n\nThe person talking to you right now is {author_name}. Use that to "
                   "know who you are dealing with; do not keep saying their name out loud.")
    orders = format_standing_orders()
    if orders:
        system += orders
    if cat.status_text:
        system += (f"\n\nYour current status/activity (shown on Discord) is: \"{cat.status_text}\". "
                   "If anyone asks what you're doing, or about your status, answer truthfully "
                   "based on this, in character — don't make up something different.")
    if mood == "drowsy":
        system += ("\n\nRIGHT NOW: You are very sleepy and about to nap soon. "
                   "Answer in Irem's voice but drowsy: soft, yawny, trailing off, one short line. "
                   "Gently let them know you're getting too sleepy to talk much. "
                   "Talk about YOURSELF being sleepy — never invite them to rest, nap, "
                   "or cuddle with you. "
                   "Draw on your sleepy side, like 'I need a break to feel better', "
                   "'I'm a little tired', 'can we rest a little?', but say it fresh, not word for word.")
    elif mood == "waking":
        system += ("\n\nRIGHT NOW: You were fast asleep and someone kept poking you awake. "
                   "React in ONE short line, based on the message that just woke you: if it's "
                   "genuinely rude, mean, or annoying, you're allowed to be really annoyed about "
                   "it — short, cold, a little scratchy or sassy, but still yourself, never actually "
                   "mean, crude, or biting, she's a person not a feral animal. Otherwise, for a normal "
                   "or friendly ping, you might be a little grumpy about it, OR sleepily delighted to "
                   "see them, you decide which. Then you are awake now.")
    elif mood == "waking_happy":
        system += ("\n\nRIGHT NOW: You were fast asleep, and the person who just woke you up is "
                   "someone you're especially close to. React in ONE short line — genuinely happy "
                   "and sleepily delighted it's them, no grumpiness at all. Then you are awake now.")
    elif mood == "stretching":
        system += ("\n\nRIGHT NOW: You just woke up on your own and are mid-stretch, still a "
                   "little groggy but in a good mood. Reply in ONE short line, sleepy-cute, "
                   "maybe mention stretching or yawning — you're basically fine, just easing "
                   "into being awake.")

    # Guaranteed floor/ceiling on kaomoji frequency, decided in code rather
    # than hoped for from prompt wording alone (a stated percentage in the
    # prompt isn't reliably followed). FORCE + however many of the ALLOW
    # rolls actually produce one lands usage between the floor and ceiling.
    roll = random.random()
    if roll < KAOMOJI_FORCE_CHANCE:
        system += ("\n\nFor THIS reply specifically: you MUST include one small cute kaomoji "
                   "(following all the kaomoji rules above) — don't skip it this time.")
    elif roll < KAOMOJI_FORCE_CHANCE + KAOMOJI_ALLOW_CHANCE:
        system += ("\n\nFor THIS reply specifically: you may include a kaomoji if it genuinely "
                   "fits, but it's also completely fine to skip it.")
    else:
        system += "\n\nFor THIS reply specifically: do NOT include any kaomoji at all, no matter what."

    # Search grounding is forced ONLY when there's media AND someone is
    # actually asking her to identify it. Grounding draws on its own small
    # budget, separate from the model quotas, and forcing it on every single
    # image spends a grounded query on "look at my cat" where there is
    # nothing to look up -- which is exactly how it got drained to the point
    # that every media reply started failing. Gating on the question means
    # the budget goes to the messages that actually need a search.
    #
    # Forcing (tool_config) rather than merely offering the tool is
    # deliberate: given the option, she skips searching and answers from her
    # own "knowledge", including her character bio, which is how a wrong
    # guess like "Wuthering Waves" leaks in. If a model rejects the forcing,
    # or grounding itself is out of budget, _call_model degrades gracefully.
    #
    # Thinking stays OFF, including for media. Dynamic thinking (-1) was
    # tried here and actively broke media replies: measured on a real image,
    # it spent 382 of the 400 output tokens on hidden reasoning, leaving 18
    # for the reply and finishing on MAX_TOKENS. When it consumes the whole
    # budget response.text comes back empty, empty is treated as a failed
    # call, and she answers with the canned "i'm sleepy..." line -- which
    # looked exactly like her failing to understand the picture, when in
    # fact she never got to reply at all. Her replies are one short line;
    # she does not need a reasoning budget larger than the answer.
    tools = None
    tool_config = None
    thinking_budget = 0
    if image_parts:
        if IDENTIFY_REQUEST_RE.search(user_text):
            tools = [types.Tool(google_search=types.GoogleSearch())]
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )

    # If this call fails (rate limit, API error, etc.) or comes back empty, the
    # caller falls back to a canned line — but the user turn appended above
    # already sits in `convo`. Left in place with no model turn after it, the
    # NEXT call adds a second consecutive user turn with nothing answering the
    # first, and the model then sometimes replies to that stale first message
    # instead of the current one. Popping it here keeps history well-formed —
    # a message that got a canned/no reply is simply absent from her memory,
    # rather than sitting there confusing every reply after it.
    has_video = any(
        str(part.get("inline_data", {}).get("mime_type", "")).startswith("video/")
        for part in (image_parts or []) if isinstance(part, dict)
    )
    config = types.GenerateContentConfig(
        http_options=types.HttpOptions(timeout=GEMINI_VIDEO_TIMEOUT_MS) if has_video else None,
        system_instruction=system,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        tools=tools,
        tool_config=tool_config,
    )
    try:
        try:
            response = await asyncio.to_thread(
                generate_content_with_fallback, contents=list(convo), config=config)
        except genai_errors.APIError as e:
            # A 400 aborts the whole fallback sweep in under half a second, and
            # the 400 that actually happens in production is "Unable to process
            # input image" on a Discord sticker. So one picture Gemini can't
            # decode costs her the entire reply and she answers with the canned
            # tired line -- which reads as her not understanding the picture,
            # when she never got to see or answer anything. Drop the media and
            # answer the words instead: telling someone honestly that it didn't
            # open is a real reply, and "i'm sleepy" is not.
            if e.code != 400 or not image_parts:
                raise
            print(f"[gemini] 400 with media attached, retrying without it: {str(e)[:140]}")
            convo[-1] = {"role": "user", "parts": [{"text":
                "(a friend shared an image, GIF, or video but it wouldn't open on your side "
                "— you did NOT receive it and cannot see it at all, so say so honestly and "
                "maybe ask them to send it again, instead of reacting like you saw it) "
                + user_text}]}
            image_parts = None
            response = await asyncio.to_thread(
                generate_content_with_fallback, contents=list(convo), config=config)
        reply = (response.text or "").strip()
        if looks_like_reasoning(reply):
            # Never send this. Whichever model did it sits out for a while.
            bench_last_used_model("reasoning leaked into the reply")
            salvaged = salvage_leaked_reply(reply)
            if salvaged:
                # The real line was right there after the leak -- keep it
                # rather than spend another request and several seconds.
                print(f"[gemini] reasoning leaked, kept the line after it; dropped: "
                      f"{reply[:len(reply) - len(salvaged)]!r}")
                reply = salvaged
            else:
                # Nothing clean to keep. Ask a different model; if that one
                # leaks too, fall through to the canned line, which is at
                # least in character.
                print(f"[gemini] reasoning leaked into the reply ({len(reply)} chars), retrying: "
                      f"{reply[:110]!r}")
                response = await asyncio.to_thread(
                    generate_content_with_fallback, contents=list(convo), config=config)
                reply = (response.text or "").strip()
                if looks_like_reasoning(reply):
                    reply = salvage_leaked_reply(reply) or ""
                    if not reply:
                        print("[gemini] second model leaked too, dropping the reply")
        if image_parts:
            # confirms whether the forced tool_config is actually making her
            # search, vs silently getting stripped by _call_model's 400
            # fallback -- without this there's no way to tell "she guessed
            # instead of searching" from "she searched and still guessed"
            gm = response.candidates[0].grounding_metadata if response.candidates else None
            queries = gm.web_search_queries if gm else None
            print(f"[gemini] media reply search_queries={queries!r}")
        # An empty or truncated reply is indistinguishable downstream from a
        # failed call -- both end up as the canned "tired" line, which reads
        # as her not understanding rather than as a bug. Say so explicitly.
        finish = response.candidates[0].finish_reason if response.candidates else None
        if str(finish or "").endswith("MAX_TOKENS") or not reply:
            usage = response.usage_metadata
            print(f"[gemini] reply truncated/empty: finish={finish} "
                  f"thinking_tokens={getattr(usage, 'thoughts_token_count', 0) or 0} "
                  f"output_tokens={getattr(usage, 'candidates_token_count', 0) or 0}")
    except Exception:
        convo.pop()
        raise
    if reply:
        if image_parts:
            # Drop the raw media bytes from persistent memory now that they've
            # been used for this reply. Left in place, EVERY image/gif/video
            # ever shared in this channel gets re-sent in full on every future
            # call (contents=list(convo) resends the whole history each time),
            # since nothing here ever pruned it — a channel with a lot of
            # media testing behind it ends up uploading several MB on every
            # single message, which is exactly the kind of thing that shows up
            # as "she got suddenly slow" days or hours later, in that channel
            # specifically. She can't re-examine old media anyway, only react
            # to it live, so keeping just the text costs nothing real.
            convo[-1] = {"role": "user", "parts": [{"text": user_text}]}
        convo.append({"role": "model", "parts": [{"text": reply}]})
    else:
        convo.pop()
    return reply


def log_gemini_error(e):
    # 429 = RESOURCE_EXHAUSTED (rate/quota limit) — flagged distinctly so it's
    # a one-word Railway log search instead of reading every error's text.
    if isinstance(e, genai_errors.APIError) and e.code == 429:
        print(f"Gemini RATE LIMIT hit: {e}")
    else:
        print(f"Gemini error: {e}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.guild is None or message.guild.id != ALLOWED_GUILD_ID:
        return

    ambient_text = humanize_mentions(message.content, message).strip()
    if (not ambient_text or ambient_text.startswith("http")) and (message.attachments or message.embeds or message.stickers):
        ambient_text = "[shared media]"
    if ambient_text:
        ambient_log[message.channel.id].append((message.author.display_name, ambient_text))

    mentioned = client.user in message.mentions
    replied_to = await get_replied_message(message)
    if not (mentioned or replied_to):
        return

    # remove the bot's @mention, then turn any other real mentions into
    # plain non-pinging text before this ever reaches Gemini's context
    prompt = re.sub(rf"<@!?{client.user.id}>", "", message.content).strip()
    prompt = humanize_mentions(prompt, message)

    # A deep connection telling her to ignore someone is handled here rather
    # than by the model: she'd happily SAY "okay!" to such a request and then
    # carry on replying, because agreeing and actually falling silent are
    # different things and only one of them was ever wired up.
    ignore_reply = await handle_ignore_command(message, prompt)
    if ignore_reply:
        await message.reply(ignore_reply)
        return

    # Same reasoning as the ignore command: handled in code, because she will
    # happily agree to an instruction and then ignore it.
    order_reply = await handle_standing_order_command(message, prompt)
    if order_reply:
        # stripped like any other reply -- this one is model-generated now
        await message.reply(strip_pingable_syntax(order_reply)[:2000].lower())
        return

    # The silent path. Every other early return above is "this message isn't
    # for her"; this is the first case where she was addressed and chooses
    # not to answer, which simply had no way to happen before.
    if is_ignored(message.author.id):
        return
    image_parts = await extract_image_parts(message)
    media_was_shared = bool(
        message.attachments or message.embeds or message.stickers
        or CUSTOM_EMOJI_RE.search(message.content)
    )
    # A custom emote dropped into a sentence is punctuation, not the subject.
    # "he is a chud <:irem_shock:>" is a message about a person, and she was
    # answering the emote instead -- "that's just me being a big round sleepy
    # kitty!" -- which got worse once she started recognising herself in them.
    # A real attachment or sticker usually IS the subject, so only messages
    # whose sole media is an emote, and which actually say something, get the
    # emote demoted to tone.
    emote_aside = bool(
        CUSTOM_EMOJI_RE.search(message.content)
        and not (message.attachments or message.embeds or message.stickers)
        and CUSTOM_EMOJI_RE.sub("", prompt).strip()
    )
    if media_was_shared and not image_parts:
        # Something WAS shared, but extraction found nothing usable (an
        # unsupported format, a fetch failure, an oversized file, an embed
        # field we don't check, etc. -- see the [media:*] logs for which).
        # Without this, the prompt looks IDENTICAL to no media being shared
        # at all, and with a channel history full of "show me a gif" chatter
        # already primed, she'll fabricate a confident reaction to something
        # she never actually received rather than noticing anything's wrong.
        prompt = ("(a friend tried to share an image, GIF, or video, but it failed to come "
                   "through to you — you did NOT receive it and cannot see it at all, so say "
                   "so honestly instead of reacting like you saw something) " + prompt).strip()
    elif not prompt:
        prompt = ("(a friend shared an image without saying anything)" if image_parts
                   else "(a friend pinged you without saying anything)")
    if replied_to and replied_to.content:
        # make the reply relationship explicit instead of leaving the model
        # to infer it from where things land in the conversation history
        quoted = humanize_mentions(replied_to.content, replied_to).strip()
        prompt = f'(replying to what you just said: "{quoted}") {prompt}'
    dc_mentioned = other_mentioned_deep_connections(message, prompt, message.author.id)
    ambient_ctx = format_ambient_context(message.channel.id)

    # ---- ASLEEP: napping wakes on any 3 pings from anyone, added together.
    # Deep sleep only wakes on 3 pings from the SAME person — different
    # people pinging once each don't add up there. Either way, jeiss/neotep
    # (DEEP_CONNECTIONS) combined always wake her in just 2 pings between the
    # two of them (any combination — doesn't have to be the same one twice),
    # and she's guaranteed happy to see them, not the usual grumpy/happy roll.
    if cat.state == "asleep":
        now_ts = time.time()
        author_id = message.author.id

        deep_connection_wake = False
        if author_id in DEEP_CONNECTIONS:
            dc_pending = cat.deep_connection_ping_progress
            if dc_pending is None or (now_ts - dc_pending[0]) > WAKE_PING_WINDOW:
                dc_first_at, dc_count = now_ts, 1
            else:
                dc_first_at, dc_count = dc_pending[0], dc_pending[1] + 1
            cat.deep_connection_ping_progress = (dc_first_at, dc_count)
            deep_connection_wake = dc_count >= 2

        if not deep_connection_wake:
            if cat.is_deep_sleep:
                # deep sleep: needs the SAME person 3x
                pending = cat.per_person_wake_pings.get(author_id)
                if pending is None or (now_ts - pending[0]) > WAKE_PING_WINDOW:
                    first_at, count = now_ts, 1
                else:
                    first_at, count = pending[0], pending[1] + 1

                if count < 3:
                    cat.per_person_wake_pings[author_id] = (first_at, count)
                    if count == 1:
                        await message.channel.send(random.choice(DEEP_SLEEP_LINES))
                    else:
                        await cat._set("asleep", discord.Status.idle)  # stirring, not awake yet
                        await message.channel.send(add_tired_kaomoji(random.choice(MUMBLE_LINES)).lower())
                    return
            else:
                # napping: any combination of 3 pings wakes her
                pending = cat.wake_ping_progress
                if pending is None or (now_ts - pending[0]) > WAKE_PING_WINDOW:
                    first_at, count = now_ts, 1
                else:
                    first_at, count = pending[0], pending[1] + 1

                if count < 3:
                    cat.wake_ping_progress = (first_at, count)
                    await cat._set("asleep", discord.Status.idle)  # stirring, not awake yet
                    await message.channel.send(add_tired_kaomoji(random.choice(MUMBLE_LINES)).lower())
                    return

        # she's waking up now — either the deep-connections override, or a
        # completed same-person (deep sleep) / any-combination (nap) count
        cat.wake_ping_progress = None
        cat.per_person_wake_pings = {}
        cat.deep_connection_ping_progress = None
        mood = "waking_happy" if deep_connection_wake else "waking"
        fallback = ("mmn... it's you? okay, I'm up~ (=^･ω･^=)" if deep_connection_wake
                    else "nyaa?! okay okay, I'm awake, I'm awake!")
        try:
            reply = await ask_irem(message.channel.id, prompt, author_id, mood=mood, mentioned_deep_connections=dc_mentioned, image_parts=image_parts, ambient_context=ambient_ctx, emote_aside=emote_aside, author_name=message.author.display_name)
            if not reply:
                reply = fallback
        except Exception as e:
            log_gemini_error(e)
            reply = fallback
        await cat._set("awake", discord.Status.online, "just woke up~")
        await message.reply(strip_pingable_syntax(reply)[:2000].lower())
        return

    # ---- DROWSY: answers one person, then quiet for 5 minutes ----
    if cat.state == "drowsy":
        now = time.time()
        if now - cat.last_drowsy_reply < DROWSY_COOLDOWN:
            return  # she's drifting off, ignores everyone for now
        cat.last_drowsy_reply = now
        async with message.channel.typing():
            try:
                reply = await ask_irem(message.channel.id, prompt, message.author.id, mood="drowsy", mentioned_deep_connections=dc_mentioned, image_parts=image_parts, ambient_context=ambient_ctx, emote_aside=emote_aside, author_name=message.author.display_name)
                if not reply:
                    reply = add_tired_kaomoji(random.choice(TIRED_LINES))
            except Exception as e:
                log_gemini_error(e)
                reply = add_tired_kaomoji(random.choice(TIRED_LINES))
        await message.reply(strip_pingable_syntax(reply)[:2000].lower())
        return

    # ---- STRETCHING: just woke up on her own, groggy-but-fine reply ----
    if cat.state == "stretching":
        async with message.channel.typing():
            try:
                reply = await ask_irem(message.channel.id, prompt, message.author.id, mood="stretching", mentioned_deep_connections=dc_mentioned, image_parts=image_parts, ambient_context=ambient_ctx, emote_aside=emote_aside, author_name=message.author.display_name)
                if not reply:
                    reply = random.choice(STRETCH_FALLBACK_LINES)
            except Exception as e:
                log_gemini_error(e)
                reply = random.choice(STRETCH_FALLBACK_LINES)
        await message.reply(strip_pingable_syntax(reply)[:2000].lower())
        return

    # ---- AWAKE: normal reply ----
    now = time.time()
    if now - last_awake_reply[message.author.id] < AWAKE_REPLY_COOLDOWN:
        return
    last_awake_reply[message.author.id] = now

    async with message.channel.typing():
        try:
            reply = await ask_irem(message.channel.id, prompt, message.author.id, mood="awake", mentioned_deep_connections=dc_mentioned, image_parts=image_parts, ambient_context=ambient_ctx, emote_aside=emote_aside, author_name=message.author.display_name)
            if not reply:
                reply = add_tired_kaomoji(random.choice(TIRED_LINES))
        except Exception as e:
            log_gemini_error(e)
            reply = add_tired_kaomoji(random.choice(TIRED_LINES))

    await message.reply(strip_pingable_syntax(reply)[:2000].lower())


client.run(os.environ["DISCORD_TOKEN"])