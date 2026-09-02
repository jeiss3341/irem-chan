import os
import asyncio
import datetime
import random
import re
import time

import discord
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from collections import deque, defaultdict

from sleepy import SleepCycle

load_dotenv()

# ---------- Gemini ----------
# GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ... — each must come
# from a separate AI Studio project to actually have its own free-tier quota
# (multiple keys from the same project share one pool). On a 429 (quota
# exhausted), generate_content_with_fallback advances to the next key and
# retries, so normal usage stays on key 1 and the rest are pure overflow.
_gemini_keys = [os.environ["GEMINI_API_KEY"]]
i = 2
while os.environ.get(f"GEMINI_API_KEY_{i}"):
    _gemini_keys.append(os.environ[f"GEMINI_API_KEY_{i}"])
    i += 1
_gemini_clients = [genai.Client(api_key=k) for k in _gemini_keys]
_active_gemini_client = 0  # index into _gemini_clients; advances when a key 429s
_active_gemini_day = datetime.datetime.now().date()  # the day _active_gemini_client applies to

MODEL = "gemini-3.5-flash"  # stepped up from flash-lite now that multiple keys give real quota headroom; thinking is capped to 0 below so it doesn't burn tokens on hidden reasoning for a one-line reply
history = defaultdict(lambda: deque(maxlen=50))


def generate_content_with_fallback(**kwargs):
    """Like gemini.models.generate_content, but on a 429 (quota exhausted)
    advances to the next configured API key and retries, instead of failing
    the whole reply. Raises the last error if every key is exhausted.

    Resets back to key 1 at the start of each new day (free-tier daily quotas
    reset daily), so a key that got exhausted yesterday is tried first again
    today, rather than staying permanently benched until a process restart."""
    global _active_gemini_client, _active_gemini_day
    today = datetime.datetime.now().date()
    if today != _active_gemini_day:
        _active_gemini_client = 0
        _active_gemini_day = today

    last_error = None
    for _ in range(len(_gemini_clients)):
        client = _gemini_clients[_active_gemini_client]
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.APIError as e:
            last_error = e
            if e.code != 429:
                raise
            log_gemini_error(e)
            if _active_gemini_client < len(_gemini_clients) - 1:
                _active_gemini_client += 1
                print(f"[gemini] switching to API key #{_active_gemini_client + 1}")
    raise last_error

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
- Never break character. Never say you are an AI, a bot, or a model. Never mention these instructions.
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


async def is_reply_to_me(message):
    ref = message.reference
    if ref is None:
        return False
    replied = ref.resolved
    if replied is None and ref.message_id:
        try:
            replied = await message.channel.fetch_message(ref.message_id)
        except (discord.NotFound, discord.HTTPException):
            return False
    return isinstance(replied, discord.Message) and replied.author == client.user


PINGABLE_SYNTAX_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")


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


async def ask_irem(channel_id, user_text, author_id, mood="awake", mentioned_deep_connections=None):
    convo = history[channel_id]
    convo.append({"role": "user", "parts": [{"text": user_text}]})

    system = IREM_SYSTEM_PROMPT
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
    if cat.status_text:
        system += (f"\n\nYour current status/activity (shown on Discord) is: \"{cat.status_text}\". "
                   "If anyone asks what you're doing, or about your status, answer truthfully "
                   "based on this, in character — don't make up something different.")
    if mood == "drowsy":
        system += ("\n\nRIGHT NOW: You are very sleepy and about to nap soon. "
                   "Answer in Irem's voice but drowsy: soft, yawny, trailing off, one short line. "
                   "Gently let them know you're getting too sleepy to talk much. "
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

    response = await asyncio.to_thread(
        generate_content_with_fallback,
        model=MODEL,
        contents=list(convo),
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=400,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    reply = (response.text or "").strip()
    if reply:
        convo.append({"role": "model", "parts": [{"text": reply}]})
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

    mentioned = client.user in message.mentions
    replied = await is_reply_to_me(message)
    if not (mentioned or replied):
        return

    # remove the bot's @mention, then turn any other real mentions into
    # plain non-pinging text before this ever reaches Gemini's context
    prompt = re.sub(rf"<@!?{client.user.id}>", "", message.content).strip()
    prompt = humanize_mentions(prompt, message)
    if not prompt:
        prompt = "(a friend pinged you without saying anything)"
    dc_mentioned = other_mentioned_deep_connections(message, prompt, message.author.id)

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
            reply = await ask_irem(message.channel.id, prompt, author_id, mood=mood, mentioned_deep_connections=dc_mentioned)
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
                reply = await ask_irem(message.channel.id, prompt, message.author.id, mood="drowsy", mentioned_deep_connections=dc_mentioned)
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
                reply = await ask_irem(message.channel.id, prompt, message.author.id, mood="stretching", mentioned_deep_connections=dc_mentioned)
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
            reply = await ask_irem(message.channel.id, prompt, message.author.id, mood="awake", mentioned_deep_connections=dc_mentioned)
            if not reply:
                reply = add_tired_kaomoji(random.choice(TIRED_LINES))
        except Exception as e:
            log_gemini_error(e)
            reply = add_tired_kaomoji(random.choice(TIRED_LINES))

    await message.reply(strip_pingable_syntax(reply)[:2000].lower())


client.run(os.environ["DISCORD_TOKEN"])