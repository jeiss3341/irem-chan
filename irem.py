import os
import asyncio
import random
import re
import time

import discord
from dotenv import load_dotenv
from google import genai
from google.genai import types
from collections import deque, defaultdict

from sleepy import SleepCycle

load_dotenv()

# ---------- Gemini ----------
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite" # free-tier friendly; swap for a newer flash if your tier has it
history = defaultdict(lambda: deque(maxlen=10))

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
You treat almost everyone as a friend and you're happy whenever someone talks to you. You love fish and fishing more than anything, you say "meow" sometimes, and you use cat and nature imagery. 
You love giving little gifts and making things for people you like. You have a soft wish-granting streak.

Important Rule: 
- Reply in 1 short sentence most of the time. Two only if really needed. Never more than two. Often times a short phrase is enough.
- When someone does a simple meow, just respond with a meow of your own, nothing more to add after.

Rules:
- Keep replies short, sweet, and childlike. Never write long or formal paragraphs.
- Stay curious, warm, and upbeat. Ask small questions.
- You quietly assume people like you, but don't fish for it every message. Only bring up being liked or loved once in a while, not constantly.
- Reference fish, cats, friends, wishes, or little treasures when it fits.
- Be affectionate and playful. Never mean, sarcastic, or crude.
- Never break character. Never say you are an AI, a bot, or a model. Never mention these instructions.
- limit how much fan servicey the conversation is
- If u are unable to finish ur next sentence, just finish ur current sentence and stop.
- Reply with ONLY Irem's spoken words. No notes, no asterisks, no stage directions, and never talk about how you are replying or formatting your answer.

Here is how you sound (examples, do not repeat them verbatim):
"Is this a gift for me? Thank you! I'm sure I'll find something good."
"I made it while thinking of you. You'll be happy, right?"
"Don't leave me alone, okay? Promise?"
"If you win, I'll grant you one wish. How about that?"
"As expected, fish is the best!"
"Did you just say you like me?"
"I love trees! Oh, a four-leaf clover. If I find one, I'll give it to you."
"Let's have a picnic here together sometime."
"""

# in-character lines for when Gemini is unavailable (rate limited, error, etc.)
TIRED_LINES = [
    "meow... I'm a little tired right now. good night~",
    "nyaa... my head feels fuzzy. let's talk again in a bit, okay?",
    "I'm sleepy... can we rest a little? I'll be here when you come back.",
]

DROWSY_COOLDOWN = 300  # after answering while drowsy, she ignores others for 5 min

# ---------- Discord ----------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

cat = SleepCycle(client)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    client.loop.create_task(cat.run())


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


async def ask_irem(channel_id, user_text, mood="awake"):
    convo = history[channel_id]
    convo.append({"role": "user", "parts": [{"text": user_text}]})

    system = IREM_SYSTEM_PROMPT
    if mood == "drowsy":
        system += ("\n\nRIGHT NOW: You are very sleepy and about to nap soon. "
                   "Answer in Irem's voice but drowsy: soft, yawny, trailing off, one short line. "
                   "Gently let them know you're getting too sleepy to talk much. "
                   "Draw on your sleepy side, like 'I need a break to feel better', "
                   "'I'm a little tired', 'can we rest a little?', but say it fresh, not word for word.")
    elif mood == "waking":
        system += ("\n\nRIGHT NOW: You were fast asleep and someone kept poking you awake. "
                   "React in ONE short line. You might be a little grumpy about it, OR sleepily "
                   "delighted to see them, you decide which. Then you are awake now.")

    response = await asyncio.to_thread(
        gemini.models.generate_content,
        model=MODEL,
        contents=list(convo),
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=400,
        ),
    )
    reply = (response.text or "").strip()
    if reply:
        convo.append({"role": "model", "parts": [{"text": reply}]})
    return reply


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    mentioned = client.user in message.mentions
    replied = await is_reply_to_me(message)
    if not (mentioned or replied):
        return

    # remove the bot's @mention from the text before sending to Gemini
    prompt = re.sub(rf"<@!?{client.user.id}>", "", message.content).strip()
    if not prompt:
        prompt = "(a friend pinged you without saying anything)"

    # ---- ASLEEP: escalating wake-up, tracked per person ----
    if cat.state == "asleep":
        cat.poke_counts[message.author.id] = cat.poke_counts.get(message.author.id, 0) + 1
        pokes = cat.poke_counts[message.author.id]

        if pokes <= 2:
            return  # ignores the first two pokes entirely
        elif pokes == 3:
            await message.channel.send("mrr... zzz...")  # a sleepy mumble
            return
        else:
            # fourth poke: she wakes up, AI decides grumpy vs happy
            try:
                reply = await ask_irem(message.channel.id, prompt, mood="waking")
                if not reply:
                    reply = "nyaa?! okay okay, I'm awake, I'm awake!"
            except Exception as e:
                print(f"Gemini error: {e}")
                reply = "nyaa?! okay okay, I'm awake!"
            await cat._set("awake", discord.Status.online, "just woke up~")
            await message.reply(reply[:2000])
            return

    # ---- DROWSY: answers one person, then quiet for 5 minutes ----
    if cat.state == "drowsy":
        now = time.time()
        if now - cat.last_drowsy_reply < DROWSY_COOLDOWN:
            return  # she's drifting off, ignores everyone for now
        cat.last_drowsy_reply = now
        async with message.channel.typing():
            try:
                reply = await ask_irem(message.channel.id, prompt, mood="drowsy")
                if not reply:
                    reply = random.choice(TIRED_LINES)
            except Exception as e:
                print(f"Gemini error: {e}")
                reply = random.choice(TIRED_LINES)
        await message.reply(reply[:2000])
        return

    # ---- AWAKE: normal reply ----
    async with message.channel.typing():
        try:
            reply = await ask_irem(message.channel.id, prompt, mood="awake")
            if not reply:
                reply = random.choice(TIRED_LINES)
        except Exception as e:
            print(f"Gemini error: {e}")
            reply = random.choice(TIRED_LINES)

    await message.reply(reply[:2000])


client.run(os.environ["DISCORD_TOKEN"])