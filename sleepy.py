import asyncio
import random
import datetime
import discord

# ---------- Sleep schedule (server local time, 24h) ----------
# On Railway the server runs in UTC. To use your time, add a Railway
# variable:  TZ = America/New_York
#
# She's a kitten with energy, not a mellow adult cat: total sleep is well
# under a real cat's 12-16 hrs/day, and sleep sessions are short.
#
# NOTE: TOTAL_SLEEP_HOURS_* is how much she sleeps per SLEEP CYCLE, not per
# real calendar day — a cycle is "keep napping + waking until that many
# hours of sleep have accumulated," and each nap is followed by a real
# AWAKE_MIN/MAX_MINUTES-long stretch. If the awake stretches are long
# relative to the naps, one cycle's worth of sleep ends up spread across
# MORE than 24 real hours, and she sleeps noticeably less than the target
# suggests. These numbers are tuned so a cycle completes in ~1 real day —
# verified with a Monte Carlo simulation (200 simulated days), not just
# arithmetic: ~7.7 actual hrs/real-day, ~12 naps/day. If you change
# SHORT_NAP/DEEP_SLEEP/DEEP_SLEEP_CHANCE/AWAKE_*, re-simulate rather than
# assuming — the real-day sleep total depends on all of them together.
TOTAL_SLEEP_HOURS_MIN = 6
TOTAL_SLEEP_HOURS_MAX = 10

SHORT_NAP_MIN_MINUTES = 15
SHORT_NAP_MAX_MINUTES = 40
DEEP_SLEEP_MIN_MINUTES = 60
DEEP_SLEEP_MAX_MINUTES = 100
DEEP_SLEEP_CHANCE = 0.2    # this fraction of sleep sessions are a longer deep sleep

AWAKE_MIN_MINUTES = 35     # short kitten bursts of energy between naps, not hours-long stretches
AWAKE_MAX_MINUTES = 100

# dawn/dusk local hours: real cats are more active then, so awake stretches run longer
DAWN_HOURS = (5, 8)
DUSK_HOURS = (18, 21)
CREPUSCULAR_AWAKE_MULTIPLIER = 1.5

DROWSY_LEAD_MINUTES = 4     # heads-up before any sleep session, so she doesn't just vanish mid-chat
WAKE_STRETCH_MINUTES = 2    # stretch/yawn moment right after waking up — long enough to realistically get pinged

# If woken during a sleep session, she stays up this long (random), then dozes off
WOKEN_MIN_MINUTES = 5
WOKEN_MAX_MINUTES = 25

# How often her awake status reshuffles on its own (random, in this range)
STATUS_SHUFFLE_MIN_HOURS = 1
STATUS_SHUFFLE_MAX_HOURS = 6

# Cute random statuses she shows while awake. Add or remove freely.
AWAKE_STATUSES = [
    "playing~ nya",
    "looking for fish",
    "meow meow",
    "waiting for friends",
    "chasing butterflies",
    "counting little fishies",
    "making a gift for you",
    "napping in a sunbeam",
    "collecting shiny shells",
    "found a four-leaf clover!",
    "thinking about snacks",
    "practicing my fishing",
    "who wants to play?",
    "guarding my treasures",
    "wishing on a star",
    "chasing my own tail",
    "picking pretty flowers",
    "waiting by the pond",
    "purring softly~",
    "looking for the biggest fish",
    "daydreaming about tuna",
    "batting at a yarn ball",
    "sunbathing by the window",
    "sniffing the sea breeze",
    "saving a snack for you",
    "watching the clouds go by",
    "hunting for four-leaf clovers",
    "listening to the waves",
    "hoping a friend visits",
    "keeping a wish safe for you",
    "fishing for the biggest catch",
    "nya~ where did that fish go",
    "drawing you a little picture",
    "humming a happy song",
    "waiting for a friend to play",
    "building a tiny sandcastle",
    "counting stars for wishes",
    "wrapping up a surprise~",
    "looking for the softest spot",
    "chasing a red dragonfly",
    "meow? did someone call me?",
    "keeping the pond company",
    "petting a friendly ladybug",
    "dreaming of a big tuna",
    "pressing flowers in a book",
    "guarding a shiny pebble",
    "waiting to grant a wish",
    "following a butterfly home",
    "making a crown of clovers",
    "poking at my reflection~",
    "saving the fluffiest cloud",
    "whispering to the fishies",
    "waiting for a picnic~",
    "collecting morning dew",
    "purr... this sunbeam is warm",
    "counting my little treasures",
    "hoping you had a good day",
    "wishing on a dandelion",
    "napping between the flowers",
    "watching fish jump in the pond",
    "granting a tiny wish~",
    "meow meow, come play!",
    "found the prettiest seashell",
    "waiting to share my fish",
    "making a flower crown for you",
    "did you bring me a treat?",
    "everyone loves me, right? nya",
    "counting fish in the stream",
    "saving the best clover for you",
    "purring by the warm rocks",
    "chasing ripples in the water",
    "picking berries for a friend",
    "nya~ I caught a little minnow",
    "watching tadpoles wiggle",
    "wishing everyone sweet dreams",
    "tucking away a shiny coin",
    "waiting for someone to pet me",
    "humming by the riverbank",
    "collecting the roundest pebbles",
    "sharing snacks with the birds",
    "you'll play with me, right?",
    "sniffing a brand new flower",
    "keeping your secret safe~",
    "catching sunbeams in my paws",
    "hoping for a big fish today",
    "meow, is it snack time yet?",
    "leaving a gift where you'll find it",
    "twirling with the falling leaves",
    "listening to the crickets sing",
    "waiting under the big tree",
    "making a wish just for you",
    "chasing the last firefly",
    "drying my paws in the sun",
    "poking a floating leaf~",
    "saving a wish for a rainy day",
    "meow? I smell something tasty",
    "braiding grass into a bracelet",
    "peeking for shooting stars",
    "keeping the fireflies company",
    "warming up on a flat stone",
]


def pick_awake_status():
    return random.choice(AWAKE_STATUSES)


def _in_crepuscular_window():
    hour = datetime.datetime.now().hour
    dawn_start, dawn_end = DAWN_HOURS
    dusk_start, dusk_end = DUSK_HOURS
    return dawn_start <= hour < dawn_end or dusk_start <= hour < dusk_end


class SleepCycle:
    """Manages Irem's awake / drowsy / stretching / asleep state and Discord presence."""

    def __init__(self, client):
        self.client = client
        self.state = "awake"          # "awake" | "drowsy" | "asleep" | "stretching"
        self.status_text = None       # her current Discord activity text, so she can answer truthfully if asked
        self.pending_wake_pings = {}  # per-person: (timestamp of 1st ping in window, ping count)
        self.last_drowsy_reply = 0.0  # timestamp of her last drowsy reply
        self.started = False          # guards against on_ready re-firing run() after a reconnect

    async def _set(self, state, status, activity_text=None):
        self.state = state
        self.status_text = activity_text
        print(f"[sleep] -> {state}")
        if activity_text:
            await self.client.change_presence(
                status=status,
                activity=discord.CustomActivity(name=activity_text),
            )
        else:
            await self.client.change_presence(status=status)

    async def _shuffle_awake_status(self):
        """Every few hours, if she's awake, give her a fresh little status."""
        await self.client.wait_until_ready()
        while not self.client.is_closed():
            wait_hours = random.uniform(STATUS_SHUFFLE_MIN_HOURS, STATUS_SHUFFLE_MAX_HOURS)
            await asyncio.sleep(wait_hours * 3600)
            # only touch presence when she's actually up; leave drowsy/asleep/stretching alone
            if self.state == "awake":
                await self._set("awake", discord.Status.online, pick_awake_status())

    async def _sleep_wakeable(self, total_seconds):
        """Sleep for total_seconds, but if a poke sets her 'awake', let her be up
        a random short while, then get drowsy and doze off again, until the time
        is used up."""
        slept = 0
        while slept < total_seconds:
            await asyncio.sleep(20)
            slept += 20
            if self.state == "awake":
                up_for = random.randint(WOKEN_MIN_MINUTES, WOKEN_MAX_MINUTES) * 60
                up_slept = 0
                while up_slept < up_for and self.state == "awake":
                    await asyncio.sleep(20)
                    up_slept += 20
                slept += up_slept
                if slept < total_seconds:
                    await self._set("drowsy", discord.Status.idle, "getting sleepy again...")
                    await asyncio.sleep(DROWSY_LEAD_MINUTES * 60)
                    slept += DROWSY_LEAD_MINUTES * 60
                    self.pending_wake_pings.clear()
                    await self._set("asleep", discord.Status.invisible)

    async def _wake_up(self):
        """Stretch and yawn, then settle into awake."""
        await self._set("stretching", discord.Status.idle, "*stretches and yawns*")
        await asyncio.sleep(WAKE_STRETCH_MINUTES * 60)
        await self._set("awake", discord.Status.online, pick_awake_status())

    async def run(self):
        await self.client.wait_until_ready()
        # background task: reshuffles her awake status every 1-6 hours
        self.client.loop.create_task(self._shuffle_awake_status())
        await self._set("awake", discord.Status.online, pick_awake_status())

        while not self.client.is_closed():
            # pick a fresh daily sleep target (not tied to calendar midnight —
            # crepuscular timing is read from the real clock each time, so this
            # cycle can run longer or shorter than 24h without drifting wrong)
            sleep_target = random.uniform(TOTAL_SLEEP_HOURS_MIN, TOTAL_SLEEP_HOURS_MAX) * 3600
            slept = 0

            while slept < sleep_target:
                # ----- AWAKE stretch -----
                awake_minutes = random.uniform(AWAKE_MIN_MINUTES, AWAKE_MAX_MINUTES)
                if _in_crepuscular_window():
                    awake_minutes *= CREPUSCULAR_AWAKE_MULTIPLIER
                await asyncio.sleep(awake_minutes * 60)

                # ----- DROWSY heads-up -----
                await self._set("drowsy", discord.Status.idle, "getting sleepy...")
                await asyncio.sleep(DROWSY_LEAD_MINUTES * 60)

                # ----- SLEEP (wakeable) -----
                if random.random() < DEEP_SLEEP_CHANCE:
                    session_minutes = random.uniform(DEEP_SLEEP_MIN_MINUTES, DEEP_SLEEP_MAX_MINUTES)
                else:
                    session_minutes = random.uniform(SHORT_NAP_MIN_MINUTES, SHORT_NAP_MAX_MINUTES)

                self.pending_wake_pings.clear()
                await self._set("asleep", discord.Status.invisible)
                await self._sleep_wakeable(session_minutes * 60)
                slept += session_minutes * 60

                # ----- WAKE UP (stretch, then awake) -----
                await self._wake_up()
            # today's sleep target is used up; loop back and pick a fresh one
