import asyncio
import random
import datetime
import discord

# ---------- Sleep schedule (server local time, 24h) ----------
# On Railway the server runs in UTC. To use your time, add a Railway
# variable:  TZ = America/New_York
NIGHT_SLEEP_START = 2     # hour she goes to sleep (1 = 1am)
NIGHT_SLEEP_HOURS = 9     # roughly how long she sleeps at night
SLEEP_JITTER_MIN  = 60    # +/- up to this many minutes, so bedtime isn't robotic
DROWSY_LEAD_HOURS = 1     # how long before sleep she gets drowsy

NAPS_MIN        = 2       # fewest daytime naps
NAPS_MAX        = 4       # most daytime naps
NAP_MIN_MINUTES = 30
NAP_MAX_MINUTES = 90

# If woken during a sleep (nap or night), she stays up this long (random), then dozes off
WOKEN_MIN_MINUTES = 5
WOKEN_MAX_MINUTES = 25

# How often her awake status reshuffles on its own (random, in this range)
STATUS_SHUFFLE_MIN_HOURS = 3
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


def seconds_until(hour, minute=0):
    """Seconds from now until the next time it is hour:minute locally."""
    now = datetime.datetime.now()
    target = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += datetime.timedelta(days=1)
    return (target - now).total_seconds()


class SleepCycle:
    """Manages Irem's awake / drowsy / asleep state and Discord presence."""

    def __init__(self, client):
        self.client = client
        self.state = "awake"          # "awake" | "drowsy" | "asleep"
        self.poke_counts = {}         # per-person poke count during current sleep
        self.last_drowsy_reply = 0.0  # timestamp of her last drowsy reply

    async def _set(self, state, status, activity_text=None):
        self.state = state
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
            # only touch presence when she's actually up; leave drowsy/asleep alone
            if self.state == "awake":
                await self.client.change_presence(
                    status=discord.Status.online,
                    activity=discord.CustomActivity(name=pick_awake_status()),
                )

    async def _sleep_wakeable(self, total_seconds):
        """Sleep for total_seconds, but if a poke sets her 'awake', let her be up
        a random short while, then doze off again, until the time is used up."""
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
                    self.poke_counts.clear()
                    await self._set("asleep", discord.Status.invisible)

    async def run(self):
        await self.client.wait_until_ready()
        # background task: reshuffles her awake status every 3-8 hours
        self.client.loop.create_task(self._shuffle_awake_status())
        while not self.client.is_closed():
            jitter = random.randint(-SLEEP_JITTER_MIN, SLEEP_JITTER_MIN)
            bedtime = seconds_until(NIGHT_SLEEP_START) + jitter * 60
            if bedtime < 0:
                bedtime = seconds_until(NIGHT_SLEEP_START)

            # ----- AWAKE (with a few spread-out daytime naps) -----
            await self._set("awake", discord.Status.online, pick_awake_status())

            naps_today = random.randint(NAPS_MIN, NAPS_MAX)
            for i in range(naps_today):
                # how much awake time is left before she needs to get drowsy
                remaining = (seconds_until(NIGHT_SLEEP_START) + jitter * 60
                             - DROWSY_LEAD_HOURS * 3600)
                # stop napping if there isn't comfortable room left in the day
                if remaining < NAP_MAX_MINUTES * 60 + 1800:
                    break
                # split the remaining day into one chunk per remaining nap, then
                # nap somewhere in the first part of this chunk (keeps them spaced)
                chunk = remaining / (naps_today - i)
                await asyncio.sleep(random.uniform(0.3, 0.7) * chunk)

                self.poke_counts.clear()
                await self._set("asleep", discord.Status.invisible)
                nap_total = random.randint(NAP_MIN_MINUTES, NAP_MAX_MINUTES) * 60
                await self._sleep_wakeable(nap_total)   # naps are wakeable too
                await self._set("awake", discord.Status.online, pick_awake_status())

            # wait out the rest of the day until she gets drowsy
            drowsy_at = max(0, seconds_until(NIGHT_SLEEP_START) + jitter * 60
                            - DROWSY_LEAD_HOURS * 3600)
            await asyncio.sleep(drowsy_at)

            # ----- DROWSY -----
            await self._set("drowsy", discord.Status.idle, "getting sleepy...")
            await asyncio.sleep(DROWSY_LEAD_HOURS * 3600)

            # ----- NIGHT SLEEP (wakeable) -----
            self.poke_counts.clear()
            await self._set("asleep", discord.Status.invisible)
            night_total = int(max(3600, NIGHT_SLEEP_HOURS * 3600 + jitter * 60))
            await self._sleep_wakeable(night_total)
            # morning: loop back to awake for the new day