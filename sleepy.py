import asyncio
import random
import datetime
import discord

# ---------- Sleep schedule (server local time, 24h) ----------
# On Railway the server runs in UTC. To use your time, add a Railway
# variable:  TZ = America/New_York
NIGHT_SLEEP_START = 1     # hour she goes to sleep (1 = 1am)
NIGHT_SLEEP_HOURS = 8     # roughly how long she sleeps at night
SLEEP_JITTER_MIN  = 60    # +/- up to this many minutes, so bedtime isn't robotic
DROWSY_LEAD_HOURS = 1     # how long before sleep she gets drowsy

NAP_CHANCE      = 0.4     # chance she takes one daytime nap (0.4 = 40%)
NAP_MIN_MINUTES = 30
NAP_MAX_MINUTES = 90


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

    async def run(self):
        await self.client.wait_until_ready()
        while not self.client.is_closed():
            jitter = random.randint(-SLEEP_JITTER_MIN, SLEEP_JITTER_MIN)
            bedtime = seconds_until(NIGHT_SLEEP_START) + jitter * 60
            if bedtime < 0:
                bedtime = seconds_until(NIGHT_SLEEP_START)
            drowsy_at = max(0, bedtime - DROWSY_LEAD_HOURS * 3600)

            # ----- AWAKE (with a possible daytime nap) -----
            await self._set("awake", discord.Status.online, "playing~ nya")

            if random.random() < NAP_CHANCE and drowsy_at > 3600:
                await asyncio.sleep(random.uniform(0.3, 0.7) * drowsy_at)
                self.poke_counts.clear()
                await self._set("asleep", discord.Status.invisible)
                await asyncio.sleep(random.randint(NAP_MIN_MINUTES, NAP_MAX_MINUTES) * 60)
                await self._set("awake", discord.Status.online, "back from a nap~")
                drowsy_at = max(0, seconds_until(NIGHT_SLEEP_START)
                                - DROWSY_LEAD_HOURS * 3600 + jitter * 60)

            await asyncio.sleep(drowsy_at)

            # ----- DROWSY -----
            await self._set("drowsy", discord.Status.idle, "getting sleepy...")
            await asyncio.sleep(DROWSY_LEAD_HOURS * 3600)

            # ----- NIGHT SLEEP -----
            self.poke_counts.clear()
            await self._set("asleep", discord.Status.invisible)
            await asyncio.sleep(max(3600, NIGHT_SLEEP_HOURS * 3600 + jitter * 60))
            # loop back to awake for the new day