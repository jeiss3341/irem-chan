"""
Standalone Monte Carlo simulation of Irem's sleep cycle (sleepy.py's run()
loop), without asyncio/Discord — just fast-forwards through simulated time
using the real constants and logic from sleepy.py, so this can't silently
drift out of sync with the actual bot.

No pings are simulated (i.e. this measures her *baseline* sleep amount,
not what happens when people wake her early — waking her early only
shortens sleep further, so this is the upper bound on how much she sleeps).

Run: venv/bin/python3 test_sleep_sim.py
"""
import random
import statistics

import sleepy as s

SIM_DAYS = 200  # simulated real days
random.seed(42)


def in_crepuscular_window(hour):
    dawn_start, dawn_end = s.DAWN_HOURS
    dusk_start, dusk_end = s.DUSK_HOURS
    return dawn_start <= hour < dawn_end or dusk_start <= hour < dusk_end


def simulate(total_seconds):
    t = 0.0  # real elapsed seconds since sim start
    asleep_seconds = 0.0
    nap_count = 0
    deep_count = 0
    daily_sleep = []  # sleep seconds accumulated, bucketed per 24h real-time window
    day_boundary = 86400
    day_accum = 0.0

    def advance(seconds, counts_as_sleep):
        nonlocal t, asleep_seconds, day_accum, day_boundary
        remaining = seconds
        nonlocal_t = t
        while remaining > 0:
            # how much time until the next day boundary
            until_boundary = day_boundary - nonlocal_t
            chunk = min(remaining, until_boundary)
            if counts_as_sleep:
                day_accum += chunk
                asleep_seconds_local = chunk
            else:
                asleep_seconds_local = 0
            nonlocal_t += chunk
            remaining -= chunk
            if counts_as_sleep:
                pass
            if nonlocal_t >= day_boundary:
                daily_sleep.append(day_accum)
                day_accum = 0.0
                day_boundary += 86400
        return nonlocal_t

    while t < total_seconds:
        sleep_target = random.uniform(s.TOTAL_SLEEP_HOURS_MIN, s.TOTAL_SLEEP_HOURS_MAX) * 3600
        slept_this_cycle = 0.0

        while slept_this_cycle < sleep_target and t < total_seconds:
            # ----- AWAKE stretch -----
            hour = int((t / 3600) % 24)
            awake_minutes = random.uniform(s.AWAKE_MIN_MINUTES, s.AWAKE_MAX_MINUTES)
            if in_crepuscular_window(hour):
                awake_minutes *= s.CREPUSCULAR_AWAKE_MULTIPLIER
            t = advance(awake_minutes * 60, counts_as_sleep=False)

            # ----- DROWSY heads-up -----
            t = advance(s.DROWSY_LEAD_MINUTES * 60, counts_as_sleep=False)

            # ----- SLEEP session (no pings simulated -> full session used) -----
            if random.random() < s.DEEP_SLEEP_CHANCE:
                session_minutes = random.uniform(s.DEEP_SLEEP_MIN_MINUTES, s.DEEP_SLEEP_MAX_MINUTES)
                deep_count += 1
            else:
                session_minutes = random.uniform(s.SHORT_NAP_MIN_MINUTES, s.SHORT_NAP_MAX_MINUTES)
            nap_count += 1
            t = advance(session_minutes * 60, counts_as_sleep=True)
            asleep_seconds += session_minutes * 60
            slept_this_cycle += session_minutes * 60

            # ----- WAKE UP (stretch, then awake) -----
            t = advance(s.WAKE_STRETCH_MINUTES * 60, counts_as_sleep=False)

    return {
        "total_seconds": t,
        "asleep_seconds": asleep_seconds,
        "nap_count": nap_count,
        "deep_count": deep_count,
        "daily_sleep": daily_sleep,
    }


def main():
    result = simulate(SIM_DAYS * 86400)
    total_days = result["total_seconds"] / 86400
    total_hours_slept = result["asleep_seconds"] / 3600
    avg_hrs_per_day = total_hours_slept / total_days
    avg_naps_per_day = result["nap_count"] / total_days
    deep_frac = result["deep_count"] / result["nap_count"]

    daily = [d / 3600 for d in result["daily_sleep"]]  # per-24h-bucket hours slept

    print(f"Simulated {total_days:.1f} real days")
    print(f"Total naps: {result['nap_count']}  (deep sleeps: {result['deep_count']}, "
          f"{deep_frac:.1%} — target was {s.DEEP_SLEEP_CHANCE:.0%})")
    print(f"Avg naps/day: {avg_naps_per_day:.2f}")
    print(f"Avg sleep/real-day: {avg_hrs_per_day:.2f} hrs "
          f"(target per-cycle range was {s.TOTAL_SLEEP_HOURS_MIN}-{s.TOTAL_SLEEP_HOURS_MAX} hrs)")
    print(f"Per-day-bucket sleep hours: min={min(daily):.2f}, max={max(daily):.2f}, "
          f"median={statistics.median(daily):.2f}, stdev={statistics.pstdev(daily):.2f}")
    print(f"Awake hours/real-day (implied): {24 - avg_hrs_per_day:.2f}")


if __name__ == "__main__":
    main()
