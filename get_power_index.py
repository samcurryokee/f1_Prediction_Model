#!/usr/bin/env python
"""
Measure how power-hungry each F1 circuit is, from FastF1 lap telemetry.

For every circuit we take the fastest qualifying lap (a clean, flat-out lap) and record
  * full_throttle_share : fraction of the lap spent with the throttle (almost) fully open
  * top_speed_kph       : highest speed reached on that lap

A circuit with a high full-throttle share rewards engine power and low drag.
The notebook reads the resulting CSV and splits circuits into low / medium / high thirds.

Only races BEFORE --before are measured, so the race you are predicting never feeds
its own features. The newest races come first, and we stop measuring a circuit once we
have --samples-per-circuit laps for it (so ~25 circuits x 2 = ~50 sessions, not ~190).

Usage
    python get_power_index.py --before 2026-09-26
    python get_power_index.py --before 2026-09-26 --samples-per-circuit 1

The script is resumable: re-run it and it skips sessions already in the CSV.
Downloads are cached by FastF1 in ./f1_telemetry_cache (this can reach a few GB).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FULL_THROTTLE_PCT = 98          # throttle at or above this counts as "flat out"
MIN_SAMPLES = 50                # ignore laps with too few telemetry points


def full_throttle_share(car):
    """Fraction of lap TIME spent flat out. Telemetry samples are unevenly spaced,
    so each sample is weighted by the time until the next one."""
    time_col = 'SessionTime' if 'SessionTime' in car.columns else 'Time'
    t = car[time_col].dt.total_seconds().to_numpy(dtype=float)
    throttle = car['Throttle'].to_numpy(dtype=float)
    dt = np.diff(t)
    if len(dt) < MIN_SAMPLES or dt.sum() <= 0:
        return np.nan
    return float((dt * (throttle[:-1] >= FULL_THROTTLE_PCT)).sum() / dt.sum())


def measure_session(session):
    """Return (full_throttle_share, top_speed_kph) for the session's fastest lap, or None."""
    lap = session.laps.pick_fastest()
    if lap is None:
        return None
    car = lap.get_car_data()
    share = full_throttle_share(car)
    if np.isnan(share):
        return None
    return share, float(car['Speed'].max())


def pick_targets(schedule, before, done, samples_per_circuit):
    """Newest-first list of (season, round, circuitId) still to measure."""
    sched = schedule.copy()
    sched['raceDate'] = pd.to_datetime(sched['raceDate']).dt.tz_localize(None)
    sched = sched.loc[sched['raceDate'] < pd.Timestamp(before)]
    sched = sched.sort_values('raceDate', ascending=False)
    have = {}
    for circuit in done['circuitId']:
        have[circuit] = have.get(circuit, 0) + 1
    done_keys = set(zip(done['season'], done['round']))
    targets = []
    for row in sched.itertuples():
        key = (int(row.season), int(row.round))
        if key in done_keys or have.get(row.circuitId, 0) >= samples_per_circuit:
            continue
        have[row.circuitId] = have.get(row.circuitId, 0) + 1
        targets.append((key[0], key[1], row.circuitId))
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--before', required=True, help='only measure races before this date (YYYY-MM-DD)')
    parser.add_argument('--start-year', type=int, default=2018)
    parser.add_argument('--end-year', type=int, default=None, help='default: year of --before')
    parser.add_argument('--samples-per-circuit', type=int, default=2)
    parser.add_argument('--out', default='circuit_power_index.csv')
    parser.add_argument('--cache', default='f1_telemetry_cache')
    args = parser.parse_args()

    import fastf1
    from fastf1.ergast import Ergast

    Path(args.cache).mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)
    api = Ergast(limit=100)
    end_year = args.end_year or pd.Timestamp(args.before).year

    out = Path(args.out)
    cols = ['circuitId', 'season', 'round', 'session', 'full_throttle_share', 'top_speed_kph']
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=cols)

    schedules = [api.get_race_schedule(season=y) for y in range(args.start_year, end_year + 1)]
    schedule = pd.concat(schedules, ignore_index=True)
    targets = pick_targets(schedule, args.before, done, args.samples_per_circuit)
    print(f'{len(targets)} sessions to measure ({len(done)} already in {out})')

    for n, (season, rnd, circuit) in enumerate(targets, start=1):
        result, used = None, None
        for identifier in ('Q', 'R'):            # qualifying first; race lap as a fallback
            try:
                session = fastf1.get_session(season, rnd, identifier)
                session.load(laps=True, telemetry=True, weather=False, messages=False)
                result = measure_session(session)
                used = identifier
            except Exception as exc:              # missing telemetry is common for some sessions
                print(f'  ! {season} R{rnd} {identifier}: {type(exc).__name__}: {exc}', file=sys.stderr)
            if result:
                break
            time.sleep(1)
        if not result:
            print(f'[{n}/{len(targets)}] {season} R{rnd} {circuit}: no usable telemetry, skipped')
            continue
        row = dict(zip(cols, [circuit, season, rnd, used, round(result[0], 4), round(result[1], 1)]))
        new_row = pd.DataFrame([row])
        done = new_row if done.empty else pd.concat([done, new_row], ignore_index=True)
        done.to_csv(out, index=False)             # save after every session so nothing is lost
        print(f'[{n}/{len(targets)}] {season} R{rnd} {circuit}: '
              f'{row["full_throttle_share"]:.1%} full throttle, top speed {row["top_speed_kph"]:.0f} km/h')

    if len(done):
        summary = done.groupby('circuitId').full_throttle_share.mean().sort_values(ascending=False)
        print('\nFull-throttle share by circuit (highest = most power-hungry):')
        print(summary.map('{:.1%}'.format).to_string())


if __name__ == '__main__':
    main()
