"""GTA VI trophy-rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent
of the players Exophase tracks who will have earned it at the case's target,
as five quantiles (5/25/50/75/95%), from the house's recording of Exophase's
trophy page up to the case's checkpoint.

How it works
------------
Exophase's page is a snapshot: it only moves when Exophase refreshes its counts,
so the history is a staircase - long flat stretches, then jumps. Two forces move
a trophy's rate between refreshes:

* dilution - the tracked-player base grows (launch surge), which pushes every
  rate down, and
* adoption - players keep earning trophies, which pushes rates up.

Early on dilution wins (the first refresh cut every rate); later, as the influx
slows, adoption wins and rates climb toward each trophy's long-run level. A
flat forecast (the house baseline) misses the climb; a straight line in logit
space overshoots it, because adoption saturates - a trophy's logit rate grows
roughly linearly in log-time while players are still arriving, then flattens
as it nears its long-run level.

So, per trophy, this program:

1. collapses the history to its refresh events (reads where the page actually
   changed) - the flat stretches carry no information;
2. fits, for each of several log-time scales tau, a recency-weighted line of
   logit(rate) against x = ln(1 + hours-since-launch / tau), using only the
   last day's refresh events. A small tau saturates fast (the slope decays
   quickly); a large tau extrapolates almost linearly in time. No single tau
   is right for every trophy, and the history is too short to tell which, so
   the forecast averages the medians across the scales instead of betting on
   one - and the disagreement between scales widens the spread, honestly;
3. shrinks each trophy's slope toward its tier's median slope and toward zero,
   with the shrinkage set by how many refresh events the trophy has and how
   noisy its fit is - a trophy with two points does not get to claim a trend;
4. extrapolates each scale to the target with the slope damped, then blends
   the result part-way back toward the last observed rate, a hedge against
   the trend turning (a fresh dilution wave) over the horizon;
5. sets the spread from four parts: the random-walk drift the house baseline
   uses, the trophy's own fit residual (how jumpy its refreshes have been,
   grown with the horizon), a model-uncertainty term, and the disagreement
   between the time scales. When the history holds almost no refresh events -
   a flat page that has not moved yet - the spread is floored wide, because
   the one thing the history cannot show is the next refresh.

The platinum trophy is capped at the lowest other median: nobody can hold the
platinum without every other trophy.

Standard library only, no network, deterministic: the same input bytes always
produce the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch-day start: the time base for log-time. The game released 2026-11-19.
LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# Log-time scales, hours: small tau saturates fast, large tau stays near-linear.
TAU_GRID = (1.0, 6.0, 24.0, 72.0, 168.0)
WINDOW_H = 24.0           # only refresh events from the last WINDOW_H hours enter a fit
HALF_LIFE_H = 12.0        # recency half-life for the trend fit, hours
W_TIER = 2.0              # shrinkage weight: the tier-median slope
W_ZERO = 3.0              # shrinkage weight: a flat slope
W_FLAT = 0.35             # blend of the extrapolated median back to the last rate
DAMP = 0.7                # damping on the extrapolated logit increment
SIGMA_RW = 0.5            # random-walk spread, logit units per sqrt(day)
SIGMA_EXTRA = 0.25        # model-uncertainty spread, logit units per sqrt(day)
RESID_FLOOR = 0.25        # smallest fit-residual spread we trust, logits
RESID_CAP = 1.2           # largest fit-residual spread we trust, logits
SIGMA_FLOOR_FLAT = 2.0    # spread floor when the history barely moved
SIGMA_CAP = 3.0           # widest spread we will ever quote, logits
SPREAD_SCALE = 0.7        # overall calibration of the quoted spread


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the house baseline's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_ts(text):
    # ISO 8601 UTC, with or without fractional seconds ("...49.360Z").
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def hours_since_launch(moment):
    return (moment - LAUNCH).total_seconds() / 3600.0


def log_time(hours, tau):
    return math.log(1.0 + max(hours, 0.0) / tau)


def refresh_events(case):
    """The reads where the page actually changed, oldest first.

    Consecutive reads with the same players-tracked and the same rates are one
    frozen page; only the first of each run is kept, plus the last read.
    """
    events = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if events and events[-1][0] == key:
            continue
        events.append((key, read))
    return [read for _, read in events]


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b x. Returns (a, b, resid_sd) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * e * e for w, e in zip(ws, resid)) / sw
    return (a, b, math.sqrt(var))


def median(xs):
    ordered = sorted(xs)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the scored levels")
    events = refresh_events(case)
    if not events:
        raise ValueError("no history")
    last = events[-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_ts(case["target"])
    t_target = hours_since_launch(target)
    t_last = hours_since_launch(parse_ts(last["at"]))

    # First pass: per-scale, per-trophy fits, for the tier-median slopes.
    # fits[tau][i] = {"n", "fit", "last_logit"}.
    tiers = [t["tier"] for t in case["trophies"]]
    fits = {}
    tier_slope = {}
    for tau in TAU_GRID:
        per = []
        for i, trophy in enumerate(case["trophies"]):
            xs, ys, ws = [], [], []
            for read in events:
                rate = read["rates"][i]
                if rate is None:
                    continue
                t = hours_since_launch(parse_ts(read["at"]))
                if t < t_last - WINDOW_H:
                    continue
                xs.append(log_time(t, tau))
                ys.append(logit(rate))
                ws.append(0.5 ** ((t_last - t) / HALF_LIFE_H))
            fit = weighted_line(xs, ys, ws) if len(xs) >= 2 else None
            last_rate = last["rates"][i]
            per.append({
                "n": len(xs),
                "fit": fit,
                "last_logit": logit(last_rate) if last_rate is not None else None,
            })
        fits[tau] = per
        by_tier = {}
        all_slopes = []
        for j, f in enumerate(per):
            if f["fit"] is not None:
                by_tier.setdefault(tiers[j], []).append(f["fit"][1])
                all_slopes.append(f["fit"][1])
        overall = median(all_slopes) if all_slopes else 0.0
        tier_slope[tau] = {}
        for tier, slopes in by_tier.items():
            tier_slope[tau][tier] = median(slopes) if len(slopes) >= 3 else overall
        tier_slope[tau]["__overall__"] = overall

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_logit = fits[TAU_GRID[0]][i]["last_logit"]
        if last_logit is None:
            # No rate anywhere: fall back to a middling rate with a wide band.
            rows.append({"id": trophy["id"], "med": logit(1.0), "sigma": SIGMA_CAP})
            continue
        meds, resids, ns = [], [], []
        for tau in TAU_GRID:
            f = fits[tau][i]
            x_last = log_time(t_last, tau)
            x_tgt = log_time(t_target, tau)
            if f["fit"] is None or f["n"] < 3:
                meds.append(last_logit)
                resids.append(RESID_CAP)
                ns.append(f["n"])
                continue
            a, b, resid = f["fit"]
            # Shrink the slope toward the tier median and toward zero.
            w_fit = float(f["n"])
            b_adj = (w_fit * b + W_TIER * tier_slope[tau].get(trophy["tier"], tier_slope[tau]["__overall__"])
                     + W_ZERO * 0.0) / (w_fit + W_TIER + W_ZERO)
            fitted_last = a + b * x_last
            move = b_adj * (x_tgt - x_last) * DAMP
            med = fitted_last + move
            med = (1.0 - W_FLAT) * med + W_FLAT * last_logit
            meds.append(med)
            resids.append(min(RESID_CAP, max(RESID_FLOOR, resid)))
            ns.append(f["n"])
        med_logit = mean(meds)
        sigma_fit = mean(resids) * math.sqrt(1.0 + horizon_days / 2.0)
        sigma_tau = stdev(meds)
        sigma = math.sqrt(SIGMA_RW ** 2 * horizon_days
                          + sigma_fit ** 2
                          + SIGMA_EXTRA ** 2 * horizon_days
                          + sigma_tau ** 2)
        if max(ns) < 3:
            sigma = max(sigma, SIGMA_FLOOR_FLAT)
        sigma = min(sigma, SIGMA_CAP) * SPREAD_SCALE
        rows.append({"id": trophy["id"], "med": med_logit, "sigma": sigma})

    # The platinum cannot sit above the lowest other median: it needs every
    # other trophy first.
    others_med = [r["med"] for r, tier in zip(rows, tiers) if tier != "platinum"]
    if others_med and any(tier == "platinum" for tier in tiers):
        cap = min(others_med)
        for r, tier in zip(rows, tiers):
            if tier == "platinum" and r["med"] > cap:
                r["med"] = cap

    out = []
    for r in rows:
        qs = [round4(clamp(inv_logit(r["med"] + z * r["sigma"]))) for z in Z]
        # Rounding can only move a quantile by a hair; keep them ordered.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out.append({"id": r["id"], "q": qs})
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
