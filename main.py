#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates (percent of tracked players) at a
case's target time, as five quantiles per trophy.

Model, in one paragraph: Exophase only republishes a game's counts every
few hours, so the recorded page is a staircase, not a curve. I collapse
each case's history to the moments something actually changed (refresh
points), then look at every trophy through two lenses and blend them on
the logit scale: a trend lens (a least-squares line of logit(rate) on
log time since launch, slopes shrunk toward the trophy's tier) and a
cohort lens (earners and tracked players projected separately, so a wave
of newly tracked players dilutes rates exactly the way the launch-night
refresh diluted them). Where the page has never moved in the window, a
stale-page prior stands in for the cohort lens: one more refresh is
expected before the target, the pool grows, and the incoming cohort
holds each trophy at a compressed rate. The spread is a random walk on
the logit scale, widened by fit residuals and extra width for rare
trophies.

Standard library only, deterministic, offline.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist

# ---------------------------------------------------------------------------
# Constants (design choices, fixed in advance of any grading)
# ---------------------------------------------------------------------------

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
RATE_LO, RATE_HI = 0.01, 99.99          # the scorer's clamp, in percent

RW_COEF = 0.5                           # random-walk sigma per sqrt(day)
SPREAD_K = 1.4                          # multiplier on the random walk
SIGMA_FLOOR = 0.35                      # irreducible logit spread
SIGMA_MAX = 3.0
SHRINK_PRIOR_N = 4.0                    # slope shrinkage pseudo-count
TREND_CAP = 3.5                         # max logit move the trend may claim

# Stale-page prior (used when the history holds a single refresh point).
# The sizes follow the one refresh the kit documents: the page sat at 638
# tracked players from before launch until a refresh multiplied the pool
# sixfold, so the pool gain is large and a refresh is expected within a
# few hours of staleness.
PRIOR_POOL_GAIN = 7.0                   # pool multiplier at full staleness
PRIOR_POOL_TAU_H = 5.0                  # hours for that growth to build up
PRIOR_REFRESH_TAU_H = 3.5               # hours scale for "a refresh lands"
PRIOR_REFRESH_CAP = 0.90                # never fully certain of a refresh
PRIOR_UPLIFT = 0.35                     # old cohort keeps earning meanwhile
PRIOR_UPLIFT_TAU_H = 12.0
NEW_COHORT_GAMMA = 1.25                 # incoming cohort rate compression

TIER_PRIOR_RATE = {                     # percent, for never-observed trophies
    "platinum": 0.05,
    "gold": 0.30,
    "silver": 0.80,
    "bronze": 1.50,
}
PLATINUM_Q50_CAP_MULT = 3.0             # platinum stays rare absent evidence

Z_OF_LEVEL = {}


def z_for(level):
    if level not in Z_OF_LEVEL:
        Z_OF_LEVEL[level] = NormalDist().inv_cdf(level)
    return Z_OF_LEVEL[level]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def clamp_rate(r):
    return min(RATE_HI, max(RATE_LO, r))


def logit_pct(rate):
    """Logit of a percentage, with the scorer's clamp applied first."""
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x >= 0:
        p = 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        p = e / (1.0 + e)
    return 100.0 * p


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def log_time(hours):
    return math.log1p(max(0.0, hours))


def ols(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b, rms residual) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    rms = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rms


# ---------------------------------------------------------------------------
# History -> refresh points
# ---------------------------------------------------------------------------

def refresh_points(history, n_trophies):
    """Collapse the read history to the reads where anything changed.

    A read joins the point list when the tracked-player count, the total
    trophies earned, or any trophy's rate differs from the previous read.
    Rates are carried forward across nulls first, so a trophy missing
    from one read keeps its last seen value instead of punching a hole
    in the series. Each point: (hours since release, players, [rates]).
    """
    points = []
    carried = [None] * n_trophies
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        for i in range(n_trophies):
            if i < len(rates) and rates[i] is not None:
                carried[i] = float(rates[i])
        players = read.get("players_tracked")
        key = (players, read.get("trophies_earned"), tuple(carried))
        if prev_key is None or key != prev_key:
            moment = parse_time(read["at"])
            points.append((hours_since_release(moment), players, list(carried)))
            prev_key = key
    return points


# ---------------------------------------------------------------------------
# The two lenses
# ---------------------------------------------------------------------------

def trend_view(points, idx, x_target):
    """OLS of logit(rate) on log-time over the refresh points.

    Returns (slope, predict_fn, rms) or None when the series cannot
    support a line (fewer than 3 points, or a single distinct x).
    """
    xs, ys = [], []
    for hours, _players, rates in points:
        r = rates[idx]
        if r is None:
            continue
        xs.append(log_time(hours))
        ys.append(logit_pct(r))
    if len(xs) < 3 or len(set(xs)) < 2:
        return None
    fit = ols(xs, ys)
    if fit is None:
        return None
    a, b, rms = fit
    return {"slope": b, "a": a, "xs": xs, "ys": ys, "rms": max(rms, 0.05)}


def pool_projection(points, x_target):
    """Project the tracked-player count to the target from refresh points."""
    xs = [log_time(h) for h, p, _r in points if p]
    ys = [math.log(p) for _h, p, _r in points if p]
    if len(xs) >= 2 and len(set(xs)) >= 2:
        fit = ols(xs, ys)
        if fit is not None:
            a, b, _rms = fit
            b = min(1.5, max(-0.02, b))
            last_p = [p for _h, p, _r in points if p][-1]
            proj = math.exp(a + b * x_target)
            return min(max(proj, 0.5 * last_p), 40.0 * last_p)
    return None


def dilution_view_fitted(points, idx, x_target, pool_at_target):
    """Project earners and players separately; the rate is their ratio."""
    if pool_at_target is None:
        return None
    xs, ys = [], []
    for hours, players, rates in points:
        r = rates[idx]
        if r is None or not players:
            continue
        earners = r / 100.0 * players
        xs.append(log_time(hours))
        ys.append(math.log(earners + 0.5))
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    fit = ols(xs, ys)
    if fit is None:
        return None
    a, b, _rms = fit
    earners_at_target = max(0.0, math.exp(a + b * x_target) - 0.5)
    return clamp_rate(100.0 * earners_at_target / pool_at_target)


def dilution_view_prior(points, idx, target_hours):
    """Cohort projection when the page has never moved in the window.

    The snapshot is stale: it predates at least one refresh the target
    will likely see. Simulate that refresh: the tracked pool grows by a
    staleness-driven factor, the existing cohort keeps earning at a
    modest uplift, and the incoming cohort holds this trophy at a
    power-compressed rate (rare trophies compress hardest, a 100% trophy
    stays 100%). Blend the refreshed page with the no-refresh page by
    the probability that a refresh lands before the target.
    Returns (rate, scenario_logit_gap) or None.
    """
    if len(points) != 1:
        return None
    hours0, players, rates = points[0]
    last = rates[idx]
    if last is None or not players:
        return None
    staleness_h = max(0.0, target_hours - hours0)
    growth = 1.0 + PRIOR_POOL_GAIN * (1.0 - math.exp(-staleness_h / PRIOR_POOL_TAU_H))
    pool_new = players * growth
    uplift = PRIOR_UPLIFT * (1.0 - math.exp(-staleness_h / PRIOR_UPLIFT_TAU_H))
    earners_old = last / 100.0 * players * (1.0 + uplift * (1.0 - last / 100.0))
    incoming_rate = 100.0 * (clamp_rate(last) / 100.0) ** NEW_COHORT_GAMMA
    earners_new = (pool_new - players) * incoming_rate / 100.0
    refreshed = clamp_rate(100.0 * (earners_old + earners_new) / pool_new)
    p_refresh = min(PRIOR_REFRESH_CAP,
                    1.0 - math.exp(-staleness_h / PRIOR_REFRESH_TAU_H))
    blended = inv_logit_pct(
        p_refresh * logit_pct(refreshed) + (1.0 - p_refresh) * logit_pct(last))
    gap = abs(logit_pct(refreshed) - logit_pct(last))
    return blended, gap


# ---------------------------------------------------------------------------
# Per-trophy forecast
# ---------------------------------------------------------------------------

def forecast_trophy(trophy, idx, points, target_hours, d_days,
                    tier_mean_slope):
    """Return (q50 in percent, sigma on the logit scale)."""
    tier = trophy.get("tier", "bronze")
    x_target = log_time(target_hours)
    series = [rates[idx] for _h, _p, rates in points]
    observed = [r for r in series if r is not None]
    last = observed[-1] if observed else None

    if not points or last is None:
        prior = TIER_PRIOR_RATE.get(tier, 1.0)
        return prior, 1.20

    trend = trend_view(points, idx, x_target)
    pool_at_target = pool_projection(points, x_target)

    trend_logit = None
    resid = None
    if trend is not None:
        n = len(trend["xs"])
        mean = tier_mean_slope.get(tier)
        if mean is None:
            mean = trend["slope"]
        slope = (n * trend["slope"] + SHRINK_PRIOR_N * mean) / (n + SHRINK_PRIOR_N)
        mx = sum(trend["xs"]) / n
        my = sum(trend["ys"]) / n
        pred = my + slope * (x_target - mx)
        last_logit = logit_pct(last)
        pred = min(last_logit + TREND_CAP, max(last_logit - TREND_CAP, pred))
        trend_logit = min(logit_pct(RATE_HI), max(logit_pct(RATE_LO), pred))
        resid = trend["rms"]

    dil_rate = None
    scenario_gap = 0.0
    if len(points) >= 2:
        dil_rate = dilution_view_fitted(points, idx, x_target, pool_at_target)
    else:
        prior_view = dilution_view_prior(points, idx, target_hours)
        if prior_view is not None:
            dil_rate, scenario_gap = prior_view

    # --- blend the lenses on the logit scale -------------------------------
    last_logit = logit_pct(last)
    if trend_logit is not None and dil_rate is not None:
        dil_logit = logit_pct(dil_rate)
        # Lean on the cohort lens right after a big player jump, and more
        # when the two lenses disagree: dilution is the mechanism that
        # moves rates between refreshes.
        jump = 1.0
        if len(points) >= 2 and points[-2][1] and points[-1][1]:
            jump = points[-1][1] / points[-2][1]
        w_dil = 0.65 if jump >= 1.5 else 0.50
        if abs(dil_logit - trend_logit) > 1.0:
            w_dil = max(w_dil, 0.70)
        q50_logit = w_dil * dil_logit + (1.0 - w_dil) * trend_logit
    elif dil_rate is not None:
        dil_logit = logit_pct(dil_rate)
        if len(points) >= 3:
            q50_logit = dil_logit
        else:
            # Two refresh points, or a stale-page prior: stay anchored
            # halfway to the last observed rate.
            q50_logit = 0.5 * dil_logit + 0.5 * last_logit
    elif trend_logit is not None:
        q50_logit = trend_logit
    else:
        q50_logit = last_logit

    q50 = inv_logit_pct(q50_logit)
    if tier == "platinum":
        q50 = min(q50, max(last * PLATINUM_Q50_CAP_MULT, last))

    # --- spread -------------------------------------------------------------
    if resid is None:
        resid = 0.45 + 0.5 * scenario_gap if scenario_gap else 0.50
    sigma = math.sqrt((RW_COEF * math.sqrt(d_days)) ** 2 * SPREAD_K ** 2
                      + resid ** 2 + SIGMA_FLOOR ** 2)
    sigma *= 1.0 + 0.5 * math.exp(-q50 / 2.0)   # rare trophies swing more
    sigma = min(SIGMA_MAX, max(0.15, sigma))
    return q50, sigma


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    points = refresh_points(history, len(trophies))
    target_hours = hours_since_release(parse_time(case["target"]))
    d_days = max(0.0, case["horizon_minutes"] / 1440.0)

    # Tier mean slopes first, so individual slopes can shrink toward them.
    raw = {}
    for idx, trophy in enumerate(trophies):
        tv = trend_view(points, idx, log_time(target_hours))
        if tv is not None:
            raw.setdefault(trophy.get("tier", "bronze"), []).append(tv["slope"])
    tier_mean_slope = {t: sum(v) / len(v) for t, v in raw.items()}

    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = []
    for idx, trophy in enumerate(trophies):
        q50, sigma = forecast_trophy(trophy, idx, points, target_hours,
                                     d_days, tier_mean_slope)
        centre = logit_pct(q50)
        qs = []
        for level in levels:
            q = inv_logit_pct(centre + z_for(level) * sigma)
            qs.append(round(min(100.0, max(0.0, q)), 4))
        for i in range(1, len(qs)):            # force non-decreasing
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = "{}.out".format(case.get("id") or name[:-3])
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
