"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (5/25/50/75/95%),
scored by pinball loss on the logit scale.

Model (my own design, standard library only, deterministic, no network):

Exophase does not refresh a game's counts continuously: the house saw the
page frozen for hours on launch night, then one refresh moved every rate at
once (players tracked 638 -> 3,868, trophies earned 943 -> 4,276, every rate
but the one at 100% down). So a trophy's displayed rate is a ratio of two
moving counts -- earners over players tracked -- sampled at Exophase's own
refresh times, and the answer is the same kind of snapshot at the target.

1. Deduplicate the history into distinct snapshots. Consecutive identical
   reads are one stale page, not new information; keeping them would
   overweight the frozen stretches.
2. When the history has real movement (>= 3 distinct snapshots), fit, per
   trophy, the logit of its rate against log-time since the release day
   began, x = ln(1 + hours since 2026-11-19T00:00Z), over the last 48 hours
   of snapshots (the recent regime), by least squares. Slopes are shrunk
   toward the cross-trophy median slope -- rare trophies have few earners
   and noisy slopes, while the launch's overall drift is shared. A second,
   counts-based forecast fits log earners and log players tracked separately
   against the same clock and takes their difference; the two forecasts are
   averaged in logit space, which keeps either fit from running away on its
   own. The spread comes from the fit residuals (shrunk toward the pooled
   residual variance), inflated by the extrapolation leverage of the target
   beyond the data, plus a floor for model and refresh error.
3. When the history is frozen (fewer than 3 distinct snapshots -- the
   practice cases, cut from launch night), there is no trend to fit. The
   page has been sitting unchanged while launch players played, so the next
   refresh dilutes the rates: the kit documents the launch-night refresh
   taking players tracked about 6x and earners about 1.34x, a logit shift of
   about -1.5 for every rate but the one at 100% (every new player earns
   that one at once, so it does not move). The fallback forecasts exactly
   that shift, scaled by how long the page has been frozen toward the
   target, with a wide spread, because the size of the pending refresh is
   the whole uncertainty.

Quantiles are the median logit plus/minus the normal z for each level times
the spread, mapped back through the logistic, rounded to 4 decimals.

Run: python3 main.py, with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
standing in for /input and /output outside the grader's container.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The release day began at 2026-11-19T00:00:00Z; the log-time clock runs from it.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# A frozen page's pending refresh: the kit documents the launch-night
# refresh moving players tracked ~6x and earners ~1.34x, i.e. a logit shift
# of about -1.5 on every rate but the one at 100%.
REFRESH_SHIFT = 1.5
REFRESH_RAMP_HOURS = 4.5
# Recent-regime window for the trend fits, in hours.
TREND_WINDOW_HOURS = 48.0
# Spread floor (logit std) for the trend model, and its cap.
SPREAD_FLOOR = 0.5
SPREAD_CAP = 1.2


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(moment):
    """Log-time clock: ln(1 + hours since the release day began)."""
    hours = (moment - LAUNCH).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def ols(xs, ys):
    """Least squares line y ~ a + b*x. Returns (slope, intercept, resid_var,
    n, x_mean, sxx); a flat x gives slope 0 and the mean as intercept."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my, 0.0, n, mx, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    var = rss / max(n - 2, 1) if n > 2 else 0.0
    return slope, intercept, var, n, mx, sxx


def distinct_snapshots(history):
    """Collapse consecutive identical reads into one snapshot each.

    Returns a list of (moment, players_tracked, rates tuple), oldest first.
    A read that repeats the previous page exactly carries no new signal:
    Exophase had not refreshed, so the counts are the same stale ones.
    """
    snaps = []
    last_key = None
    for read in history:
        rates = tuple(read["rates"])
        key = (read["players_tracked"], rates)
        if key == last_key:
            continue
        last_key = key
        snaps.append((parse_time(read["at"]), read["players_tracked"], rates))
    return snaps


def last_known_rate(snaps, index):
    """A trophy's rate in the latest snapshot that lists it, else None."""
    for _, _, rates in reversed(snaps):
        if rates[index] is not None:
            return rates[index]
    return None


def trend_forecast(case, snaps, x_target):
    """Forecast medians and spreads from the snapshots' movement.

    Per trophy: logit(rate) ~ log-time, slopes shrunk toward the pooled
    slope; and log(earners) - log(players) ~ log-time, the same way. The
    two medians are averaged; the spread is the shrunk residual variance
    times the extrapolation leverage, plus a model-error floor.
    """
    trophies = case["trophies"]
    checkpoint = parse_time(case["at"])
    window_start = checkpoint - timedelta(hours=TREND_WINDOW_HOURS)
    window = [s for s in snaps if s[0] >= window_start]
    if len(window) < 5:
        window = snaps
    xs = [log_time(moment) for moment, _, _ in window]

    # Players tracked: one shared series, fitted once.
    log_players = [math.log(max(players, 1)) for _, players, _ in window]
    b_players, a_players, _, _, _, _ = ols(xs, log_players)

    # Per-trophy fits on the logit scale and on the log-earner scale.
    per_trophy = []
    for i in range(len(trophies)):
        ys, ys_e, xsi = [], [], []
        for (moment, players, rates), x in zip(window, xs):
            rate = rates[i]
            if rate is None:
                continue
            ys.append(logit(rate))
            earners = rate * players / 100.0
            ys_e.append(math.log(earners + 0.5))
            xsi.append(x)
        if len(xsi) >= 2:
            b, a, var, n, mx, sxx = ols(xsi, ys)
            b_e, a_e, _, _, _, _ = ols(xsi, ys_e)
        else:
            b, a, var, n, mx, sxx = 0.0, (ys[0] if ys else logit(1.0)), 0.0, len(xsi), (xsi[0] if xsi else 0.0), 0.0
            b_e, a_e = 0.0, (ys_e[0] if ys_e else 0.0)
        per_trophy.append(
            {
                "n": n,
                "slope": b,
                "intercept": a,
                "var": var,
                "x_mean": mx,
                "sxx": sxx,
                "y_mean": sum(ys) / len(ys) if ys else logit(1.0),
                "slope_e": b_e,
                "intercept_e": a_e,
                "y_e_mean": sum(ys_e) / len(ys_e) if ys_e else 0.0,
                "x_e_mean": sum(xsi) / len(xsi) if xsi else 0.0,
            }
        )

    pool_slopes = [f["slope"] for f in per_trophy if f["n"] >= 3]
    pool_slope = median(pool_slopes) if pool_slopes else 0.0
    pool_slopes_e = [f["slope_e"] for f in per_trophy if f["n"] >= 3]
    pool_slope_e = median(pool_slopes_e) if pool_slopes_e else 0.0
    pool_vars = [f["var"] for f in per_trophy if f["n"] >= 3]
    pool_var = median(pool_vars) if pool_vars else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        f = per_trophy[i]
        n = f["n"]
        if n >= 2:
            # Shrink this trophy's slopes toward the pooled slopes; a
            # trophy with few earners has a noisy slope of its own.
            w = n / (n + 4.0)
            slope = w * f["slope"] + (1.0 - w) * pool_slope
            intercept = f["y_mean"] - slope * f["x_mean"]
            median_logit = intercept + slope * x_target
            slope_e = w * f["slope_e"] + (1.0 - w) * pool_slope_e
            intercept_e = f["y_e_mean"] - slope_e * f["x_e_mean"]
            log_earners = intercept_e + slope_e * x_target
            log_players_target = a_players + b_players * x_target
            rate_counts = 100.0 * math.exp(log_earners - log_players_target)
            median_counts = logit(rate_counts)
            median_all = 0.5 * median_logit + 0.5 * median_counts
            var_shrunk = (n * f["var"] + 4.0 * pool_var) / (n + 4.0)
            leverage = 1.0 + (x_target - f["x_mean"]) ** 2 / max(f["sxx"], 1e-9)
            leverage = min(leverage, 4.0)
            spread = math.sqrt(var_shrunk * leverage + SPREAD_FLOOR ** 2)
            spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
            if len(window) < 3:
                spread = max(spread, 0.9)
        elif n == 1:
            median_all = f["y_mean"]
            spread = 0.9
        else:
            median_all = logit(1.0)
            spread = 1.0
        rows.append((trophy["id"], median_all, spread))
    return rows


def frozen_forecast(case, snaps, target):
    """Forecast when the page never moved in the history.

    The displayed rates are a stale snapshot; the next refresh dilutes
    them as new players (who hold almost no trophies yet) join. The shift
    grows with how long the page has been frozen toward the target, up to
    the kit-documented launch-night refresh, and does not touch the trophy
    at 100%: every newly tracked player earns that one at once.
    """
    first_moment = snaps[0][0] if snaps else parse_time(case["at"])
    frozen_hours = (target - first_moment).total_seconds() / 3600.0
    shift = REFRESH_SHIFT * min(1.0, max(frozen_hours, 0.0) / REFRESH_RAMP_HOURS)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_known_rate(snaps, i)
        if rate is None:
            rate = 1.0
        # A trophy everyone holds stays held; the rest dilute.
        moved = shift * (1.0 - rate / 100.0)
        median_all = logit(rate) - moved
        spread = 0.1 + 0.8 * (1.0 - rate / 100.0)
        rows.append((trophy["id"], median_all, spread))
    return rows


def forecast_case(case, name):
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    target = parse_time(case["target"])
    x_target = log_time(target)
    snaps = distinct_snapshots(case["history"])
    if len(snaps) >= 3:
        medians = trend_forecast(case, snaps, x_target)
    else:
        medians = frozen_forecast(case, snaps, target)
    rows = []
    for trophy_id, median_all, spread in medians:
        qs = [round4(logistic(median_all + z * spread)) for z in Z]
        # Rounding can only equalise neighbours, never invert them, but
        # clamp defensively so the row is always valid and non-decreasing.
        qs = [min(100.0, max(0.01, q)) for q in qs]
        for j in range(1, len(qs)):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": trophy_id, "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
