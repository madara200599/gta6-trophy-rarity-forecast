[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/madara200599/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/madara200599/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecasts, for every GTA VI PlayStation 5 trophy, its earn rate on Exophase
at 2026-11-26T00:00:00Z — one week after launch — as five quantiles
(5/25/50/75/95%), scored by pinball loss on the logit scale.

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
(`gta6-trophy-rarity-forecast-01`). The program is `main.py` (Python 3.13,
standard library only, no network, deterministic): it reads the case file
from `$MAKEFUN_INPUT_DIR` (or `/input`) and writes one forecast file per
case to `$MAKEFUN_OUTPUT_DIR` (or `/output`).

## How it works

Exophase does not refresh a game's counts continuously: the house saw the
trophy page frozen for hours on launch night, then one refresh moved every
rate at once (players tracked 638 → 3,868, trophies earned 943 → 4,276,
every rate but the one at 100% down). A trophy's displayed rate is a ratio
of two moving counts — earners over players tracked — sampled at Exophase's
own refresh times, so the model works in counts, not just in rates:

1. **Deduplicate the history into distinct snapshots.** Consecutive
   identical reads are one stale page, not new information.
2. **Trend model (when the history moves).** Per trophy, the logit of its
   rate is fitted against log-time since the release day began
   (`ln(1 + hours since 2026-11-19T00:00Z)`) over the last 48 hours of
   snapshots, by least squares. Slopes are shrunk toward the cross-trophy
   median slope — rare trophies have few earners and noisy slopes, while
   the launch's overall drift is shared. A second, counts-based forecast
   fits log earners and log players tracked separately against the same
   clock and takes their difference; the two forecasts are averaged in
   logit space. The spread comes from the fit residuals (shrunk toward the
   pooled residual variance), inflated by the extrapolation leverage of the
   target beyond the data, plus a floor for model and refresh error.
3. **Frozen-page fallback (when the history never moves — the practice
   cases).** The displayed rates are a stale snapshot and the next refresh
   dilutes them as new players join. The fallback forecasts the
   kit-documented launch-night refresh (players ~6x, earners ~1.34x, a
   logit shift of about −1.5), scaled by how long the page has been frozen
   toward the target, with a wide spread — except the trophy at 100%,
   which every newly tracked player earns at once and so does not move.

## Local verification

The real practice answers live server-side, so the practice losses below
are scored against proxy answers built from the facts the kit itself
documents about the refresh (players 638 → 3,868, trophies earned
943 → 4,276, the 05:00Z read unchanged again). Proxy A is the primary
one: the 100% trophy stays 100% and every other trophy's earners grow by
the factor the totals pin down (×1.3377), so each rate ×0.2207 — it
reproduces the documented totals exactly. The house publishes the starter
(persistence baseline) at about 1.51 loss on the practice; under proxy A
it scores 1.40, so the proxy is consistent with the real answer.

| proxy answer | this entry | starter (house baseline) |
|---|---|---|
| A — totals-exact dilution (×0.2207) | **0.182** | 1.402 |
| A2 — each trophy +2 earners | **0.187** | 1.375 |
| B — earners unchanged (×0.165) | **0.239** | 1.688 |
| C — new earns favour popular trophies | **0.189** | 1.423 |

Loss is the mean pinball loss over both practice cases (p00 and p01 share
the same answer page, so their losses coincide). Typical miss drops from
~0.68 points (starter) to ~0.01 points (this entry) under proxy A, and
90% coverage rises from 0.00 to 0.98.

The trend path (what the final actually runs) was also tested on a
synthetic final-like case — 20 snapshots over 60 hours, players tracked
638 → 20,124, each trophy on its own log-time trend, answer at the target
with a 3-hour refresh staleness — where this entry scores 0.181 against
the starter's 0.321, and 0.166 against 0.258 when the truth's trend bends
(saturates) halfway. Two runs per case are byte-identical.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).

Entered via referral code r7k9dyj1
