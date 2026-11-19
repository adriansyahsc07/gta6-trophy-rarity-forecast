# gta6-trophy-rarity-forecast
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adriansyahsc07/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/adriansyahsc07/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's PlayStation 5 trophy earn rates on Exophase. Exophase republishes a game's counts only every few hours, so the recorded page is a staircase, not a curve: `main.py` first collapses each case's history to the reads where something actually changed (refresh points), then reads every trophy through two lenses and blends them on the logit scale — a trend lens (least-squares line of logit(rate) on log time since launch, slopes shrunk toward the trophy's tier) and a cohort lens (earners and tracked players projected separately, so a wave of newly tracked players dilutes rates the way the launch-night refresh did). When the page has never moved in the window, a stale-page prior stands in for the cohort lens: a refresh is expected before the target, the pool grows, and the incoming cohort holds each trophy at a power-compressed rate. The spread is a widened random walk on the logit scale, with extra width for rare trophies. Python 3 standard library only, deterministic, offline.

## Run it
```sh
python3 main.py
```
---
<sub>Entered on [make.fun](https://make.fun/?ref=kj9js265&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
