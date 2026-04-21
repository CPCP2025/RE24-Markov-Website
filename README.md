# Baseball RE24 Simulator

A modular baseball simulation system that uses Statcast batted-ball data to power a Markov-chain inning simulator and produce RE24-style run-expectancy tables.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pre-cache Statcast data (requires internet; takes ~10–20 min per season)
python scripts/build_cache.py --seasons 2022 2023 2024

# 3. Launch the Streamlit UI
streamlit run app.py
```

---

## Project Structure

```
baseball_sim/
├── app.py                      # Streamlit UI entry point
├── requirements.txt
├── scripts/
│   └── build_cache.py          # Pre-download and cache Statcast data
├── data/                       # Auto-created; holds .parquet cache files
├── baseball_sim/
│   ├── __init__.py
│   ├── models.py               # Dataclasses: Bucket, PAEvent, PlayerProfile, BaseOutState
│   ├── ingestion.py            # pybaseball scraping + Parquet caching
│   ├── buckets.py              # LA/EV bucket grid + BucketModel (outcome probabilities)
│   ├── player_profile.py       # Build per-player bucket distributions
│   ├── fallback.py             # Youth/MiLB fallback probability model
│   ├── plate_appearance.py     # Single PA simulator
│   ├── markov.py               # Inning simulator + RE24 computation
│   └── validation.py           # Draft validation mode
└── tests/
    ├── test_buckets.py
    ├── test_player_profile.py
    ├── test_markov.py
    └── test_plate_appearance.py
```

---

## Running Tests

```bash
# From repo root
pytest tests/ -v
```

All tests are self-contained and use synthetic data — **no network access or pre-cached files required**.

---

## Architecture

### A. Bucketed Batted-Ball System (`buckets.py`)

- Launch angle: 10° bins from –90° to +90° (18 bins)
- Exit velocity: 5 mph bins from 40 to 125 mph (17 bins)
- **306 total buckets**
- Each bucket stores `P(outcome | bucket)` for all 9 outcome types
- Laplace smoothing (configurable `k`, default 1.0) prevents zero probabilities in sparse buckets
- `BucketModel.load_or_fit()` auto-caches to `data/bucket_model.pkl`

### B. Player Profiles (`player_profile.py`)

For each player:
1. Pull Statcast PA data for the configured lookback window
2. Compute K%, BB%, HBP%, SAC% from plate appearance events
3. Map batted balls to LA/EV buckets → bucket weight distribution

### C. Service-Time / Youth Fallback (`fallback.py`)

- Service time is **estimated** by counting seasons with ≥10 PA in Statcast
  (real MLB service time is not publicly available — this is a documented proxy)
- Players below the threshold (default: 1.0 season) use the fallback model
- Fallback rates (all configurable):
  - K: 26%, BB: 8%, HBP: 1%, SAC: 0.5%, Contact: 64.5%
  - Contact outcome distribution is conservative (reduced HR vs MLB average)
- Clearly isolated in `fallback.py` — swap MiLB data in by changing `get_fallback_profile()`

### D. Draft Validation Mode (`validation.py`)

```python
from baseball_sim.validation import run_validation
report = run_validation(df, bucket_model, n_sample=500)
```

Compares bucket model's top-1 prediction vs observed Statcast outcomes.
Also available as the **Validation Mode** tab in the UI.

### E. Markov Inning Simulator (`markov.py`)

- Standard 24-state (bases × outs) representation
- Simplified runner advancement rules (documented in `markov.py` docstring)
- `run_re24()` simulates N innings per state → expected runs + P(≥1 run)
- Reproducible via `seed` parameter

### F. Streamlit UI (`app.py`)

Three main tabs:
1. **Lineup Builder** — search players, add to slots, view profile badges
2. **Run Simulation** — configure and execute RE24 computation, download CSV
3. **Validation Mode** — compare predictions vs observed Statcast outcomes

---

## Key Assumptions

| Decision | Assumption |
|----------|-----------|
| Service time proxy | Count seasons with ≥10 PA in Statcast (real figure not public) |
| Non-contact PA ordering | Sequential Bernoulli: K → BB → HBP → SAC → contact |
| Single advancement | 1B → 3B; 2B/3B score (no first-to-third split modeled) |
| Double advancement | All runners score; batter to 2B |
| Fallback K rate | 26% (league avg ~22%; rookies trend higher) |
| Bucket midpoints | Used when sampling player-specific batted ball location |
| Laplace smoothing | k=1.0 default (1 pseudo-count per outcome cell) |

---

## Configuration Reference

All configurable via the Streamlit sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Bucket model seasons | 2022, 2023, 2024 | Seasons used to build outcome probabilities |
| Player lookback window | 2 seasons | How far back to pull player batting data |
| Service time threshold | 1.0 season | Minimum seasons before using MLB data |
| Simulations per state | 10,000 | RE24 accuracy increases with count |
| Random seed | 42 | For reproducible results |
| Laplace k | 1.0 | Smoothing strength |

---

## Limitations

- **Runner advancement** uses simplified rules; sac-fly scoring, tagging up, and first-to-third on singles are approximated
- **MiLB data** is not integrated; the fallback uses manually calibrated estimates
- **Scraping** requires network access to Baseball Savant; run `build_cache.py` first for offline use
- **Bucket sparsity**: very unusual LA/EV combinations (e.g. 130+ mph) have few observations; smoothing helps but predictions are uncertain
- **Lineup-specific RE24**: the tables reflect the specific lineup provided, not a generic MLB-average lineup
