# Luxury Hype Pipeline

**Research project**: *Stochastic Modeling of Luxury Brand Hype through Limited-Edition Product Releases*

A production-grade data collection pipeline for building a real-world dataset to fit Poisson processes, shot-noise models, and Markov chains to luxury drop hype dynamics.

---

## Requirements

- **Python ≥ 3.10** (uses `match`-free type hints; `tuple[...]` syntax requires 3.9+)
- Tested on Python 3.12

```bash
pip install -r requirements.txt
```

---

## Project structure

```
hype_pipeline/
├── config.py          ← All tunable parameters (brands, keywords, paths, rate limits)
├── logger.py          ← Centralised logging (console + rotating file)
├── trends.py          ← Google Trends collector (window stitching, caching, retry)
├── drops.py           ← Drop event scraper (RSS + HTML, concurrency, dedup)
├── merger.py          ← Merge + save (CSV + SQLite)
├── pipeline.py        ← Orchestrator + CLI entry point
├── requirements.txt
├── tests/
│   └── test_pipeline.py  ← Unit tests (stdlib unittest, no extra install needed)
├── data/
│   ├── trends.csv
│   ├── drops.csv
│   ├── merged_dataset.csv   ← final research dataset
│   └── hype_pipeline.db     ← SQLite mirror of all three CSVs
├── cache/             ← Parquet cache for Google Trends windows (auto-managed)
└── logs/
    └── pipeline.log   ← Rotating log, 5 MB × 3 backups
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (trends + drops + merge)
python pipeline.py

# 3. Check output
head data/merged_dataset.csv
```

---

## CLI options

```bash
# Full run (default: last 3 years → today)
python pipeline.py

# Custom date range
python pipeline.py --start 2021-01-01 --end 2024-06-01

# Google Trends only
python pipeline.py --trends-only

# Drop events only
python pipeline.py --drops-only

# Collect but skip merge
python pipeline.py --no-merge

# Re-run merge from previously saved CSVs (no re-scraping)
python pipeline.py --from-cache
```

**Invalid flag combinations** — the CLI rejects these with exit code 2:

| Combination | Reason |
|---|---|
| `--trends-only --drops-only` | Mutually exclusive; neither stage would run |
| `--from-cache --no-merge` | `--from-cache` only exists to re-run the merge |
| `--start` ≥ `--end` | End date must be strictly after start date |
| Malformed date strings | Must be `YYYY-MM-DD` |

---

## Running the tests

No extra packages needed — the tests use Python's stdlib `unittest`:

```bash
python -m unittest tests/test_pipeline.py -v
```

If you have `pytest` installed:

```bash
pytest tests/ -v
```

**Test coverage:**

| Module | What is tested |
|---|---|
| `drops._parse_date` | RFC-2822, ISO with/without TZ, date-only, empty/garbage input; regression for the truncated-slice bug |
| `drops._matched_brands` | Single brand, multi-brand, case-insensitive, aliases (`Air Jordan` → Nike, `Hermes` → Hermès) |
| `drops._matched_drop_keyword` | Each keyword variant, first-match order, no-match, case-insensitive |
| `drops._normalise` | Non-drop filtering, brand expansion, NaN-brand preservation, deduplication, tz-naive output |
| `merger.merge_data` | Event flag set/unset, empty drops, NaN-brand row survival, same-day multi-drop concatenation, orphan brand, both-empty guard, output dtypes |
| `pipeline` CLI | All invalid flag combinations, valid flag combinations, date validation |

---

## Output schema

### `data/merged_dataset.csv` — final research dataset

| Column       | Type    | Source    | Description |
|--------------|---------|-----------|-------------|
| `timestamp`  | date    | scraped   | Daily date |
| `brand`      | str     | scraped   | Brand name (NaN if unmatched) |
| `keyword`    | str     | scraped   | Google Trends keyword |
| `hype_raw`   | float   | scraped   | Trends interest 0–100 (NaN if no data) |
| `event`      | int     | scraped   | 1 = drop occurred on this day, 0 = no drop |
| `event_name` | str     | scraped   | Article title(s) for drop events (NaN otherwise) |
| `source`     | str     | scraped   | News source name (NaN if no drop) |
| `url`        | str     | scraped   | Article URL (NaN if no drop) |

> **Missing values are NaN — never synthetic.** No values are fabricated or simulated.

### Columns to derive after collection

| Column              | How to derive |
|---------------------|---------------|
| `momentum`          | `df['hype_raw'].diff()` |
| `acceleration`      | `df['hype_raw'].diff().diff()` |
| `hype_smooth`       | `df['hype_raw'].rolling(7).mean()` |
| `log_hype`          | `np.log1p(df['hype_raw'])` |
| `hype_peak`         | `df.groupby(['brand','keyword'])['hype_raw'].transform('max')` |
| `hype_peak_day`     | Day index of `hype_peak` within each event window |
| `hype_decay_halflife` | Fit exponential decay to post-peak segment |
| `state`             | Bin `hype_raw` into LOW/RISING/PEAK/DECAYING/BASELINE |
| `prev_state`        | `df['state'].shift(1)` |
| `next_state`        | `df['state'].shift(-1)` |
| `rolling_volatility`| `df['hype_raw'].rolling(7).std()` |
| `days_since_last_drop` | Forward-fill from event dates |
| `competing_drops`   | Count of `event==1` on same day across all brands |

---

## What each source covers

| Source               | Categories          | Fields populated                               |
|----------------------|---------------------|------------------------------------------------|
| Google Trends        | All                 | `hype_raw`, `keyword`, `timestamp`             |
| Hypebeast RSS        | Sneaker, Apparel, Bag | `brand`, `event_name`, `drop_type`           |
| Highsnobiety RSS     | Apparel, Bag        | `brand`, `event_name`, `collab_flag`           |
| SneakerNews RSS      | Sneaker             | `brand`, `event_name`, `timestamp`             |
| Vogue RSS            | Bag, Apparel        | `brand`, `event_name`, `timestamp`             |

---

## Customising

**Add a new brand** — in `config.py`:
```python
"Moncler": {
    "trends_kw": ["Moncler collection", "Moncler drop"],
    "drop_kw":   ["Moncler"],
},
```

**Add a news source** — in `config.py`:
```python
"Grailed": {
    "rss":   ["https://www.grailed.com/drycleanonly/feed"],
    "pages": [],
    "max_p": 0,
},
```

**Tune rate limits** — in `config.py`:
```python
DELAY_MIN = 2.0
DELAY_MAX = 5.0
MAX_WORKERS = 2
TRENDS_RETRY_BACKOFF = 90
```

---

## Google Trends notes

- Daily data is only available for windows ≤ 270 days. The pipeline stitches consecutive 90-day windows and normalises them at overlap regions so values are on a consistent 0–100 scale.
- Every window is cached as a Parquet file under `cache/` (TTL: 24 h). Re-runs use the cache, avoiding repeated 429 rate-limit errors.
- If you hit rate limits, increase `TRENDS_RETRY_BACKOFF` in `config.py`.

---

## Downstream usage (stochastic modelling)

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/merged_dataset.csv", parse_dates=["timestamp"])

# ── Poisson: inter-arrival times between drops ──────────────────────────
events = df[df["event"] == 1].sort_values("timestamp")
events["inter_arrival"] = events.groupby("brand")["timestamp"].diff().dt.days
lambda_hat = 1 / events["inter_arrival"].dropna().mean()

# ── Hype decay: fit exponential after each peak ─────────────────────────
from scipy.optimize import curve_fit

def exp_decay(t, A, lam, c):
    return A * np.exp(-lam * t) + c

brand_df = df[(df["brand"] == "Supreme") & df["hype_raw"].notna()].copy()
brand_df["t"] = (brand_df["timestamp"] - brand_df["timestamp"].min()).dt.days
popt, _ = curve_fit(exp_decay, brand_df["t"], brand_df["hype_raw"], p0=[100, 0.01, 5])
halflife = np.log(2) / popt[1]

# ── Markov chain: discretise hype into states ───────────────────────────
df["state"] = pd.cut(
    df["hype_raw"],
    bins=[0, 20, 40, 60, 80, 100],
    labels=["BASELINE", "LOW", "RISING", "HIGH", "PEAK"],
)
df["prev_state"] = df.groupby(["brand", "keyword"])["state"].shift(1)
transitions = pd.crosstab(df["prev_state"], df["state"], normalize="index")
```

---

## Bug fixes applied (v1 → v2)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `trends.py` | `import numpy as np` unused | Removed |
| 2 | `pipeline.py` | `from pathlib import Path` unused | Removed |
| 3 | `drops.py` | Module-level `_SESSION` singleton shared across `ThreadPoolExecutor` threads | Per-call `requests.Session()` inside `with` block |
| 4 | `drops.py` | `datetime.strptime(raw[:len(fmt)], fmt)` sliced by format-string length (8) not rendered-date length (10) — every ISO parse silently failed | Removed the slice; call `strptime(raw, fmt)` directly |
| 5 | `drops.py` | `.dt.tz_localize(None)` on a tz-aware Series raises `TypeError` | Changed to `.dt.tz_convert(None)` |
| 6 | `merger.py` | `groupby(dropna=True)` silently discarded `NaN`-brand drop rows | Added `dropna=False` to groupby |
| 7 | `merger.py` | `groupby(dropna=False)` returns `float64` for the `brand` column when any key is `None`, causing a `ValueError` on merge | Cast `drops_agg["brand"]` to `object` after groupby |
| 8 | `merger.py` | `merge_data(empty, empty)` raised `KeyError: 'timestamp'` | Added an early-return guard for both-empty input |
| 9 | `merger.py` | `quality_report` would crash on NaN-only `hype_raw` with `:.1f` format | Added empty/NaN guard before formatting min/max |
| 10 | `pipeline.py` | `--trends-only --drops-only` together produced a silent no-op empty merge | Added explicit error + `sys.exit(2)` |
| 11 | `pipeline.py` | `--from-cache --no-merge` was accepted silently despite being meaningless | Added explicit error + `sys.exit(2)` |
| 12 | `pipeline.py` | Invalid date strings and `--start >= --end` not validated | Added `try/except ValueError` and range guard |
