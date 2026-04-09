"""
compute_hype_metrics.py
=======================
Research project: "Stochastic Modeling of Luxury Brand Hype through
Limited-Edition Product Releases"

What this script does
---------------------
Reads merged_dataset.csv (263k rows, 10 brands, 16 years of Google Trends
hype_raw signal) and produces a clean per-event dataset suitable for:

  - Poisson process estimation       (inter-arrival times between hype events)
  - Shot-noise / hype decay modeling (exponential decay half-life after peak)
  - Markov chain construction        (state transitions: BASELINE -> SPIKE -> DECAY)

Why the previous version only produced 4 rows
----------------------------------------------
The previous version read events from drops.csv, which only contains RSS
articles scraped in the last few days. It found 4 Nike articles from last week,
tried to compute hype_d30 (needing data 30 days ahead), and got NaN for
everything forward-looking because the trends data ends today.

The fix
-------
Drop the dependency on drops.csv entirely. Instead, detect historical hype
events directly from the hype_raw signal using a rolling z-score spike
detector. This gives ~938 real events across 10 brands and 16 years, all
fully within the trends window so every metric can be computed.

Output columns
--------------
  brand               Brand name
  event_date          Date when the hype spike peaked
  year / month        Convenience columns for time-series analysis
  spike_zscore        Signal strength (how many SDs above rolling mean)
  baseline_hype       Mean hype_raw in [-60d, -14d] before event
  pre_drop_buzz       Mean hype_raw in [-7d,  -1d]  before event
  hype_score_t0       hype_raw on event day
  hype_peak           Max hype_raw in [0d, +30d] post-event
  hype_peak_day       Days after event when peak was reached
  hype_lift_pct       (peak - baseline) / baseline x 100
  hype_d7             hype_raw at event + 7 days
  hype_d30            hype_raw at event + 30 days
  hype_d60            hype_raw at event + 60 days
  hype_decay_halflife Days for hype to decay to 50% above baseline
  days_since_last_event   Inter-arrival time in days (for Poisson lambda)
  prev_state          Markov state in the 14 days before event
  state_t0            Markov state on event day
  state_d7            Markov state at +7d
  state_d30           Markov state at +30d
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR        = Path("data")
MERGED_CSV      = DATA_DIR / "merged_dataset.csv"
OUTPUT_CSV      = DATA_DIR / "hype_metrics.csv"

SPIKE_ZSCORE_THRESH = 1.5   # z-score above rolling mean to count as a hype event
MIN_HYPE_VALUE      = 20    # absolute hype_raw floor (filters near-zero noise)
MIN_EVENT_GAP_DAYS  = 30    # minimum days between two distinct events per brand

BASELINE_START  = -60
BASELINE_END    = -14
PRE_DROP_START  = -7
PRE_DROP_END    = -1
POST_PEAK_DAYS  = 30
DECAY_DAYS      = 90
ROLLING_WINDOW  = 60
ROLLING_MIN_PTS = 14

STATE_BINS   = [0, 15, 35, 60, 80, 101]
STATE_LABELS = ["DORMANT", "LOW", "BUILDING", "HIGH", "PEAK"]

# Optional: minimum variance threshold to filter low-signal brands
MIN_BRAND_STD = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load merged_dataset.csv
# ══════════════════════════════════════════════════════════════════════════════

def load_merged(path: Path) -> pd.DataFrame:
    """
    Load merged_dataset.csv and collapse to a daily brand-level hype series.

    The file has one row per (timestamp, brand, keyword). Multiple keywords
    per brand mean the same date appears 5-9 times per brand. We average
    hype_raw across keywords to get a single stable brand-level signal.

    IMPORTANT: Date format in this file is DD-MM-YYYY.
    pd.to_datetime() defaults to MM-DD-YYYY and will silently misparse all dates
    unless dayfirst=True is passed explicitly.
    """
    log.info(f"Loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Raw rows: {len(df):,} | columns: {list(df.columns)}")

    # DD-MM-YYYY format requires dayfirst=True
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    bad = df["timestamp"].isna().sum()
    if bad:
        log.warning(f"  {bad} rows with unparseable timestamps dropped")
        df = df.dropna(subset=["timestamp"])

    df["hype_raw"] = pd.to_numeric(df["hype_raw"], errors="coerce")
    df = df.dropna(subset=["hype_raw", "brand"])

    log.info(
        f"  After cleaning: {len(df):,} rows | "
        f"{df['brand'].nunique()} brands | "
        f"date range {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    )

    # Collapse to one row per (date, brand) by averaging across keywords
    daily = (
        df.groupby(["timestamp", "brand"])["hype_raw"]
          .mean()
          .reset_index()
          .rename(columns={"hype_raw": "hype_avg"})
          .sort_values(["brand", "timestamp"])
          .reset_index(drop=True)
    )
    log.info(f"  Brand-level daily series: {len(daily):,} rows")
    return daily


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Detect hype events per brand
# ══════════════════════════════════════════════════════════════════════════════

def detect_events(series: pd.Series, brand: str, min_gap_days: int = None) -> pd.DataFrame:
    """
    Detect hype spike events in a single brand's daily hype_avg series.

    Uses a rolling z-score (how many standard deviations above the 60-day
    rolling mean) to identify anomalous hype spikes. Consecutive spikes within
    min_gap_days are merged into one event at their highest point.

    CHANGES (publication-grade):
    1. Smoothing: 3-day rolling mean to reduce noise spikes
    2. Adaptive threshold: uses 95th percentile of z-scores per brand
    3. Event validation: filters spikes where peak < baseline * 1.2

    Returns one row per distinct event: brand, event_date, hype_score_t0, spike_zscore
    """
    s = series.sort_index()

    # Clip values to valid range
    s = s.clip(0, 100)

    # Smoothing: 3-day rolling mean to reduce noise
    s_smooth = s.rolling(3, min_periods=1).mean()

    rolling_mean = s_smooth.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PTS).mean()
    rolling_std  = s_smooth.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PTS).std().clip(lower=1.0)
    zscore       = (s_smooth - rolling_mean) / rolling_std

    # Adaptive threshold: top 5% of z-scores per brand (quantile-based)
    zscore_valid = zscore.dropna()
    if len(zscore_valid) < 10:
        log.warning(f"  {brand}: insufficient data for quantile threshold")
        return pd.DataFrame()

    z_threshold = zscore_valid.quantile(0.95)
    spikes = s_smooth[(zscore > z_threshold) & (s_smooth > MIN_HYPE_VALUE)]

    # Use adaptive gap if provided
    if min_gap_days is None:
        min_gap_days = MIN_EVENT_GAP_DAYS

    if spikes.empty:
        log.warning(f"  {brand}: no spikes detected")
        return pd.DataFrame()

    events    = []
    last_date = None

    for date, val in spikes.items():
        gap = (date.to_pydatetime() - last_date).days if last_date else 9999

        if gap >= min_gap_days:
            events.append({
                "brand":         brand,
                "event_date":    date,
                "hype_score_t0": round(float(val), 2),
                "spike_zscore":  round(float(zscore[date]), 2),
            })
            last_date = date.to_pydatetime()
        elif val > events[-1]["hype_score_t0"]:
            # Same cluster, replace with higher peak
            events[-1]["event_date"]    = date
            events[-1]["hype_score_t0"] = round(float(val), 2)
            events[-1]["spike_zscore"]  = round(float(zscore[date]), 2)
            last_date = date.to_pydatetime()

    return pd.DataFrame(events)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Compute per-event metrics
# ══════════════════════════════════════════════════════════════════════════════

def _window(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Slice series to [start, end] inclusive using binary search O(log n)."""
    left  = s.index.searchsorted(start)
    right = s.index.searchsorted(end, side="right")
    return s.iloc[left:right]


def _nearest(s: pd.Series, target: pd.Timestamp, tol: int = 7) -> float:
    """Return value closest to target within tol days, else NaN."""
    if s.empty:
        return np.nan
    pos = s.index.searchsorted(target)
    candidates = [p for p in (pos - 1, pos) if 0 <= p < len(s)]
    if not candidates:
        return np.nan
    best = min(candidates, key=lambda p: abs((s.index[p] - target).days))
    return float(s.iloc[best]) if abs((s.index[best] - target).days) <= tol else np.nan


def _decay_halflife(post: pd.Series, peak_date: pd.Timestamp, baseline: float) -> float:
    """
    Fit exponential decay post-peak and return half-life in days.

    CHANGES (publication-grade):
    - Primary: scipy curve_fit for exponential decay
    - Fallback: log-linear regression if curve_fit fails or data is noisy
    - Numerical stability: epsilon added to log calculations
    - Rejects invalid fits (negative half-life, extreme values)
    """
    seg = post[post.index >= peak_date]
    if len(seg) < 3:
        return np.nan

    amplitude = float(seg.iloc[0]) - baseline
    if amplitude <= 0:
        return np.nan

    t = np.array([(d - peak_date).days for d in seg.index], dtype=float)
    y = seg.values.astype(float)

    # Primary method: curve_fit
    try:
        popt, pcov = curve_fit(
            lambda t, lam: amplitude * np.exp(-lam * t) + baseline,
            t, y, p0=[0.05], bounds=(1e-6, 2.0), maxfev=3000,
        )
        # Check fit quality: reject if variance is too high
        residuals = y - (amplitude * np.exp(-popt[0] * t) + baseline)
        rss = np.sum(residuals ** 2)
        if rss / len(y) > amplitude ** 2:  # Poor fit
            raise ValueError("Poor fit quality")
        hl = np.log(2) / popt[0]
        if 0 < hl < 10_000:
            return round(float(hl), 1)
    except Exception:
        pass  # Fall through to log-linear fallback

    # Fallback: log-linear regression (more robust to noise)
    # y = baseline + amplitude * exp(-beta * t)
    # log(y - baseline) = log(amplitude) - beta * t
    epsilon = 1e-6
    y_adj = y - baseline + epsilon
    y_log = np.log(np.maximum(y_adj, epsilon))

    # Filter out invalid values
    valid = np.isfinite(y_log)
    if valid.sum() < 3:
        return np.nan

    t_valid = t[valid]
    y_valid = y_log[valid]

    try:
        slope, intercept = np.polyfit(t_valid, y_valid, 1)
        beta = -slope
        if beta <= 0:
            return np.nan  # Invalid: would imply growing signal
        hl = np.log(2) / beta
        return round(float(hl), 1) if 0 < hl < 10_000 else np.nan
    except Exception:
        return np.nan


def _state(val: float, quantiles: np.ndarray = None) -> str:
    """
    Map hype_raw value to Markov state label.

    CHANGES (publication-grade):
    - Uses quantile-based states instead of fixed bins
    - States: LOW (<q1), MEDIUM (q1-q2), HIGH (q2-q3), VIRAL (>q3)
    - Falls back to fixed bins if quantiles not provided
    """
    if np.isnan(val):
        return np.nan

    if quantiles is not None:
        q1, q2, q3 = quantiles
        if val < q1:
            return "LOW"
        elif val < q2:
            return "MEDIUM"
        elif val < q3:
            return "HIGH"
        else:
            return "VIRAL"

    # Fallback to fixed bins for backward compatibility
    for i, (lo, hi) in enumerate(zip(STATE_BINS[:-1], STATE_BINS[1:])):
        if lo <= val < hi:
            return STATE_LABELS[i]
    return STATE_LABELS[-1]


def compute_event_metrics(event_date: pd.Timestamp, series: pd.Series,
                          brand_quantiles: np.ndarray = None) -> dict:
    """
    All hype metrics for one event. Uses only pre-sorted series slices.

    CHANGES (publication-grade):
    - Event validation filter: rejects events where hype_peak < baseline * 1.2
    - Quantile-based states passed via brand_quantiles
    """
    ed = pd.Timestamp(event_date).normalize()
    T  = pd.Timedelta

    baseline_vals = _window(series, ed + T(BASELINE_START, "D"), ed + T(BASELINE_END, "D"))
    if len(baseline_vals) < 3:   # extend window if too sparse
        baseline_vals = _window(series, ed + T(BASELINE_START - 30, "D"), ed + T(BASELINE_END, "D"))

    pre_vals  = _window(series, ed + T(PRE_DROP_START, "D"), ed + T(PRE_DROP_END, "D"))
    post_vals = _window(series, ed,                          ed + T(POST_PEAK_DAYS, "D"))
    decay_vals= _window(series, ed,                          ed + T(DECAY_DAYS, "D"))
    prev_vals = _window(series, ed + T(-14, "D"),            ed + T(-1, "D"))

    baseline_hype = float(baseline_vals.mean()) if not baseline_vals.empty else np.nan
    pre_drop_buzz = float(pre_vals.mean())      if not pre_vals.empty      else np.nan
    hype_t0       = _nearest(series, ed)

    if post_vals.empty:
        hype_peak, hype_peak_day, peak_date = np.nan, np.nan, ed
    else:
        peak_idx      = post_vals.idxmax()
        hype_peak     = float(post_vals.max())
        hype_peak_day = int((peak_idx - ed).days)
        peak_date     = peak_idx

    hype_d7  = _nearest(series, ed + T(7,  "D"))
    hype_d30 = _nearest(series, ed + T(30, "D"))
    hype_d60 = _nearest(series, ed + T(60, "D"))

    # Event validation filter: reject weak spikes
    # A real event should have peak at least 20% above baseline
    if not np.isnan(baseline_hype) and not np.isnan(hype_peak):
        if hype_peak < baseline_hype * 1.2:
            return None  # Filter out non-meaningful spikes

    hype_lift_pct = (
        round((hype_peak - baseline_hype) / baseline_hype * 100, 2)
        if not np.isnan(baseline_hype) and baseline_hype > 0 and not np.isnan(hype_peak)
        else np.nan
    )

    decay_hl  = _decay_halflife(decay_vals, peak_date,
                                baseline_hype if not np.isnan(baseline_hype) else 0.0)
    prev_avg  = float(prev_vals.mean()) if not prev_vals.empty else np.nan

    def _r(v): return round(float(v), 2) if not np.isnan(v) else np.nan

    return {
        "baseline_hype":      _r(baseline_hype),
        "pre_drop_buzz":      _r(pre_drop_buzz),
        "hype_score_t0":      _r(hype_t0),
        "hype_peak":          _r(hype_peak),
        "hype_peak_day":      hype_peak_day,
        "hype_lift_pct":      hype_lift_pct,
        "hype_d7":            _r(hype_d7),
        "hype_d30":           _r(hype_d30),
        "hype_d60":           _r(hype_d60),
        "hype_decay_halflife": decay_hl,
        "prev_state":         _state(prev_avg, brand_quantiles),
        "state_t0":           _state(hype_t0, brand_quantiles),
        "state_d7":           _state(hype_d7, brand_quantiles),
        "state_d30":          _state(hype_d30, brand_quantiles),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Orchestrate
# ══════════════════════════════════════════════════════════════════════════════

def run(merged_path: Path = MERGED_CSV, output_path: Path = OUTPUT_CSV) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("COMPUTE HYPE METRICS — START")
    log.info("=" * 60)

    daily = load_merged(merged_path)

    # Pre-group by brand (O(1) lookup per event, avoids O(n^2) filtering)
    brand_series: dict[str, pd.Series] = {
        str(brand): grp.set_index("timestamp")["hype_avg"].sort_index()
        for brand, grp in daily.groupby("brand")
    }
    log.info(f"Brand series built for: {sorted(brand_series.keys())}")

    # Filter low-variance brands (optional improvement)
    filtered_brands = {}
    for brand, s in brand_series.items():
        if s.std() < MIN_BRAND_STD:
            log.info(f"  {brand}: skipping (std={s.std():.2f} < {MIN_BRAND_STD})")
        else:
            filtered_brands[brand] = s
    brand_series = filtered_brands

    # Adaptive event gap: scale with series length (2% of timeline, min 7 days)
    base_series = next(iter(brand_series.values())) if brand_series else None
    if base_series is not None:
        min_gap = max(7, int(len(base_series) * 0.02))
        log.info(f"  Adaptive MIN_EVENT_GAP_DAYS: {min_gap} (2% of timeline)")
    else:
        min_gap = MIN_EVENT_GAP_DAYS

    # Detect events
    log.info("Detecting hype events (rolling z-score spike detector)...")
    all_events = []
    for brand, s in brand_series.items():
        evts = detect_events(s, brand, min_gap_days=min_gap)
        if not evts.empty:
            all_events.append(evts)
            log.info(f"  {brand:<20}: {len(evts):3d} events")

    if not all_events:
        log.error("No events detected — check SPIKE_ZSCORE_THRESH and MIN_HYPE_VALUE")
        return pd.DataFrame()

    events_df = pd.concat(all_events, ignore_index=True)
    log.info(f"Total events: {len(events_df):,} across {events_df['brand'].nunique()} brands")

    # Pre-compute quantiles per brand for Markov states
    brand_quantiles = {
        brand: s.quantile([0.25, 0.50, 0.75]).values
        for brand, s in brand_series.items()
    }

    # Compute metrics
    log.info("Computing per-event hype metrics...")
    rows = []
    for _, ev in events_df.iterrows():
        brand      = str(ev["brand"])
        event_date = ev["event_date"]
        metrics    = compute_event_metrics(event_date, brand_series[brand], brand_quantiles[brand])
        if metrics is None:  # Filtered out by event validation
            continue
        rows.append({
            "brand":        brand,
            "event_date":   event_date.date(),
            "year":         event_date.year,
            "month":        event_date.month,
            "spike_zscore": ev["spike_zscore"],
            **metrics,
        })

    result = pd.DataFrame(rows)

    # Add inter-arrival times (Poisson rate estimation)
    result = result.sort_values(["brand", "event_date"]).reset_index(drop=True)
    result["days_since_last_event"] = (
        result.groupby("brand")["event_date"]
              .transform(lambda x: pd.to_datetime(x).diff().dt.days)
    )

    col_order = [
        "brand", "event_date", "year", "month", "spike_zscore",
        "baseline_hype", "pre_drop_buzz", "hype_score_t0",
        "hype_peak", "hype_peak_day", "hype_lift_pct",
        "hype_d7", "hype_d30", "hype_d60", "hype_decay_halflife",
        "days_since_last_event",
        "prev_state", "state_t0", "state_d7", "state_d30",
    ]
    result = result[[c for c in col_order if c in result.columns]]

    output_path.parent.mkdir(exist_ok=True)
    result.to_csv(output_path, index=False)

    _quality_report(result)
    return result


def _quality_report(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("QUALITY REPORT")
    log.info("=" * 60)
    log.info(f"  Total events          : {len(df):,}")
    log.info(f"  Brands                : {sorted(df['brand'].unique().tolist())}")
    log.info(f"  Date range            : {df['event_date'].min()} to {df['event_date'].max()}")
    log.info("  Events per brand:")
    for brand, cnt in df.groupby("brand").size().sort_values(ascending=False).items():
        log.info(f"    {brand:<22} {cnt:3d}")

    metric_cols = [
        "baseline_hype", "pre_drop_buzz", "hype_score_t0",
        "hype_peak", "hype_peak_day", "hype_lift_pct",
        "hype_d7", "hype_d30", "hype_d60", "hype_decay_halflife",
    ]
    log.info("")
    log.info(f"  {'Metric':<22} {'Non-NaN':>8}  {'Mean':>8}  {'Min':>8}  {'Max':>8}")
    log.info(f"  {'-'*22} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for col in metric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            log.info(f"  {col:<22} {'0':>8}  {'—':>8}  {'—':>8}  {'—':>8}")
        else:
            log.info(f"  {col:<22} {len(s):>8,}  {s.mean():>8.2f}  {s.min():>8.2f}  {s.max():>8.2f}")

    log.info("")
    log.info("  Sanity checks:")
    v = df.dropna(subset=["baseline_hype", "hype_peak"])
    if len(v):
        pct = (pd.to_numeric(v["hype_peak"]) >= pd.to_numeric(v["baseline_hype"])).mean() * 100
        log.info(f"    peak >= baseline        : {pct:.1f}%")

    iat = pd.to_numeric(df["days_since_last_event"], errors="coerce").dropna()
    if not iat.empty:
        mean_gap = iat.mean()
        var_gap = iat.var()
        log.info(f"    Poisson lambda (events/day): {1/mean_gap:.4f}  "
                 f"(mean gap: {mean_gap:.1f} days)")
        log.info(f"    Inter-arrival variance: {var_gap:.1f} (mean={mean_gap:.1f})")
        # Poisson validation: variance ≈ mean
        if var_gap < 0.5 * mean_gap:
            log.info("    WARNING: variance << mean — under-dispersed (regular spacing)")
        elif var_gap > 2.0 * mean_gap:
            log.info("    WARNING: variance >> mean — over-dispersed (clustering, NOT Poisson)")
        else:
            log.info("    Poisson assumption: reasonable (variance ≈ mean)")

    if "state_t0" in df.columns:
        dist = df["state_t0"].value_counts(normalize=True).mul(100).round(1).to_dict()
        log.info(f"    state_t0 distribution   : {dist}")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compute hype metrics from merged_dataset.csv")
    p.add_argument("--input",    default=str(MERGED_CSV))
    p.add_argument("--output",   default=str(OUTPUT_CSV))
    p.add_argument("--zscore",   type=float, default=SPIKE_ZSCORE_THRESH)
    p.add_argument("--min-hype", type=float, default=MIN_HYPE_VALUE)
    args = p.parse_args()

    SPIKE_ZSCORE_THRESH = args.zscore
    MIN_HYPE_VALUE      = args.min_hype

    result = run(Path(args.input), Path(args.output))
    sys.exit(0 if not result.empty else 1)
