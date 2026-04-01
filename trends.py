"""
trends.py — Google Trends data collector.

Key design decisions
---------------------
1. **Window stitching**: pytrends only returns daily data for windows ≤ 270 days.
   We request consecutive 90-day chunks (overlapping by TRENDS_OVERLAP_DAYS) and
   normalise each chunk against its overlap region so all values sit on the same
   0–100 scale.

2. **Caching**: Each (keyword, start, end) triple is cached as a Parquet file
   under cache/. If the file is younger than TRENDS_CACHE_TTL seconds we return
   it immediately — no network call.

3. **Rate-limit retry**: Google Trends returns HTTP 429 after a few rapid
   requests. We back off exponentially (up to TRENDS_RETRY_MAX attempts) before
   giving up and returning NaN rows, so the rest of the pipeline continues.

4. **Output schema**
   timestamp   : date (daily)
   brand       : brand name
   keyword     : the exact keyword sent to Trends
   hype_raw    : Trends interest value (0–100), NaN if unavailable
"""

import hashlib
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    BRANDS,
    CACHE_DIR,
    TRENDS_CACHE_TTL,
    TRENDS_GEO,
    TRENDS_OVERLAP_DAYS,
    TRENDS_RETRY_BACKOFF,
    TRENDS_RETRY_MAX,
    TRENDS_WINDOW_DAYS,
)
from logger import get_logger

log = get_logger(__name__)

try:
    from pytrends.request import TrendReq
    _HAS_PYTRENDS = True
except ImportError:
    _HAS_PYTRENDS = False
    log.warning("pytrends not installed. Run: pip install pytrends")


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_key(keyword: str, start: str, end: str) -> Path:
    """Deterministic filename from keyword + date range."""
    raw = f"{keyword}|{start}|{end}|{TRENDS_GEO}"
    h   = hashlib.md5(raw.encode()).hexdigest()[:12]
    return CACHE_DIR / f"trends_{h}.parquet"


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > TRENDS_CACHE_TTL:
        log.debug(f"Cache expired ({age/3600:.1f}h old): {path.name}")
        return None
    try:
        df = pd.read_parquet(path)
        log.debug(f"Cache hit: {path.name} ({len(df)} rows)")
        return df
    except Exception as e:
        log.warning(f"Cache read error ({path.name}): {e}")
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
        log.debug(f"Cached {len(df)} rows → {path.name}")
    except Exception as e:
        log.warning(f"Cache write error: {e}")


# ── Single window fetch ────────────────────────────────────────────────────────

def _fetch_window(
    pytrends: "TrendReq",
    keyword: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Fetch one (≤270 day) window from Google Trends.
    Returns a DataFrame with columns [date, keyword, hype_raw].
    On failure returns an empty DataFrame (caller handles NaN fill).
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")
    cache_path = _cache_key(keyword, start_str, end_str)

    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    timeframe = f"{start_str} {end_str}"
    log.info(f"  Trends fetch: '{keyword}' [{timeframe}]")

    for attempt in range(1, TRENDS_RETRY_MAX + 1):
        try:
            pytrends.build_payload(
                [keyword],
                timeframe=timeframe,
                geo=TRENDS_GEO,
                gprop="",
            )
            raw = pytrends.interest_over_time()
            if raw.empty:
                log.warning(f"  No data returned for '{keyword}' [{timeframe}]")
                return pd.DataFrame()

            raw = raw.drop(columns=["isPartial"], errors="ignore")
            raw = raw.reset_index().rename(columns={"date": "timestamp", keyword: "hype_raw"})
            raw["keyword"] = keyword
            raw = raw[["timestamp", "keyword", "hype_raw"]]
            raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.date

            _save_cache(raw, cache_path)
            return raw

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Too Many Requests" in err_str:
                wait = TRENDS_RETRY_BACKOFF * attempt
                log.warning(
                    f"  Rate limited (attempt {attempt}/{TRENDS_RETRY_MAX}). "
                    f"Waiting {wait}s…"
                )
                time.sleep(wait)
            else:
                log.error(f"  Trends error for '{keyword}': {e}")
                return pd.DataFrame()

    log.error(f"  Giving up on '{keyword}' [{timeframe}] after {TRENDS_RETRY_MAX} attempts")
    return pd.DataFrame()


# ── Window stitching ───────────────────────────────────────────────────────────

def _stitch_windows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate overlapping windows into a single continuous series.

    For each consecutive pair (A, B), we compute a scale factor from the
    overlapping date range so that B's values align with A's absolute scale.
    This preserves relative amplitude across the full date range.
    """
    if not frames:
        return pd.DataFrame(columns=["timestamp", "keyword", "hype_raw"])

    result = frames[0].copy()

    for nxt in frames[1:]:
        if nxt.empty:
            continue

        overlap_dates = set(result["timestamp"]) & set(nxt["timestamp"])

        if overlap_dates:
            prev_overlap = result[result["timestamp"].isin(overlap_dates)]["hype_raw"].mean()
            nxt_overlap  = nxt[nxt["timestamp"].isin(overlap_dates)]["hype_raw"].mean()

            if nxt_overlap and nxt_overlap > 0:
                scale = prev_overlap / nxt_overlap
                nxt   = nxt.copy()
                nxt["hype_raw"] = (nxt["hype_raw"] * scale).clip(0, 100)

        # Keep only non-overlapping rows from nxt
        new_rows = nxt[~nxt["timestamp"].isin(set(result["timestamp"]))]
        result   = pd.concat([result, new_rows], ignore_index=True)

    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


# ── Per-keyword full history fetch ─────────────────────────────────────────────

def _fetch_keyword_history(
    pytrends: "TrendReq",
    keyword: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Break date range into TRENDS_WINDOW_DAYS chunks and stitch."""
    frames  = []
    current = start_date

    while current < end_date:
        window_end = min(current + timedelta(days=TRENDS_WINDOW_DAYS - 1), end_date)

        frame = _fetch_window(pytrends, keyword, current, window_end)
        if not frame.empty:
            frames.append(frame)

        # BREAK condition to prevent infinite loop
        if window_end >= end_date:
            break

        # Advance by overlap to ensure continuous coverage
        current = window_end - timedelta(days=TRENDS_OVERLAP_DAYS) + timedelta(days=1)

        time.sleep(2.0)  # polite pause between windows

    return _stitch_windows(frames)


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_trends(
    start_date: Optional[date] = None,
    end_date:   Optional[date] = None,
) -> pd.DataFrame:
    """
    Fetch Google Trends for all brands defined in config.BRANDS.

    Parameters
    ----------
    start_date : first date to collect (default: 3 years ago)
    end_date   : last date  to collect (default: today)

    Returns
    -------
    DataFrame with columns:
        timestamp (date), brand, keyword, hype_raw (float, NaN if missing)
    """
    if not _HAS_PYTRENDS:
        log.error("pytrends is not installed — cannot fetch trends.")
        return pd.DataFrame(columns=["timestamp", "brand", "keyword", "hype_raw"])

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=3 * 365)

    log.info(
        f"fetch_trends: {start_date} → {end_date} "
        f"({(end_date - start_date).days} days, {len(BRANDS)} brands)"
    )

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=2, backoff_factor=1.0)
    all_frames: list[pd.DataFrame] = []

    for brand, cfg in BRANDS.items():
        log.info(f"Brand: {brand}")
        brand_frames = []

        for kw in cfg["trends_kw"]:
            log.info(f"  Keyword: '{kw}'")
            try:
                df = _fetch_keyword_history(pytrends, kw, start_date, end_date)
                if df.empty:
                    log.warning(f"  No data for '{kw}' — will be NaN")
                    continue
                df["brand"] = brand
                brand_frames.append(df)
            except Exception as e:
                log.error(f"  Unhandled error for '{kw}': {e}")
                continue

        if not brand_frames:
            log.warning(f"No trends data for brand '{brand}'")
            continue

        # Normalise keywords within this brand so they're on the same scale.
        # Strategy: for each keyword, divide by its own max then ×100.
        brand_df = pd.concat(brand_frames, ignore_index=True)

        brand_max = brand_df["hype_raw"].max()
        if brand_max and brand_max > 0:
            brand_df["hype_raw"] = brand_df["hype_raw"] / brand_max * 100

        all_frames.append(brand_df)

    if not all_frames:
        log.error("fetch_trends: no data collected at all.")
        return pd.DataFrame(columns=["timestamp", "brand", "keyword", "hype_raw"])

    result = pd.concat(all_frames, ignore_index=True)
    result["hype_raw"]  = pd.to_numeric(result["hype_raw"], errors="coerce")
    result = result.sort_values(["brand", "keyword", "timestamp"]).reset_index(drop=True)

    log.info(
        f"fetch_trends complete: {len(result):,} rows, "
        f"{result['brand'].nunique()} brands, "
        f"{result['keyword'].nunique()} keywords"
    )
    return result
