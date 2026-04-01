"""
merger.py — Merges trends + drops into the final research dataset.

Merge logic
-----------
The two datasets share (timestamp, brand) as the natural join key, but they
live at different granularities:

  trends.csv  : one row per (day, brand, keyword)  — dense time-series
  drops.csv   : one row per (drop_event, brand)     — sparse events

Strategy
~~~~~~~~
1. Normalise both to daily granularity (date only, no time component).
2. Left-join trends ← drops on (date, brand) so every Trends row exists
   in the output.  Drop events that match a (date, brand) in Trends get
   event = 1, event_name filled; all other Trends rows get event = 0.
3. For brands that appear in drops but have no Trends data at all, we
   emit the drop rows alone (event = 1, hype_raw = NaN).
4. Sort by (brand, keyword, timestamp).

Output schema (all real, no synthetic values)
---------------------------------------------
  timestamp         : date
  brand             : str
  keyword           : str (NaN for pure-event rows)
  hype_raw          : float (0–100, NaN if no Trends data)
  event             : int  (0 or 1)
  event_name        : str  (NaN if no drop on that day)
  source            : str  (NaN if no drop on that day)
  url               : str  (NaN if no drop on that day)

Storage
-------
  • CSV  : data/trends.csv, data/drops.csv, data/merged_dataset.csv
  • SQLite: data/hype_pipeline.db  (tables: trends, drops, merged)
    — SQLite lets downstream analysis query without loading the full CSV.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from config import DB_PATH, DROPS_CSV, MERGED_CSV, TRENDS_CSV
from logger import get_logger

log = get_logger(__name__)


# ── Save helpers ───────────────────────────────────────────────────────────────

def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    try:
        df.to_csv(path, index=False)
        log.info(f"Saved {label}: {path} ({len(df):,} rows)")
    except Exception as e:
        log.error(f"CSV save failed ({label}): {e}")


def _save_sqlite(df: pd.DataFrame, table: str, db_path: Path = DB_PATH) -> None:
    try:
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table, conn, if_exists="replace", index=False)
        log.info(f"SQLite table '{table}' written ({len(df):,} rows) → {db_path}")
    except Exception as e:
        log.error(f"SQLite write failed (table={table}): {e}")


def save_trends(df: pd.DataFrame) -> None:
    _save_csv(df, TRENDS_CSV, "trends")
    _save_sqlite(df, "trends")


def save_drops(df: pd.DataFrame) -> None:
    _save_csv(df, DROPS_CSV, "drops")
    _save_sqlite(df, "drops")


def save_merged(df: pd.DataFrame) -> None:
    _save_csv(df, MERGED_CSV, "merged")
    _save_sqlite(df, "merged")


# ── Core merge ─────────────────────────────────────────────────────────────────

def merge_data(
    trends_df: pd.DataFrame,
    drops_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge trends + drops into the final research dataset.

    Parameters
    ----------
    trends_df : output of fetch_trends()
    drops_df  : output of scrape_drops()

    Returns
    -------
    Merged DataFrame ready for Poisson / decay / Markov analysis.
    """
    log.info(
        f"merge_data: trends={len(trends_df):,} rows, drops={len(drops_df):,} rows"
    )

    # ── Guard: both empty ────────────────────────────────────────────────────
    # Accessing columns on a column-less empty DataFrame raises KeyError.
    # Return early with a well-formed empty frame so callers can check .empty.
    if trends_df.empty and drops_df.empty:
        return pd.DataFrame(
            columns=["timestamp", "brand", "keyword", "hype_raw",
                     "event", "event_name", "source", "url"]
        )

    # ── 1. Normalise dates ────────────────────────────────────────────────────
    trends = trends_df.copy()
    drops  = drops_df.copy()

    if not trends.empty:
        trends["timestamp"] = pd.to_datetime(trends["timestamp"]).dt.tz_localize(None).dt.normalize()

    if not drops.empty:
        drops["timestamp"] = pd.to_datetime(drops["timestamp"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        # Ensure brand column is object dtype so the merge key types match
        # the trends side.  pd.NA in a newly-constructed frame can produce a
        # float64 "brand" column which causes a merge type-mismatch error.
        drops["brand"] = drops["brand"].astype(object)

    # ── 2. Prepare drops side: keep one row per (date, brand) for the join.
    #       If multiple articles dropped on the same day for the same brand,
    #       concatenate names and preserve all sources/URLs.
    if drops.empty:
        log.warning("drops_df is empty — merged dataset will have event=0 throughout")
        drops_agg = pd.DataFrame(
            columns=["timestamp", "brand", "event", "event_name", "source", "url"]
        )
    else:
        # dropna=False ensures rows where brand is NaN are kept as their own
        # group rather than being silently discarded by groupby's default
        # behaviour (dropna=True).  Without this, unmatched-brand drop events
        # are lost before the merge.
        drops_agg = (
    drops.dropna(subset=["timestamp"])
    .groupby(["timestamp", "brand"], as_index=False, dropna=False)
    .agg(
        event      = ("event", "max"),
        event_name = ("event_name", lambda x: " | ".join(x.dropna().unique())),
        source     = ("source", lambda x: ", ".join(x.dropna().unique())),
        url        = ("url", lambda x: " | ".join(x.dropna().unique())),
    )
)
        # groupby with dropna=False represents a missing group key as float NaN
        # in the output, which changes the brand column dtype to float64.
        # Cast back to object so the merge key types match the trends side and
        # pd.merge does not raise a type-mismatch ValueError.
        drops_agg["brand"] = drops_agg["brand"].astype(object)

    # ── 3. Left-join trends ← drops ──────────────────────────────────────────
    if trends.empty:
        log.warning("trends_df is empty — using drops-only dataset")
        merged = drops.copy()
        merged["hype_raw"] = float("nan")
        merged["keyword"]  = pd.NA
    else:
        merged = trends.merge(
            drops_agg[["timestamp", "brand", "event", "event_name", "source", "url"]],
            on=["timestamp", "brand"],
            how="left",
        )
        merged["event"] = merged["event"].fillna(0).astype(int)

    # ── 4. Append pure-drop rows (brands in drops but not in trends) ──────────
    if not drops.empty and not trends.empty:
        trend_brands = set(trends["brand"].dropna().unique())
        trend_dates = set(trends["timestamp"].dropna().unique())

        # Orphans by brand (brands with no Trends data at all)
        orphan_by_brand = drops[~drops["brand"].isin(trend_brands)].copy()

        # Orphans by date (drop events outside the trends date range)
        orphan_by_date = drops[
            drops["brand"].isin(trend_brands) &
            ~drops["timestamp"].isin(trend_dates)
        ].copy()

        orphan_drops = pd.concat([orphan_by_brand, orphan_by_date], ignore_index=True)

        if not orphan_drops.empty:
            orphan_drops["hype_raw"] = float("nan")
            orphan_drops["keyword"]  = pd.NA
            merged = pd.concat([merged, orphan_drops], ignore_index=True)
            log.info(
                f"  Appended {len(orphan_drops):,} drop-only rows "
                f"(brands with no Trends data or dates outside trends range)"
            )

    # ── 5. Final column order and types ──────────────────────────────────────
    col_order = [
        "timestamp", "brand", "keyword", "hype_raw",
        "event", "event_name", "source", "url",
    ]
    for c in col_order:
        if c not in merged.columns:
            merged[c] = pd.NA

    merged = merged[col_order]
    merged["hype_raw"] = pd.to_numeric(merged["hype_raw"], errors="coerce")
    merged["event"]    = pd.to_numeric(merged["event"],    errors="coerce").fillna(0).astype(int)
    merged = (
        merged.sort_values(["brand", "keyword", "timestamp"])
              .reset_index(drop=True)
    )

    log.info(
        f"merge_data complete: {len(merged):,} rows | "
        f"event rows: {merged['event'].sum():,} | "
        f"NaN hype_raw: {merged['hype_raw'].isna().sum():,} | "
        f"brands: {merged['brand'].nunique()}"
    )
    return merged


# ── Quality report ─────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame) -> None:
    """Print a concise data-quality summary to the log."""
    if df.empty:
        log.warning("quality_report: dataset is empty — nothing to report")
        return
    log.info("── Quality report ───────────────────────────────────")
    log.info(f"  Total rows      : {len(df):,}")
    log.info(f"  Date range      : {df['timestamp'].min()} → {df['timestamp'].max()}")
    log.info(f"  Brands          : {sorted(df['brand'].dropna().unique().tolist())}")
    log.info(f"  Keywords        : {df['keyword'].nunique()}")
    log.info(f"  Event rows      : {df['event'].sum():,}")
    hype = df["hype_raw"].dropna()
    if hype.empty:
        log.info("  hype_raw        : all NaN")
    else:
        log.info(f"  hype_raw NaN %  : {df['hype_raw'].isna().mean()*100:.1f}%")
        log.info(f"  hype_raw range  : {hype.min():.1f} – {hype.max():.1f}")
    log.info(f"  Sources         : {df['source'].dropna().unique().tolist()}")
    log.info("────────────────────────────────────────────────────")
