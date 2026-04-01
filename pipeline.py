"""
pipeline.py — Main entry point for the hype data collection pipeline.

Usage
-----
    # Full run (trends + drops + merge):
    python pipeline.py

    # Trends only:
    python pipeline.py --trends-only

    # Drops only:
    python pipeline.py --drops-only

    # Custom date range:
    python pipeline.py --start 2022-01-01 --end 2024-01-01

    # Skip merge (collect but don't join):
    python pipeline.py --no-merge

    # Resume from saved CSVs (skip scraping):
    python pipeline.py --from-cache

Resilience
----------
• Each stage is isolated: a failure in trends collection does not abort
  the drops scraper, and vice versa.
• Intermediate CSVs are written after each stage so a crash mid-pipeline
  doesn't lose everything.
• The --from-cache flag lets you re-run the merge without re-scraping.
"""

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from config import DROPS_CSV, MERGED_CSV, TRENDS_CSV
from drops import scrape_drops
from logger import get_logger
from merger import merge_data, quality_report, save_drops, save_merged, save_trends
from trends import fetch_trends

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Luxury Hype Pipeline — data collection for stochastic modelling"
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD (default: 3 years ago)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--trends-only",
        action="store_true",
        help="Run only the Google Trends collector",
    )
    parser.add_argument(
        "--drops-only",
        action="store_true",
        help="Run only the drop event scraper",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Collect data but skip the merge step",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Load trends.csv + drops.csv from disk and re-run merge only",
    )
    return parser.parse_args()


def _load_from_disk() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously saved CSVs; return empty DataFrames if missing."""
    trends_df = drops_df = pd.DataFrame()

    if TRENDS_CSV.exists():
        try:
            trends_df = pd.read_csv(TRENDS_CSV, parse_dates=["timestamp"])
            log.info(f"Loaded trends from cache: {len(trends_df):,} rows")
        except Exception as e:
            log.error(f"Failed to load trends cache: {e}")
    else:
        log.warning(f"Trends cache not found: {TRENDS_CSV}")

    if DROPS_CSV.exists():
        try:
            drops_df = pd.read_csv(DROPS_CSV, parse_dates=["timestamp"])
            log.info(f"Loaded drops from cache: {len(drops_df):,} rows")
        except Exception as e:
            log.error(f"Failed to load drops cache: {e}")
    else:
        log.warning(f"Drops cache not found: {DROPS_CSV}")

    return trends_df, drops_df


def run(
    start_date=None,
    end_date=None,
    run_trends: bool = True,
    run_drops:  bool = True,
    do_merge:   bool = True,
    from_cache: bool = False,
) -> pd.DataFrame:
    """
    Orchestrate the full pipeline.

    Parameters
    ----------
    start_date  : datetime.date or None
    end_date    : datetime.date or None
    run_trends  : whether to collect Google Trends data
    run_drops   : whether to scrape drop events
    do_merge    : whether to run the merge step
    from_cache  : if True, skip collection and load from saved CSVs

    Returns
    -------
    Merged DataFrame (empty if do_merge=False)
    """
    log.info("═" * 60)
    log.info("HYPE PIPELINE — START")
    log.info("═" * 60)

    # ── Date range ────────────────────────────────────────────────────────────
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=3 * 365)

    log.info(f"Date range: {start_date} → {end_date}")

    # ── Load from cache or collect fresh ─────────────────────────────────────
    if from_cache:
        log.info("Loading data from saved CSVs (--from-cache)")
        trends_df, drops_df = _load_from_disk()
    else:
        trends_df = pd.DataFrame()
        drops_df  = pd.DataFrame()

        # [FIX P1] Removed misindented AI-generated comment block (6-space indent
        # inside an 8-space block) that was cosmetically wrong and confusing.
        # The logic itself was always correct — only the comments were malformed.

        # ── Stage 1: Google Trends ────────────────────────────────────────────
        if run_trends:
            log.info("── Stage 1: Google Trends ───────────────────────────")
            try:
                trends_df = fetch_trends(start_date=start_date, end_date=end_date)
                save_trends(trends_df)
                log.info(f"  Trends saved: {len(trends_df):,} rows")
            except Exception as e:
                log.error(f"Trends stage failed: {e}")
                trends_df = pd.DataFrame()
        else:
            log.info("── Stage 1: Trends skipped (--drops-only)")
            if TRENDS_CSV.exists():
                trends_df, _ = _load_from_disk()

        # ── Stage 2: Drop events ──────────────────────────────────────────────
        if run_drops:
            log.info("── Stage 2: Drop event scraping ─────────────────────")
            try:
                drops_df = scrape_drops()
                save_drops(drops_df)
                log.info(f"  Drops saved: {len(drops_df):,} rows")
            except Exception as e:
                log.error(f"Drops stage failed: {e}")
                drops_df = pd.DataFrame()
        else:
            log.info("── Stage 2: Drops skipped (--trends-only)")
            if DROPS_CSV.exists():
                _, drops_df = _load_from_disk()

    # ── Stage 3: Merge ────────────────────────────────────────────────────────
    merged_df = pd.DataFrame()
    if do_merge:
        log.info("── Stage 3: Merge ───────────────────────────────────")
        try:
            merged_df = merge_data(trends_df, drops_df)
            save_merged(merged_df)
            quality_report(merged_df)
        except Exception as e:
            log.error(f"Merge stage failed: {e}")
    else:
        log.info("── Stage 3: Merge skipped (--no-merge)")

    log.info("═" * 60)
    log.info(
        f"PIPELINE COMPLETE  |  "
        f"trends={len(trends_df):,}  drops={len(drops_df):,}  merged={len(merged_df):,}"
    )
    log.info("═" * 60)
    return merged_df


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()

    # [FIX P2] Guard: --trends-only + --drops-only together silently ran
    # neither stage (run_trends = not drops_only = False; run_drops = not
    # trends_only = False). Now exits with a clear error message.
    if args.trends_only and args.drops_only:
        log.error(
            "--trends-only and --drops-only are mutually exclusive. "
            "Use one flag to restrict which stage runs, or neither for a full run."
        )
        sys.exit(2)

    # [FIX P3] Guard: --from-cache + --no-merge is a useless no-op.
    if args.from_cache and args.no_merge:
        log.error(
            "--from-cache loads saved CSVs and re-runs the merge. "
            "Combining it with --no-merge skips the only thing --from-cache does."
        )
        sys.exit(2)

    # [FIX P4] Validate date strings and logical order.
    try:
        start = date.fromisoformat(args.start) if args.start else None
        end   = date.fromisoformat(args.end)   if args.end   else None
    except ValueError as exc:
        log.error(f"Invalid date format: {exc}. Expected YYYY-MM-DD.")
        sys.exit(2)

    if start and end and start >= end:
        log.error(f"--start ({start}) must be strictly before --end ({end}).")
        sys.exit(2)

    # [FIX P5] Removed unused `from pathlib import Path` import.
    run_trends = not args.drops_only
    run_drops  = not args.trends_only
    do_merge   = not args.no_merge

    result = run(
        start_date  = start,
        end_date    = end,
        run_trends  = run_trends,
        run_drops   = run_drops,
        do_merge    = do_merge,
        from_cache  = args.from_cache,
    )

    if result.empty:
        log.warning("Pipeline produced an empty dataset.")
        sys.exit(1)
    else:
        log.info(f"Final dataset: {len(result):,} rows → {MERGED_CSV}")
        sys.exit(0)
