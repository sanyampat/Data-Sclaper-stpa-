"""
drops.py — Drop event scraper.

Architecture
------------
• RSS-first: all four sources expose RSS feeds. BeautifulSoup parses the XML
  in O(1) network calls, giving us clean title / date / link without rendering JS.
• HTML fallback: if RSS returns nothing (or max_p > 0), we fall back to paginated
  HTML scraping with ThreadPoolExecutor(MAX_WORKERS).
• Brand extraction: we match article titles against each brand's drop_kw list
  (case-insensitive substring search). An article may match multiple brands;
  we emit one row per matching brand.
• Deduplication: final de-dup on (url, brand) so the same article doesn't
  appear twice for the same brand.

Output schema
-------------
  timestamp   : datetime (publication date) — NaT if unparseable
  brand       : matched brand name (str) — NaN if no match
  event_name  : article title (str)
  event       : 1 (every scraped article is a candidate drop event)
  source      : source name from config.SOURCES
  url         : canonical article URL
  keyword     : matched drop keyword that triggered inclusion
"""

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    BRANDS,
    DELAY_MAX,
    DELAY_MIN,
    DROP_KEYWORDS,
    MAX_WORKERS,
    REQUEST_TIMEOUT,
    SOURCES,
    USER_AGENTS,
)
from logger import get_logger

log = get_logger(__name__)

# [FIX B1] Removed the dead `extract_brand()` function that was defined here.
# It hard-coded ["gucci","prada","hermes"] (a tiny, wrong subset of brands),
# was never called anywhere, and shadowed the correct _matched_brands() logic.

# ── HTTP helpers ───────────────────────────────────────────────────────────────
# [FIX B2] Removed module-level _SESSION singleton and _session() factory.
# A single Session shared across ThreadPoolExecutor workers causes race
# conditions on the connection pool.  Each _get() call now creates its own
# short-lived Session inside a `with` block, which is thread-safe.

def _get(url: str, retries: int = 3) -> Optional[requests.Response]:
    """GET with retry + polite delay.  Thread-safe: one Session per call."""
    for attempt in range(1, retries + 1):
        try:
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            with requests.Session() as session:
                session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            log.warning(f"  GET failed (attempt {attempt}/{retries}) {url}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


# ── Date parsing ───────────────────────────────────────────────────────────────

def _parse_date(raw: str) -> Optional[datetime]:
    """Try RFC-2822 (RSS) then common ISO formats, then pandas fallback."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    # RFC-2822 (standard RSS pubDate)
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        pass
    # [FIX B3] The original code used raw[:len(fmt)] which slices by the
    # format-string length (e.g. len('%Y-%m-%d') == 8), not the rendered
    # date length (10).  raw[:8] of '2024-01-15' yields '2024-01-', which
    # never matches '%Y-%m-%d'.  All ISO formats silently failed and fell
    # through to pandas.  Fix: pass the full string to strptime.
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    # Fallback: let pandas try
    try:
        return pd.to_datetime(raw)
    except Exception:
        return None


# ── Brand & keyword matching ───────────────────────────────────────────────────

def _matched_brands(text: str) -> list[str]:
    """Return all brand names whose drop_kw appear in text (case-insensitive)."""
    text_lower = text.lower()
    return [
        brand
        for brand, cfg in BRANDS.items()
        if any(kw.lower() in text_lower for kw in cfg["drop_kw"])
    ]


def _matched_drop_keyword(text: str) -> Optional[str]:
    """Return the first DROP_KEYWORDS match found in text, or None."""
    text_lower = text.lower()
    return next((kw for kw in DROP_KEYWORDS if kw in text_lower), None)


# ── RSS parser ─────────────────────────────────────────────────────────────────

def _parse_rss_feed(source_name: str, feed_url: str) -> list[dict]:
    """Parse one RSS/Atom feed and return a list of raw article dicts."""
    resp = _get(feed_url)
    if resp is None:
        log.error(f"  RSS unreachable: {feed_url}")
        return []

    soup = BeautifulSoup(resp.content, "xml")
    items = soup.find_all("item")          # RSS 2.0
    if not items:
        items = soup.find_all("entry")     # Atom

    log.info(f"  RSS {source_name} ({feed_url.split('/')[2]}) → {len(items)} items")

    rows = []
    for item in items:
        try:
            title_tag = item.find("title")
            link_tag  = item.find("link")
            date_tag  = (
                item.find("pubDate")
                or item.find("published")
                or item.find("updated")
                or item.find("dc:date")
            )

            title = title_tag.get_text(strip=True) if title_tag else ""
            if not title:
                continue

            # Link: RSS 2.0 uses text content; Atom uses href attribute
            if link_tag:
                link = link_tag.get("href") or link_tag.get_text(strip=True)
            else:
                link = ""

            date_raw    = date_tag.get_text(strip=True) if date_tag else ""
            parsed_date = _parse_date(date_raw)

            rows.append({
                "raw_title": title,
                "url":       link,
                "timestamp": parsed_date,
                "source":    source_name,
            })
        except Exception as e:
            log.debug(f"  RSS item parse error: {e}")
            continue

    return rows


# ── HTML paginated scraper ─────────────────────────────────────────────────────

def _scrape_html_page(source_name: str, url: str) -> list[dict]:
    """
    Generic HTML scraper.  Looks for <article> or common card patterns.
    This is a best-effort fallback — structure varies by site.
    """
    resp = _get(url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []

    # Try <article> tags first (semantic HTML)
    cards = soup.find_all("article")
    if not cards:
        cards = soup.find_all(
            True,
            class_=re.compile(r"(post|article|story|card|item)", re.I),
        )

    log.debug(f"  HTML {source_name} {url} → {len(cards)} cards")

    for card in cards:
        try:
            heading = (
                card.find("h1")
                or card.find("h2")
                or card.find("h3")
                or card.find(class_=re.compile(r"title|headline", re.I))
            )
            if not heading:
                continue
            title = heading.get_text(strip=True)
            if not title:
                continue

            a    = card.find("a", href=True)
            link = urljoin(url, a["href"]) if a else ""

            time_el  = (
                card.find("time")
                or card.find(class_=re.compile(r"date|time|published", re.I))
            )
            date_raw = ""
            if time_el:
                date_raw = time_el.get("datetime") or time_el.get_text(strip=True)
            parsed_date = _parse_date(date_raw)

            rows.append({
                "raw_title": title,
                "url":       link,
                "timestamp": parsed_date,
                "source":    source_name,
            })
        except Exception as e:
            log.debug(f"  HTML card parse error: {e}")
            continue

    return rows


# ── Per-source fetcher ─────────────────────────────────────────────────────────

def _fetch_source(source_name: str, cfg: dict) -> list[dict]:
    rows = []

    for feed_url in cfg.get("rss", []):
        try:
            rows.extend(_parse_rss_feed(source_name, feed_url))
        except Exception as e:
            log.error(f"  RSS fetch error ({source_name}, {feed_url}): {e}")

    pages_tmpl = cfg.get("pages", [])
    max_p      = cfg.get("max_p", 0)

    if pages_tmpl and max_p > 0:
        page_urls = [
            tmpl.format(p)
            for tmpl in pages_tmpl
            for p in range(1, max_p + 1)
        ]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(_scrape_html_page, source_name, u): u
                for u in page_urls
            }
            for fut in as_completed(futures):
                url = futures[fut]
                try:
                    rows.extend(fut.result())
                except Exception as e:
                    log.error(f"  HTML page error ({url}): {e}")

    log.info(f"  {source_name}: {len(rows)} raw items collected")
    return rows


# ── Filter, brand-match, and normalise ────────────────────────────────────────

def _normalise(raw_rows: list[dict]) -> pd.DataFrame:
    """
    1. Filter to rows that contain at least one DROP_KEYWORD.
    2. Expand: one row per (article × matching brand).
    3. If no brand matches, keep the row with brand = NaN.
    4. Return clean DataFrame.
    """
    out = []
    for row in raw_rows:
        title = row.get("raw_title", "")
        if not title:
            continue

        kw_match = _matched_drop_keyword(title)
        if kw_match is None:
            continue

        brands = _matched_brands(title)

        if brands:
            for brand in brands:
                out.append({
                    "timestamp":  row["timestamp"],
                    "brand":      brand,
                    "event_name": title,
                    "event":      1,
                    "source":     row["source"],
                    "url":        row["url"],
                    "keyword":    kw_match,
                })
        else:
            out.append({
                "timestamp":  row["timestamp"],
                "brand":      pd.NA,
                "event_name": title,
                "event":      1,
                "source":     row["source"],
                "url":        row["url"],
                "keyword":    kw_match,
            })

    if not out:
        return pd.DataFrame(
            columns=["timestamp", "brand", "event_name", "event", "source", "url", "keyword"]
        )

    df = pd.DataFrame(out)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # [FIX B4] tz_localize(None) raises TypeError on tz-aware Series.
    # tz_convert(None) is the correct way to strip timezone info.
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    before = len(df)
    df = df.drop_duplicates(subset=["url", "brand"], keep="first")
    log.info(f"Deduplication: {before} → {len(df)} rows")

    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def scrape_drops() -> pd.DataFrame:
    """
    Scrape drop events from all sources defined in config.SOURCES.

    Returns
    -------
    DataFrame with columns:
        timestamp, brand, event_name, event, source, url, keyword
    """
    log.info(f"scrape_drops: collecting from {len(SOURCES)} sources")
    all_raw: list[dict] = []

    for source_name, cfg in SOURCES.items():
        log.info(f"Source: {source_name}")
        try:
            raw = _fetch_source(source_name, cfg)
            all_raw.extend(raw)
        except Exception as e:
            log.error(f"Source '{source_name}' failed entirely: {e}")
            continue

    log.info(f"scrape_drops: {len(all_raw)} raw items → filtering…")
    df = _normalise(all_raw)

    log.info(
        f"scrape_drops complete: {len(df):,} rows, "
        f"{df['brand'].nunique()} brands, "
        f"{df['source'].nunique()} sources"
    )
    return df
