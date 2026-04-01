"""
tests/test_pipeline.py
======================
Unit tests for the luxury hype pipeline.

Covers
------
  - drops._parse_date          : date parsing correctness (was broken)
  - drops._matched_brands      : brand keyword matching
  - drops._matched_drop_keyword: drop keyword matching
  - drops._normalise           : NaN-brand row preservation + tz stripping
  - merger.merge_data          : merge logic, NaN brand, multi-event aggregation
  - pipeline CLI flags         : contradictory flag detection

Run
---
    pip install pytest && pytest tests/ -v
    # or without pytest:
    python -m unittest tests/test_pipeline.py -v
"""

import sys
import unittest
from datetime import date as date_cls, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from drops import _matched_brands, _matched_drop_keyword, _normalise, _parse_date
from merger import merge_data


# ══════════════════════════════════════════════════════════════════════════════
# _parse_date
# ══════════════════════════════════════════════════════════════════════════════

class TestParseDate(unittest.TestCase):

    def test_rfc2822(self):
        result = _parse_date("Mon, 15 Jan 2024 10:30:00 +0000")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.month, 1)
        self.assertEqual(result.day, 15)

    def test_iso_with_tz_offset(self):
        result = _parse_date("2024-03-20T14:00:00+00:00")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.day, 20)

    def test_iso_with_z(self):
        result = _parse_date("2024-06-01T09:00:00Z")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.day, 1)

    def test_iso_datetime_no_tz(self):
        result = _parse_date("2023-11-22 18:45:00")
        self.assertIsNotNone(result)
        self.assertEqual(result.month, 11)

    def test_iso_date_only(self):
        # Original bug: raw[:len('%Y-%m-%d')] = raw[:8] = '2024-01-' — always failed.
        result = _parse_date("2022-07-04")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2022)
        self.assertEqual(result.month, 7)
        self.assertEqual(result.day, 4)

    def test_empty_string(self):
        self.assertIsNone(_parse_date(""))

    def test_whitespace_only(self):
        self.assertIsNone(_parse_date("   "))

    def test_none_input(self):
        self.assertIsNone(_parse_date(None))  # type: ignore[arg-type]

    def test_garbage_input(self):
        self.assertIsNone(_parse_date("not-a-date!!"))

    def test_full_iso_date_not_truncated(self):
        # Belt-and-suspenders check that '2024-01-15' is NOT truncated to '2024-01-'
        result = _parse_date("2024-01-15")
        self.assertIsNotNone(result, "Full ISO date-only string must parse correctly")
        self.assertEqual(result.day, 15)


# ══════════════════════════════════════════════════════════════════════════════
# Brand and keyword matching
# ══════════════════════════════════════════════════════════════════════════════

class TestBrandMatching(unittest.TestCase):

    def test_single_brand(self):
        self.assertIn("Supreme", _matched_brands("Supreme drops new hoodie collection"))

    def test_multiple_brands(self):
        brands = _matched_brands("Nike x Supreme limited sneaker release")
        self.assertIn("Nike", brands)
        self.assertIn("Supreme", brands)

    def test_case_insensitive(self):
        self.assertIn("Nike", _matched_brands("NIKE Air Jordan 1 drop"))

    def test_no_match_returns_empty(self):
        self.assertEqual([], _matched_brands("Random article about weather"))

    def test_alias_match(self):
        # "Air Jordan" is in Nike's drop_kw
        self.assertIn("Nike", _matched_brands("Air Jordan 4 Retro release date"))

    def test_hermes_ascii_alias(self):
        # "Hermes" (no accent) should match Hermès
        self.assertIn("Hermès", _matched_brands("Hermes Birkin bag waitlist opens"))


class TestDropKeyword(unittest.TestCase):

    def test_drop_keyword(self):
        self.assertEqual("drop", _matched_drop_keyword("Supreme Thursday drop live"))

    def test_release_keyword(self):
        self.assertEqual("release", _matched_drop_keyword("Nike Air Force 1 release date"))

    def test_limited_keyword(self):
        self.assertEqual("limited", _matched_drop_keyword("Yeezy limited edition"))

    def test_returns_first_match_in_list_order(self):
        # "drop" appears before "release" in DROP_KEYWORDS
        self.assertEqual("drop", _matched_drop_keyword("Nike drop and release announcement"))

    def test_no_match(self):
        self.assertIsNone(_matched_drop_keyword("Interview with a designer"))

    def test_case_insensitive(self):
        self.assertEqual("exclusive", _matched_drop_keyword("EXCLUSIVE Balenciaga collab"))


# ══════════════════════════════════════════════════════════════════════════════
# _normalise
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalise(unittest.TestCase):

    @staticmethod
    def _raw(title, url="http://example.com", ts=None):
        return {
            "raw_title": title,
            "url":       url,
            "timestamp": ts or datetime(2024, 1, 15, 10, 0, 0),
            "source":    "TestSource",
        }

    def test_non_drop_article_filtered_out(self):
        df = _normalise([self._raw("Weather forecast for Paris")])
        self.assertTrue(df.empty)

    def test_one_article_two_brand_matches_gives_two_rows(self):
        df = _normalise([self._raw("Nike x Supreme limited sneaker drop")])
        self.assertGreaterEqual(len(df), 2)
        self.assertIn("Nike",    list(df["brand"]))
        self.assertIn("Supreme", list(df["brand"]))

    def test_nan_brand_preserved_when_no_brand_matched(self):
        # "Mystery brand" matches no brand keyword → NaN brand row must be kept
        df = _normalise([self._raw("Mystery brand limited release announced")])
        self.assertEqual(len(df), 1)
        self.assertTrue(pd.isna(df.iloc[0]["brand"]))

    def test_deduplication_by_url_and_brand(self):
        raw = [
            self._raw("Nike Air Jordan drop", url="http://a.com"),
            self._raw("Nike Air Jordan drop", url="http://a.com"),
        ]
        df = _normalise(raw)
        nike = df[df["brand"] == "Nike"]
        self.assertEqual(len(nike), 1)

    def test_output_timestamp_is_tz_naive(self):
        df = _normalise([self._raw("Supreme drop live")])
        self.assertIsNone(df["timestamp"].dt.tz)

    def test_event_column_always_one(self):
        df = _normalise([self._raw("Yeezy limited release")])
        self.assertTrue((df["event"] == 1).all())

    def test_empty_input_returns_correct_columns(self):
        df = _normalise([])
        self.assertTrue(df.empty)
        expected = ["timestamp", "brand", "event_name", "event", "source", "url", "keyword"]
        self.assertEqual(list(df.columns), expected)


# ══════════════════════════════════════════════════════════════════════════════
# merge_data
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeData(unittest.TestCase):

    @staticmethod
    def _trends(rows):
        return pd.DataFrame(rows, columns=["timestamp", "brand", "keyword", "hype_raw"])

    @staticmethod
    def _drops(rows):
        return pd.DataFrame(
            rows,
            columns=["timestamp", "brand", "event_name", "event", "source", "url", "keyword"]
        )

    def test_event_flag_set_on_matching_day(self):
        trends = self._trends([
            ["2024-01-15", "Nike", "Nike drop", 80.0],
            ["2024-01-16", "Nike", "Nike drop", 60.0],
        ])
        drops = self._drops([
            ["2024-01-15", "Nike", "Air Jordan 1 Drop", 1, "Hypebeast", "http://hb.com", "drop"],
        ])
        merged = merge_data(trends, drops)
        jan15 = merged[merged["timestamp"].astype(str).str.startswith("2024-01-15")]
        jan16 = merged[merged["timestamp"].astype(str).str.startswith("2024-01-16")]
        self.assertEqual(int(jan15.iloc[0]["event"]), 1)
        self.assertEqual(int(jan16.iloc[0]["event"]), 0)

    def test_no_drops_gives_all_zero_events(self):
        trends = self._trends([["2024-02-01", "Supreme", "Supreme drop", 50.0]])
        merged = merge_data(trends, self._drops([]))
        self.assertTrue((merged["event"] == 0).all())

    def test_nan_brand_drop_rows_preserved(self):
        # This was silently lost due to groupby(dropna=True) — now fixed.
        trends = self._trends([["2024-01-15", "Nike", "Nike drop", 70.0]])
        drops  = self._drops([
            ["2024-01-15", None, "Unknown collab drop", 1, "SneakerNews", "http://sn.com", "drop"],
        ])
        merged = merge_data(trends, drops)
        nan_rows = merged[merged["brand"].isna()]
        self.assertGreaterEqual(len(nan_rows), 1, "NaN-brand drop rows must survive the merge")

    def test_multiple_drops_same_day_concatenated(self):
        trends = self._trends([["2024-03-10", "Nike", "Nike drop", 90.0]])
        drops  = self._drops([
            ["2024-03-10", "Nike", "Air Jordan 1",  1, "Hypebeast",   "http://a.com", "drop"],
            ["2024-03-10", "Nike", "Nike Dunk Low", 1, "SneakerNews", "http://b.com", "release"],
        ])
        merged = merge_data(trends, drops)
        row = merged[merged["timestamp"].astype(str).str.startswith("2024-03-10")]
        self.assertEqual(len(row), 1)
        self.assertIn("Air Jordan 1",  row.iloc[0]["event_name"])
        self.assertIn("Nike Dunk Low", row.iloc[0]["event_name"])

    def test_orphan_brand_has_nan_hype_raw(self):
        trends = self._trends([["2024-01-15", "Nike", "Nike drop", 70.0]])
        drops  = self._drops([
            ["2024-01-20", "Palace", "Palace Thursday drop", 1, "Hypebeast", "http://p.com", "drop"],
        ])
        merged = merge_data(trends, drops)
        palace = merged[merged["brand"] == "Palace"]
        self.assertGreaterEqual(len(palace), 1)
        self.assertTrue(pd.isna(palace.iloc[0]["hype_raw"]))

    def test_both_empty_returns_empty(self):
        merged = merge_data(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(merged.empty)

    def test_output_has_all_required_columns(self):
        trends = self._trends([["2024-01-01", "Nike", "Nike drop", 50.0]])
        merged = merge_data(trends, self._drops([]))
        required = {"timestamp", "brand", "keyword", "hype_raw", "event", "event_name", "source", "url"}
        self.assertTrue(required.issubset(set(merged.columns)))

    def test_hype_raw_is_float(self):
        trends = self._trends([["2024-01-01", "Nike", "Nike drop", 50.0]])
        merged = merge_data(trends, self._drops([]))
        self.assertTrue(pd.api.types.is_float_dtype(merged["hype_raw"]))

    def test_event_is_integer(self):
        trends = self._trends([["2024-01-01", "Nike", "Nike drop", 50.0]])
        merged = merge_data(trends, self._drops([]))
        self.assertTrue(pd.api.types.is_integer_dtype(merged["event"]))


# ══════════════════════════════════════════════════════════════════════════════
# CLI flag validation
# ══════════════════════════════════════════════════════════════════════════════

class TestCLIFlags(unittest.TestCase):
    """
    Tests the guard logic in pipeline.__main__ by calling _parse_args()
    with a patched sys.argv and replicating the guard checks.
    No network calls or file I/O are made.
    """

    @staticmethod
    def _check_flags(argv: list[str]) -> int:
        """
        Run the flag-validation block from pipeline.__main__ and return the
        exit code it would produce (0 = valid, 2 = rejected).
        """
        with patch("sys.argv", ["pipeline.py"] + argv):
            from pipeline import _parse_args
            args = _parse_args()

            if args.trends_only and args.drops_only:
                return 2
            if args.from_cache and args.no_merge:
                return 2
            try:
                start = date_cls.fromisoformat(args.start) if args.start else None
                end   = date_cls.fromisoformat(args.end)   if args.end   else None
            except ValueError:
                return 2
            if start and end and start >= end:
                return 2
            return 0

    def test_both_only_flags_rejected(self):
        self.assertEqual(2, self._check_flags(["--trends-only", "--drops-only"]))

    def test_from_cache_with_no_merge_rejected(self):
        self.assertEqual(2, self._check_flags(["--from-cache", "--no-merge"]))

    def test_invalid_start_date_rejected(self):
        self.assertEqual(2, self._check_flags(["--start", "not-a-date"]))

    def test_invalid_end_date_rejected(self):
        self.assertEqual(2, self._check_flags(["--end", "32/13/2024"]))

    def test_start_after_end_rejected(self):
        self.assertEqual(2, self._check_flags(["--start", "2024-06-01", "--end", "2024-01-01"]))

    def test_start_equal_end_rejected(self):
        self.assertEqual(2, self._check_flags(["--start", "2024-01-01", "--end", "2024-01-01"]))

    def test_trends_only_alone_accepted(self):
        self.assertEqual(0, self._check_flags(["--trends-only"]))

    def test_drops_only_alone_accepted(self):
        self.assertEqual(0, self._check_flags(["--drops-only"]))

    def test_valid_date_range_accepted(self):
        self.assertEqual(0, self._check_flags(["--start", "2023-01-01", "--end", "2024-01-01"]))

    def test_no_flags_accepted(self):
        self.assertEqual(0, self._check_flags([]))

# ══════════════════════════════════════════════════════════════════════════════
# trends loop safety
# ══════════════════════════════════════════════════════════════════════════════

class TestTrends(unittest.TestCase):

    def test_trends_no_infinite_loop(self):
        from trends import _fetch_keyword_history
        from datetime import date

        class Dummy:
            def build_payload(self, *args, **kwargs):
                pass

            def interest_over_time(self):
                import pandas as pd
                return pd.DataFrame({
                    "date": pd.date_range("2024-01-01", periods=5),
                    "test": [1, 2, 3, 4, 5],
                    "isPartial": [False] * 5
                }).set_index("date")

        df = _fetch_keyword_history(
            Dummy(),
            "test",
            date(2024, 1, 1),
            date(2024, 1, 10)
        )

        self.assertFalse(df.empty)
if __name__ == "__main__":
    unittest.main(verbosity=2)
