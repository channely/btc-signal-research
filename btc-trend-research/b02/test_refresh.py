"""Offline validation, provenance, and chronology tests for the staged refresh.

Run: .venv/bin/python -m unittest discover -s btc-trend-research/b02 -v
No exchange requests or tracked research artifacts are changed by these tests.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("b02_refresh", ROOT / "refresh.py")
refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh)


def frame(start="2026-09-15", days=5):
    # Reading a CSV removes DatetimeIndex.freq, as does the actual frozen input.
    index = pd.DatetimeIndex(list(pd.date_range(start, periods=days)), name="date")
    return pd.DataFrame([[100.0, 110.0, 90.0, 101.0, 20.0]] * days,
                        index=index, columns=refresh.COLUMNS)


def raw_rows(data):
    result = []
    for stamp, row in data.iterrows():
        opening = int(stamp.timestamp() * 1000)
        result.append([opening, *[str(value) for value in row],
                       opening + refresh.DAY_MS - 1, "0", 1, "0", "0", "0"])
    return result


def encoded(value):
    return json.dumps(value).encode()


class CandleValidationTests(unittest.TestCase):
    def setUp(self):
        self.data = frame()
        self.cutoff = pd.Timestamp("2026-09-20")

    def test_accepts_only_fully_closed_valid_daily_frame(self):
        refresh.validate(self.data, self.cutoff)

    def test_rejects_empty_frame(self):
        with self.assertRaises(ValueError):
            refresh.validate(self.data.iloc[:0], self.cutoff)

    def test_rejects_gap_duplicate_and_reverse_order(self):
        variants = [self.data.drop(self.data.index[2]),
                    pd.concat([self.data, self.data.iloc[-1:]]),
                    self.data.iloc[::-1]]
        for data in variants:
            with self.subTest(index=list(data.index)), self.assertRaises(ValueError):
                refresh.validate(data, self.cutoff)

    def test_rejects_nonmidnight_daily_rows(self):
        self.data.index += pd.Timedelta(hours=12)
        with self.assertRaises(ValueError):
            refresh.validate(self.data, self.cutoff)

    def test_rejects_noncanonical_timezone_index(self):
        self.data.index = self.data.index.tz_localize("America/New_York")
        with self.assertRaises(ValueError):
            refresh.validate(self.data, self.cutoff)

    def test_rejects_unclosed_and_future_rows(self):
        for cutoff in [self.data.index.max(), self.data.index[-2]]:
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                refresh.validate(self.data, cutoff)

    def test_rejects_wrong_or_reordered_columns(self):
        for data in [self.data.drop(columns="volume"), self.data[self.data.columns[::-1]]]:
            with self.assertRaises(ValueError):
                refresh.validate(data, self.cutoff)

    def test_rejects_invalid_numeric_ohlcv(self):
        for column, value in [("open", 0), ("close", -1), ("volume", -1),
                              ("high", 99), ("low", 102), ("close", np.nan),
                              ("volume", np.inf), ("low", -np.inf)]:
            data = self.data.copy()
            data.loc[data.index[2], column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                refresh.validate(data, self.cutoff)


class DecodeBatchTests(unittest.TestCase):
    def setUp(self):
        self.data = frame()
        self.rows = raw_rows(self.data)
        self.start = self.rows[0][0]
        self.stop = self.rows[-1][6] + 1

    def test_decodes_closed_rows_and_preserves_values(self):
        result = refresh.decode_batch(encoded(self.rows), self.start, self.stop)
        result.index = result.index.as_unit(self.data.index.unit)
        pd.testing.assert_frame_equal(result, self.data, check_freq=False)

    def test_rejects_nonlist_empty_and_short_rows(self):
        for rows in [{"code": -1}, [], [self.rows[0][:-1]], [None]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                refresh.decode_batch(encoded(rows), self.start, self.stop)

    def test_rejects_gap_duplicate_wrong_close_or_start(self):
        gap = self.rows[:1] + self.rows[2:]
        duplicate = self.rows[:1] + self.rows
        wrong_close = copy.deepcopy(self.rows)
        wrong_close[0][6] -= 1
        wrong_start = copy.deepcopy(self.rows)
        wrong_start[0][0] += 1
        for rows in [gap, duplicate, wrong_close, wrong_start]:
            with self.assertRaises(ValueError):
                refresh.decode_batch(encoded(rows), self.start, self.stop)

    def test_rejects_candle_not_closed_before_cutoff(self):
        with self.assertRaises(ValueError):
            refresh.decode_batch(encoded(self.rows), self.start, self.stop - 1)


class ReplayIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "snapshot"
        self.source.mkdir()
        self.data = frame()
        self.frozen = self.data.iloc[:2].copy()
        self.frozen.to_csv(self.root / "btc_daily.csv", float_format="%.10f")
        self.patch_research = mock.patch.object(refresh, "RESEARCH", self.root)
        self.patch_research.start()
        self.addCleanup(self.patch_research.stop)
        self.server_ms = int(pd.Timestamp("2026-09-20T12:00:00").timestamp() * 1000)
        self.rows = raw_rows(self.data.iloc[1:])
        with mock.patch.object(refresh, "read_json", side_effect=[
                encoded({"serverTime": self.server_ms}), encoded(self.rows)]) as get, \
                mock.patch.object(refresh, "datetime", wraps=refresh.datetime) as clock:
            clock.now.return_value = pd.Timestamp("2026-09-20T12:01:00Z").to_pydatetime()
            acquired, self.manifest = refresh.acquire(self.source, self.frozen, "2026-09-20")
        self.assertEqual(get.call_count, 2)
        pd.testing.assert_frame_equal(acquired, self.data, check_freq=False)

    def save_manifest(self):
        refresh.dump(self.source / "manifest.json", self.manifest)

    def update_raw(self, rows):
        body = encoded(rows)
        (self.source / self.manifest["requests"][0]["raw_file"]).write_bytes(body)
        self.manifest["requests"][0]["sha256"] = hashlib.sha256(body).hexdigest()
        self.save_manifest()

    def test_acquired_snapshot_replays_identically_without_network(self):
        with mock.patch.object(refresh, "read_json", side_effect=AssertionError("Network forbidden")):
            replayed, manifest = refresh.replay(self.source, self.frozen)
        pd.testing.assert_frame_equal(replayed, self.data, check_freq=False)
        self.assertEqual(manifest["cutoff_exclusive"], "2026-09-20")
        self.assertEqual(manifest["last_closed_candle"], "2026-09-19")

    def test_acquisition_paginates_without_reusing_overlap(self):
        out = self.root / "paginated"
        out.mkdir()
        with mock.patch.object(refresh, "read_json", side_effect=[
                encoded({"serverTime": self.server_ms}),
                encoded(self.rows[:2]), encoded(self.rows[2:])]) as get, \
                mock.patch.object(refresh, "datetime", wraps=refresh.datetime) as clock:
            clock.now.return_value = pd.Timestamp("2026-09-20T12:01:00Z").to_pydatetime()
            acquired, manifest = refresh.acquire(out, self.frozen, "2026-09-20")
        self.assertEqual(get.call_count, 3)
        self.assertEqual(len(manifest["requests"]), 2)
        pd.testing.assert_frame_equal(acquired, self.data, check_freq=False)
        replayed, _ = refresh.replay(out, self.frozen)
        pd.testing.assert_frame_equal(replayed, self.data, check_freq=False)

    def test_rejects_csv_bytes_modified_without_hash_update(self):
        with (self.source / "btc_daily.csv").open("a") as output:
            output.write("\n")
        with self.assertRaises(ValueError):
            refresh.replay(self.source, self.frozen)

    def test_rejects_raw_bytes_modified_without_hash_update(self):
        with (self.source / "raw/klines-000.json").open("a") as output:
            output.write(" ")
        with self.assertRaises(ValueError):
            refresh.replay(self.source, self.frozen)

    def test_rejects_changed_exchange_time_evidence(self):
        (self.source / "raw/time.json").write_bytes(encoded({"serverTime": self.server_ms + 1}))
        with self.assertRaises(ValueError):
            refresh.replay(self.source, self.frozen)

    def test_rejects_changed_frozen_prefix_hash(self):
        self.manifest["frozen_prefix_sha256"] = "0" * 64
        self.save_manifest()
        with self.assertRaises(ValueError):
            refresh.replay(self.source, self.frozen)

    def test_rejects_modified_frozen_values_even_with_updated_snapshot_hash(self):
        data = pd.read_csv(self.source / "btc_daily.csv", index_col="date")
        data.iloc[0, 0] += 1
        data.to_csv(self.source / "btc_daily.csv", float_format="%.10f")
        self.manifest["sha256"] = refresh.v3.sha256(self.source / "btc_daily.csv")
        self.save_manifest()
        with self.assertRaises((AssertionError, ValueError)):
            refresh.replay(self.source, self.frozen)

    def test_rejects_csv_not_matching_raw_even_with_updated_snapshot_hash(self):
        data = pd.read_csv(self.source / "btc_daily.csv", index_col="date")
        data.iloc[-1, 3] += 1
        data.to_csv(self.source / "btc_daily.csv", float_format="%.10f")
        self.manifest["sha256"] = refresh.v3.sha256(self.source / "btc_daily.csv")
        self.save_manifest()
        with self.assertRaises((AssertionError, ValueError)):
            refresh.replay(self.source, self.frozen)

    def test_rejects_raw_suffix_omitted_even_if_remaining_hashes_match(self):
        self.update_raw(self.rows[:1])
        with self.assertRaises(ValueError):
            refresh.replay(self.source, self.frozen)

    def test_rejects_raw_gap_even_if_hashes_updated(self):
        self.update_raw(self.rows[:1] + self.rows[2:])
        with self.assertRaises(ValueError):
            refresh.replay(self.source, self.frozen)

    def test_rejects_path_traversal_and_absolute_raw_paths(self):
        for path in ["../outside.json", str(self.root / "outside.json")]:
            self.manifest["requests"][0]["raw_file"] = path
            self.save_manifest()
            with self.subTest(path=path), self.assertRaises(ValueError):
                refresh.replay(self.source, self.frozen)

    def test_rejects_raw_file_that_could_overwrite_run_protocol(self):
        # main() copies verified raw_file entries into a fresh output directory.
        # An arbitrary relative path must not overwrite its frozen protocol.
        (self.source / "protocol.json").write_bytes(encoded(self.rows))
        self.manifest["requests"][0]["raw_file"] = "protocol.json"
        self.save_manifest()
        with self.assertRaises(ValueError):
            refresh.replay(self.source, self.frozen)

    def test_rejects_incorrect_manifest_extent(self):
        for key, value in [("rows", len(self.data) + 1),
                           ("first_closed_candle", "2026-09-14"),
                           ("last_closed_candle", "2026-09-18")]:
            original = self.manifest[key]
            self.manifest[key] = value
            self.save_manifest()
            with self.subTest(key=key), self.assertRaises(ValueError):
                refresh.replay(self.source, self.frozen)
            self.manifest[key] = original

    def test_rejects_cutoff_ahead_of_exchange_clock(self):
        self.manifest["cutoff_exclusive"] = "2026-09-21"
        self.save_manifest()
        with self.assertRaises(ValueError):
            refresh.replay(self.source, self.frozen)


class ChronologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.data = refresh.v3.load_inputs()

    def test_future_prices_and_truncation_do_not_change_past_features(self):
        cutoff = pd.Timestamp("2023-06-30")
        original = refresh.v3.make_features(self.data).loc[:cutoff]
        changed = self.data.copy()
        changed.loc[changed.index > cutoff, refresh.COLUMNS[:4]] *= 3
        pd.testing.assert_frame_equal(original, refresh.v3.make_features(changed).loc[:cutoff])
        pd.testing.assert_frame_equal(original, refresh.v3.make_features(self.data.loc[:cutoff]))

    def test_monthly_refits_never_train_on_boundary_or_future_labels(self):
        observed = []

        def inspect(train, test, horizon, config):
            cutoff = test.index.min().to_period("M").start_time
            self.assertTrue((train.label_end < cutoff).all())
            self.assertTrue(train.target.notna().all())
            observed.append((horizon, cutoff))
            return {name: np.full(len(test), 0.5)
                    for name in [*config["selectable_candidates"], *config["diagnostic_baselines"]]}

        with mock.patch.object(refresh.v3, "predict_candidates", side_effect=inspect):
            for horizon in [7, 30]:
                full = refresh.v3.make_dataset(self.data, horizon, self.config, include_unmatured=True)
                _, audit = refresh.v3.monthly_predictions(full, horizon, self.config)
                self.assertTrue(all(row["train_label_end_max"] < row["refit_date"] for row in audit))
        self.assertGreater(len(observed), 100)

    def test_selection_ignores_current_year_and_unmatured_outcomes(self):
        dates = pd.to_datetime(["2025-11-01", "2025-12-31", "2026-01-01"])
        history = pd.DataFrame({"label_end": dates + pd.Timedelta(days=7),
                                "target": [1.0, 0.0, 0.0]}, index=dates)
        for candidate in self.config["selectable_candidates"]:
            history[candidate] = 0.6 if candidate == "historical_prior" else 0.2
        cutoff = pd.Timestamp("2026-01-01")
        selected, audit = refresh.v3.select_from_past(history, cutoff, self.config)
        changed = history.copy()
        future = changed.label_end >= cutoff
        changed.loc[future, "target"] = 1 - changed.loc[future, "target"]
        for candidate in self.config["selectable_candidates"]:
            changed.loc[future, candidate] = 1 - changed.loc[future, candidate]
        self.assertEqual((selected, audit), refresh.v3.select_from_past(changed, cutoff, self.config))
        self.assertEqual(selected, "historical_prior")
        self.assertLess(audit["validation_label_end_max"], "2026-01-01")

    def test_full_v3_leakage_checks_remain_true(self):
        checks = refresh.v3.run_checks(self.data, self.config)
        self.assertTrue(checks)
        self.assertTrue(all(checks.values()), checks)


if __name__ == "__main__":
    unittest.main()
