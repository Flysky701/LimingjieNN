"""Exercise the S0 CLI against local data, a loopback simulator and CPU training."""
from __future__ import annotations

import ast
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from labyrinth_nn.data import s0_preburn
from labyrinth_nn.data.sample_schema import OUTCOME_TIMEOUT, SingleRunResult, write_samples
from labyrinth_nn.data.sampler import QUEST_BUCKETS, quest_stratum
from tests.test_catalog_and_data import NORMAL, ROOT, FixtureCase, mock_server
from tests.test_critic_and_onnx import battle, tiny_catalog


def reject_json_constant(value):
    raise ValueError(f"Unexpected JSON constant: {value}")


class CliAssertions:
    def assert_report(self, output, code, expected_code, out=None):
        self.assertEqual(code, expected_code, output)
        report = json.loads(output, parse_constant=reject_json_constant)
        if out is not None:
            encoded = out.read_bytes()
            self.assertNotIn(b"\r\n", encoded)
            self.assertTrue(encoded.endswith(b"\n"))
            self.assertEqual(encoded.decode("utf-8"), output)
        return report

    def cli(self, args, expected_code=0, *, out=None):
        argv = [str(arg) for arg in args]
        if out is not None:
            argv.extend(("--out", str(out)))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = s0_preburn.main(argv)
        return self.assert_report(stdout.getvalue(), code, expected_code, out)


class VerifyDataTests(CliAssertions, FixtureCase):
    def setUp(self):
        self.catalog_path = self.catalog.save(self.root / "特征目录.json")
        self.report_path = self.root / "audit.json"

    def nonwin(self, sample_id, bucket="normal", *, ratio=None, final_hps=(), clipped=False):
        quest = next((q for q in self.catalog.quests.values() if quest_stratum(q) == bucket),
                     self.catalog.quests[NORMAL])
        total_damage = quest.wave_total_hp * 2 if clipped else quest.wave_total_hp // 2
        if ratio is None:
            ratio = min(1.0, total_damage / quest.wave_total_hp)
        run = SingleRunResult(17, 2, OUTCOME_TIMEOUT, 5400, 90, 90.0, 0.0, total_damage, ratio, list(final_hps))
        return battle(sample_id=sample_id, db_version=self.catalog.db_version, quest_id=quest.quest_id,
                      bucket=bucket, runs=[run], outcome_counts=[0, 1, 0], mean_damage_ratio=ratio,
                      mean_win_rem_time=None)

    def audit(self, rows, *flags, expected_code=0):
        shard = write_samples(self.root / "pilot.jsonl", rows)
        return self.cli(["verify-data", "--catalog", self.catalog_path, "--samples", shard, *flags],
                        expected_code, out=self.report_path)

    def test_verify_data_round_robin_hashes_and_atomic_report(self):
        # Later input rows in a bucket sort first. Known buckets precede sorted custom buckets.
        rows = [self.nonwin("normal-z"), self.nonwin("custom-z", "z-custom"),
                self.nonwin("robot-b", "robot_ex"), self.nonwin("normal-b"),
                self.nonwin("custom-a", "a-custom"), self.nonwin("robot-a", "robot_ex", clipped=True),
                self.nonwin("normal-a")]
        first = write_samples(self.root / "pilot-1.jsonl", rows[:3])
        second = write_samples(self.root / "pilot-2.jsonl", rows[3:])
        self.report_path.write_text('{"old": true}', encoding="utf-8")
        original_replace = os.replace

        def check_replace(temporary, target):
            self.assertEqual(Path(temporary).parent, self.report_path.parent)
            self.assertEqual(target, self.report_path)
            self.assertEqual(json.loads(self.report_path.read_bytes()), {"old": True})
            original_replace(temporary, target)

        with patch.object(s0_preburn.os, "replace", side_effect=check_replace) as publish:
            report = self.cli([
                "verify-data", "--catalog", self.catalog_path, "--samples", first, second,
                "--db", self.db_path, "--db-sha256", self.catalog.db_sha256.upper(),
                "--catalog-sha256", self.catalog.catalog_sha256.upper(),
                "--expected-shards", 2, "--expected-unique", 7, "--min-nonwin-audit", 5,
            ], out=self.report_path)
        publish.assert_called_once()
        self.assertFalse(Path(publish.call_args.args[0]).exists())
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["db_sha256"], self.catalog.db_sha256)
        self.assertEqual(report["damage_label_policy"], "truncated_aggregate_damage_proxy")
        self.assertEqual(report["damage_policy_restriction"],
                         "non_win_damage_is_truncated_aggregate_proxy_only_not_terminal_hp")
        self.assertEqual(report["audited_distinct_nonwin_samples"], 5)
        self.assertEqual(report["audited_nonwin_sample_ids"],
                         ["normal-a", "robot-a", "custom-a", "custom-z", "normal-b"])
        self.assertEqual(report["audited_nonwin_by_bucket"], {"normal": 2, "robot_ex": 1, "a-custom": 1, "z-custom": 1})
        self.assertEqual(report["corpus_nonwin_samples"], 7)
        self.assertEqual(report["corpus_nonwin_runs"], 7)
        self.assertEqual(report["corpus_proxy_only_runs"], 7)
        self.assertEqual(report["corpus_clipped_runs"], 1)
        self.assertEqual(report["clipped_count"], 1)

    def test_verify_data_terminal_hp_uses_enemy_hp_and_excludes_environment(self):
        quest = self.catalog.quests[NORMAL]
        final_hps = [self.catalog.enemies[eid].hp for eid in quest.enemy_ids
                     if eid in self.catalog.enemies and eid not in self.catalog.environment_enemy_ids]
        data_catalog = replace(self.catalog, quests={**self.catalog.quests, NORMAL: replace(
            quest, enemy_ids=[*quest.enemy_ids, *self.catalog.environment_enemy_ids],
        )})
        data_catalog.save(self.catalog_path)
        row = self.nonwin("terminal", ratio=0.0, final_hps=final_hps, clipped=True)
        report = self.audit([row], "--min-nonwin-audit", 1)
        self.assertEqual(report["damage_label_policy"], "independent_enemy_final_hp")
        self.assertIsNone(report["damage_policy_restriction"])
        self.assertEqual(report["archived_terminal_hp_count"], 1)
        self.assertEqual(report["corpus_archived_terminal_hp_runs"], 1)
        self.assertEqual(report["proxy_only_runs"], 0)
        self.assertEqual(report["arith_mismatches"], 0)

        proxy = self.nonwin("first")
        report = self.audit([row, proxy], "--min-nonwin-audit", 1)
        self.assertEqual(report["archived_terminal_hp_count"], 0)
        self.assertEqual(report["corpus_archived_terminal_hp_runs"], 1)
        self.assertEqual(report["damage_label_policy"], "independent_enemy_final_hp")

    def test_verify_data_duplicate_id_fails_and_is_audited_once(self):
        row = self.nonwin("duplicate")
        report = self.audit([row, replace(row, bucket="robot_ex")], "--min-nonwin-audit", 1, expected_code=1)
        self.assertEqual(report["duplicate_sample_ids"], 1)
        self.assertEqual(report["unique_sample_ids"], 1)
        self.assertEqual(report["audited_distinct_nonwin_samples"], 1)
        self.assertFalse(report["passed"])

    def test_verify_data_checks_catalog_hash_independently_of_expected_hash(self):
        row = self.nonwin("hash")
        report = self.audit([row], "--catalog-sha256", "0" * 64, "--min-nonwin-audit", 1, expected_code=1)
        self.assertFalse(report["checks"]["catalog_sha256"])
        self.assertTrue(report["checks"]["catalog_self_sha256"])
        catalog = json.loads(self.catalog_path.read_bytes())
        catalog["catalog_sha256"] = "0" * 64
        self.catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        report = self.audit([row], "--catalog-sha256", "0" * 64, "--min-nonwin-audit", 1, expected_code=1)
        self.assertTrue(report["checks"]["catalog_sha256"])
        self.assertFalse(report["checks"]["catalog_self_sha256"])

    def test_verify_data_checks_db_hash_without_db_and_detects_wrong_db(self):
        row = self.nonwin("hash")
        flags = ["--min-nonwin-audit", 1, "--db-sha256", self.catalog.db_sha256.upper()]
        self.assertTrue(self.audit([row], *flags)["checks"]["db_sha256"])
        report = self.audit([row], "--min-nonwin-audit", 1, "--db-sha256", "0" * 64, expected_code=1)
        self.assertFalse(report["checks"]["db_sha256"])
        wrong_db = self.root / "other.db"
        wrong_db.write_bytes(b"a different database")
        report = self.audit([row], *flags, "--db", wrong_db, expected_code=1)
        self.assertFalse(report["checks"]["db_sha256"])
        self.assertFalse(report["checks"]["catalog_db_sha256"])

    def test_verify_data_fails_on_arithmetic_mismatches_and_missing_runs(self):
        row = self.nonwin("wrong-ratio")
        wrong_ratio = replace(row.runs[0], damage_ratio=row.runs[0].damage_ratio + 1.01e-6)
        rows = [replace(row, runs=[wrong_ratio]), replace(row, sample_id="counts-only", runs=[]),
                self.nonwin("wrong-terminal-length", final_hps=[1, 2, 3, 4]),
                replace(row, sample_id="unknown-quest", quest_id=-1)]
        report = self.audit(rows, "--min-nonwin-audit", 4, expected_code=1)
        self.assertEqual(report["arith_mismatches"], 4)
        self.assertEqual({item["reason"] for item in report["arith_mismatch_examples"]}, {
            "damage_ratio_mismatch", "missing_nonwin_runs", "invalid_terminal_hp_accounting", "invalid_wave_total_hp",
        })
        self.assertTrue(report["checks"]["min_nonwin_audit"])
        self.assertFalse(report["checks"]["nonwin_arithmetic"])

    def test_verify_data_nonfinite_damage_writes_strict_failure_report(self):
        report = self.audit([self.nonwin("nan", ratio=float("nan"))], "--min-nonwin-audit", 1, expected_code=1)
        self.assertEqual(report["arith_mismatches"], 1)
        self.assertIsNone(report["arith_mismatch_examples"][0]["damage_ratio"])

    def test_verify_data_default_minimum_counts_distinct_nonwins(self):
        rows = [self.nonwin(str(i)) for i in range(199)]
        rows.append(replace(rows[0], sample_id="win", runs=[], outcome_counts=[1, 0, 0]))
        report = self.audit(rows, "--expected-unique", 200, expected_code=1)
        self.assertEqual(report["min_nonwin_audit"], 200)
        self.assertEqual(report["audited_distinct_nonwin_samples"], 199)
        self.assertFalse(report["checks"]["min_nonwin_audit"])
        self.assertEqual(report["arith_mismatches"], 0)

    def test_verify_data_checks_shard_count_version_and_nonempty(self):
        report = self.audit([replace(self.nonwin("different-db"), db_version="other")],
                            "--expected-shards", 2, "--expected-unique", 2, "--min-nonwin-audit", 0,
                            expected_code=1)
        self.assertFalse(report["checks"]["expected_shards"])
        self.assertFalse(report["checks"]["expected_unique"])
        self.assertFalse(report["checks"]["db_version"])
        report = self.audit([], "--min-nonwin-audit", 0, expected_code=1)
        self.assertFalse(report["checks"]["samples_nonempty"])
        directory = self.root
        report = self.cli(["verify-data", "--catalog", self.catalog_path, "--samples", directory],
                          1, out=self.report_path)
        self.assertFalse(report["checks"]["sample_shard_files"])
        self.assertEqual(report["missing_sample_shards"], [str(directory)])


class CpuCostTests(CliAssertions, FixtureCase):
    def setUp(self):
        self.catalog_path = self.catalog.save(self.root / "feature_pack.json")
        self.profile_path = self.root / "profile.json"
        profile = json.loads((ROOT / s0_preburn.DEFAULT_PROFILE).read_bytes())
        profile["exclude_environment_enemy_ids"] = self.profile["exclude_environment_enemy_ids"]
        self.profile_path.write_text(json.dumps(profile), encoding="utf-8")
        self.sampling_path = self.root / "sampling.json"
        sampling = json.loads((ROOT / s0_preburn.DEFAULT_SAMPLING).read_bytes())
        # With this small catalog seed 42 draws all six buckets in the measured window.
        self.sampling_path.write_text(json.dumps(sampling | {"seed": 42}), encoding="utf-8")
        self.report_path = self.root / "cpu.json"

    def cost(self, url, *flags, expected_code=0):
        return self.cli(["cpu-cost", "--catalog", self.catalog_path, "--profile", self.profile_path,
                         "--sampling", self.sampling_path, "--url", url,
                         "--rows", 200, "--batch-size", 32, *flags], expected_code, out=self.report_path)

    def test_cpu_cost_real_probe_warmup_and_steady_state_budget(self):
        probe_path = self.root / "probe.json"
        with mock_server(self.catalog) as (url, requests):
            report = self.cost(url, "--probe-out", probe_path, "--cpu-cc-per-hour", 0.08)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(json.loads(probe_path.read_bytes()), report["probe"])
        self.assertNotIn(b"\r\n", probe_path.read_bytes())
        self.assertEqual(len(report["probe"]["checks"]), 16)
        self.assertTrue(all(check["passed"] for check in report["probe"]["checks"]))
        self.assertEqual(report["probe"]["benchmark"]["completed_runs"], 64)
        self.assertIn(64, map(len, requests))
        self.assertGreater(report["warmup_sec"], 0)
        self.assertEqual(report["warmup_rows"], 8)
        self.assertGreaterEqual(report["warmup_subruns"], 16)
        self.assertEqual(report["new_battles"], 200)
        self.assertTrue(report["all_6_buckets_covered"])
        self.assertEqual(set(report["buckets_new"]), set(QUEST_BUCKETS))
        self.assertEqual(sum(report["buckets_new"].values()), 200)
        self.assertEqual(sum(report["guilds_new"].values()), 200)
        self.assertGreaterEqual(report["unique_contexts_est"], 50)
        self.assertLessEqual(report["unique_contexts_est"], 200)
        self.assertGreater(report["elapsed_sec"], 0)
        self.assertGreater(report["rows_per_sec"], 0)
        self.assertEqual(report["new_subruns"], 400 + round(report["repeat_rate"] * 200) * 4)
        self.assertAlmostEqual(report["cc_per_100k_subruns"], 100000 / report["subruns_per_sec"] / 3600 * 0.08)
        self.assertAlmostEqual(report["projected_s1_subruns"], 125000 * report["new_subruns"] / 200)
        self.assertAlmostEqual(report["projected_s1_cc_with_overhead"],
                               report["projected_s1_wall_hours"] * 0.08 * 1.15)
        self.assertTrue(report["gate_cc_per_100k_passed"])
        self.assertTrue(report["gate_s1_budget_passed"])

    def test_cpu_cost_fails_when_s1_budget_is_exceeded(self):
        with mock_server(self.catalog) as (url, _):
            report = self.cost(url, "--cpu-cc-per-hour", 0.08, "--s1-budget-cc", 1e-12, expected_code=1)
        self.assertEqual(report["status"], "gate_failed")
        self.assertFalse(report["gate_s1_budget_passed"])
        self.assertTrue(report["all_6_buckets_covered"])

    def test_cpu_cost_fails_when_measured_rows_miss_a_bucket(self):
        sampling = json.loads(self.sampling_path.read_bytes())
        sampling["quest_type_weights"]["boss_5f_d5"] = 0
        self.sampling_path.write_text(json.dumps(sampling), encoding="utf-8")
        with mock_server(self.catalog) as (url, _):
            report = self.cost(url, expected_code=1)
        self.assertTrue(report["probe"]["passed"])
        self.assertEqual(report["new_battles"], 200)
        self.assertEqual(report["status"], "gate_failed")
        self.assertFalse(report["all_6_buckets_covered"])

    def test_cpu_cost_fails_on_cache_miss_and_scans_only_log_tail(self):
        log = self.root / "sim.log"
        examples = ["DoWnLoAdInG FrOm cache"] * 18 + ["Download Failed", "CacheUnavailable", "DependencyUnavailable"]
        log.write_bytes(b"Downloading from ignored\n" + b"-" * (4 * 1024 * 1024) + b"\n" +
                        "\n".join(examples).encode("utf-8"))
        with mock_server(self.catalog) as (url, _):
            report = self.cost(url, "--sim-log", log, expected_code=1)
        self.assertEqual(report["status"], "gate_failed")
        self.assertTrue(report["sim_log_present"])
        self.assertEqual(report["cache_miss_count"], len(examples))
        self.assertEqual(report["cache_miss_examples"], examples[:20])
        self.assertIsNone(report["gate_s1_budget_passed"])

    def test_cpu_cost_timeout_is_neither_retried_nor_bisected(self):
        release = threading.Event()
        log = self.root / "timeout.log"
        log.write_text("DependencyUnavailable", encoding="utf-8")
        with mock_server(self.catalog, withhold_response=release) as (url, requests):
            try:
                # Isolate Phase C so every observed HTTP request comes from sampling.
                with patch.object(s0_preburn.probe_simulator, "run_probe", return_value={"passed": True}):
                    report = self.cost(url, "--warmup-rows", 0, "--timeout-sec", 0.2, "--sim-log", log,
                                       expected_code=1)
            finally:
                release.set()
        self.assertEqual(report["status"], "timeout_restart_server_required")
        self.assertFalse(report["passed"])
        self.assertIn("timed out", report["error"].lower())
        self.assertEqual(report["cache_miss_count"], 1)
        self.assertEqual(len(requests), 1)
        self.assertGreater(len(requests[0]), 1)

    def test_cpu_cost_stops_when_real_probe_fails(self):
        release = threading.Event()
        probe_path = self.root / "failed-probe.json"
        with mock_server(self.catalog, withhold_response=release) as (url, requests):
            try:
                report = self.cost(url, "--timeout-sec", 0.2, "--probe-out", probe_path, expected_code=1)
            finally:
                release.set()
        self.assertEqual(report["status"], "probe_failed")
        self.assertEqual(report["warmup_rows"], 0)
        self.assertFalse(report["sim_log_present"])
        self.assertFalse(json.loads(probe_path.read_bytes())["passed"])
        self.assertEqual(len(requests), 1)

    def test_cpu_cost_rejects_invalid_parameters_without_probing(self):
        with patch.object(s0_preburn.probe_simulator, "run_probe") as probe:
            for flag, number in (("--rows", 0), ("--warmup-rows", -1), ("--batch-size", 0),
                                 ("--concurrency", 0), ("--timeout-sec", 0), ("--probe-bench-count", -1),
                                 ("--target-s1-rows", -1), ("--overhead-factor", 0), ("--cpu-cc-per-hour", 0)):
                with self.subTest(flag=flag):
                    report = self.cost("http://127.0.0.1:1", flag, number, expected_code=1)
                    self.assertFalse(report["passed"])
                    self.assertIn(flag[2:].replace("-", "_"), report["error"])
        probe.assert_not_called()


class LocalFilesCase(CliAssertions, unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / f"labyrinth-s0-test-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)


class T4TimingTests(LocalFilesCase):
    def setUp(self):
        super().setUp()
        self.catalog_path = tiny_catalog().save(self.root / "tiny.json")
        self.report_path = self.root / "timing.json"
        rows = [battle(sample_id=f"row-{i}", canonical_wave_sig=f"wave-{i}", context_id=f"context-{i}",
                       outcome_counts=[2, 1, 1] if i % 2 else [1, 1, 2], mean_damage_ratio=0.6 + 0.01 * i)
                for i in range(24)]
        first = write_samples(self.root / "tiny-1.jsonl", rows[:12])
        second = write_samples(self.root / "tiny-2.jsonl", rows[12:])
        self.args = ["t4-timing", "--catalog", self.catalog_path, "--config", ROOT / s0_preburn.DEFAULT_CRITIC,
                     "--samples", first, second, "--epochs", 1, "--device", "cpu", "--group-by", "context",
                     "--t4-cc-per-hour", 1.17, "--s2-target-samples", 120, "--s2-runs", 2, "--s2-est-epochs", 3]

    def timing_subprocess(self, *flags, expected_code=0):
        argv = [sys.executable, "-m", "labyrinth_nn.data.s0_preburn", *self.args, *flags,
                "--out", self.report_path]
        result = subprocess.run(
            [str(arg) for arg in argv], cwd=ROOT, capture_output=True, encoding="utf-8", timeout=90,
            env=os.environ | {"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                              "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        self.assertEqual(result.returncode, expected_code, result.stderr + result.stdout)
        return self.assert_report(result.stdout, result.returncode, expected_code, self.report_path)

    def test_t4_timing_real_cpu_training_and_budget_projection(self):
        report = self.timing_subprocess()
        self.assertTrue(report["passed"])
        self.assertTrue(report["all_finite"])
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["group_by"], "context")
        self.assertEqual(report["sample_count"], 24)
        self.assertEqual(report["sample_shards"], 2)
        self.assertEqual(report["epochs_completed"], 1)
        self.assertGreater(report["data_load_sec"], 0)
        self.assertGreater(report["sec_per_epoch"], 0)
        self.assertAlmostEqual(report["samples_per_sec"], 24 / report["train_wall_sec"])
        self.assertIsNone(report["peak_cuda_mem_mb"])
        self.assertAlmostEqual(report["s2_est_wall_sec"], report["sec_per_epoch"] * 5 * 2 * 3)
        self.assertAlmostEqual(report["s2_est_wall_hours"], report["s2_est_wall_sec"] / 3600)
        self.assertAlmostEqual(report["s2_est_cc_raw"], report["s2_est_wall_hours"] * 1.17)
        self.assertAlmostEqual(report["s2_est_cc_with_overhead"], report["s2_est_cc_raw"] * 1.15)
        self.assertTrue(report["gate_s2_budget_passed"])

    def test_t4_timing_budget_failure_still_publishes_report(self):
        self.report_path.write_text('{"old": true}', encoding="utf-8")
        report = self.timing_subprocess("--s2-budget-cc", 1e-12, expected_code=1)
        self.assertTrue(report["all_finite"])
        self.assertEqual(report["epochs_completed"], 1)
        self.assertFalse(report["gate_s2_budget_passed"])
        self.assertFalse(report["passed"])

    def test_t4_timing_rejects_invalid_cost_parameters_before_loading_samples(self):
        with patch.object(s0_preburn, "_read_shards") as read_shards:
            for name in ("overhead_factor", "t4_cc_per_hour", "s2_target_samples", "s2_runs",
                         "s2_est_epochs", "s2_budget_cc"):
                for value in (0, -1, float("nan"), float("inf"), float("-inf")):
                    with self.subTest(name=name, value=value):
                        with self.assertRaisesRegex(ValueError, rf"^{name} "):
                            s0_preburn.run_t4_timing(self.catalog_path, [], **{name: value})
        read_shards.assert_not_called()

    def test_t4_timing_floating_point_error_returns_strict_json(self):
        with patch("labyrinth_nn.train.train_critic.train_critic", side_effect=FloatingPointError("bad loss")):
            report = self.cli(self.args, 1, out=self.report_path)
        self.assertEqual(report["error"], "bad loss")
        self.assertFalse(report["all_finite"])
        self.assertFalse(report["passed"])

    def test_t4_timing_checks_reported_losses_without_emitting_nonfinite_numbers(self):
        row = dict.fromkeys(s0_preburn.HISTORY_FIELDS, 0.5) | {"lr": float("nan"), "val_select_nll": float("inf")}
        with patch("labyrinth_nn.train.train_critic.train_critic", return_value={"history": [row]}):
            report = self.cli(self.args, 1, out=self.report_path)
        self.assertEqual(report["epochs_completed"], 1)
        self.assertFalse(report["all_finite"])
        self.assertEqual([item["field"] for item in report["nonfinite_history_fields"]], ["lr", "val_select_nll"])


class ReceiptTests(LocalFilesCase):
    def setUp(self):
        super().setUp()
        self.receipt = self.root / "status" / "latest_receipt.json"

    def update(self, *flags):
        return self.cli(["update-receipt", "--receipt", self.receipt, *flags])

    def test_receipt_creates_and_merges_stages_costs_artifacts_and_blocker(self):
        initial = self.update()
        self.assertEqual(initial["schema_version"], 1)
        self.assertIsNone(initial["locked_commit"])
        self.assertEqual(initial["stages"], {})
        self.assertEqual(initial["cc_ledger"], {"per_stage": {}, "total_estimated_cc": 0.0, "hard_cap_cc": 30.0,
                                               "reserve_cc_untouched": 20.0, "last_observed_balance_cc": None})
        self.assertEqual(datetime.fromisoformat(initial["updated_at_utc"]).utcoffset(), timedelta(0))
        report = self.update("--locked-commit", "abcdef", "--current-stage", "S0", "--stage-name", "S0",
                             "--stage-status", "validated", "--stage-cc", 0.25, "--observed-balance-cc", 389.5,
                             "--artifact-key", "s0_cpu_cost", "--artifact-value", "status/成本.json",
                             "--blocker", "等待计时", "--note", "记录成本")
        self.assertEqual(report["blocker"], "等待计时")
        self.assertEqual(report["cc_ledger"]["total_estimated_cc"], 0.25)
        self.assertEqual(report["notes"][0]["text"], "记录成本")
        report = self.update("--current-stage", "S1", "--stage-name", "S1", "--stage-status", "running",
                             "--stage-cc", 1.5, "--clear-blocker", "--artifact-key", "s1_manifest",
                             "--artifact-value", "sampling.manifest.json")
        self.assertEqual(report["locked_commit"], "abcdef")
        self.assertEqual(report["current_stage"], "S1")
        self.assertEqual(report["stages"], {"S0": {"status": "validated"}, "S1": {"status": "running"}})
        self.assertEqual(report["cc_ledger"]["per_stage"], {"S0": 0.25, "S1": 1.5})
        self.assertEqual(report["cc_ledger"]["total_estimated_cc"], 1.75)
        self.assertEqual(report["cc_ledger"]["last_observed_balance_cc"], 389.5)
        self.assertEqual(report["artifacts"], {"s0_cpu_cost": "status/成本.json", "s1_manifest": "sampling.manifest.json"})
        self.assertIsNone(report["blocker"])
        report = self.update("--stage-name", "S1", "--stage-cc", 1.0)
        self.assertEqual(report["stages"]["S1"]["status"], "running")
        self.assertEqual(report["cc_ledger"]["total_estimated_cc"], 1.25)
        self.assertEqual(json.loads(self.receipt.read_bytes()), report)
        self.assertNotIn(b"\r\n", self.receipt.read_bytes())

    def test_receipt_keeps_latest_twenty_notes_with_utc_timestamps(self):
        self.update("--note", "old")
        notes = [f"记录 {i}" for i in range(22)]
        report = self.update(*(arg for note in notes for arg in ("--note", note)))
        self.assertEqual([note["text"] for note in report["notes"]], notes[-20:])
        for note in report["notes"]:
            self.assertEqual(datetime.fromisoformat(note["at_utc"]).utcoffset(), timedelta(0))
        self.assertEqual(self.update()["notes"], report["notes"])

    def test_receipt_keeps_single_string_as_one_note(self):
        report = s0_preburn.update_status_receipt(self.receipt, note="单条备注")
        self.assertEqual([note["text"] for note in report["notes"]], ["单条备注"])
        self.assertEqual(json.loads(self.receipt.read_bytes())["notes"], report["notes"])

    def test_receipt_parameter_errors_do_not_replace_existing_receipt(self):
        self.update("--locked-commit", "stay")
        original = self.receipt.read_bytes()
        for flags in (("--artifact-key", "key"), ("--artifact-value", "path"), ("--stage-status", "ready"),
                      ("--stage-cc", 0.1), ("--blocker", "waiting", "--clear-blocker"),
                      ("--observed-balance-cc", "nan")):
            with self.subTest(flags=flags):
                report = self.cli(["update-receipt", "--receipt", self.receipt, *flags], 1)
                self.assertFalse(report["passed"])
                self.assertEqual(self.receipt.read_bytes(), original)


class RepositoryHygieneTests(unittest.TestCase):
    def test_shell_scripts_and_notebooks_use_lf(self):
        paths = sorted((ROOT / "scripts").glob("*.sh")) + sorted((ROOT / "notebooks").glob("*.ipynb"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn(b"\r\n", path.read_bytes())

    def test_colab_pipeline_code_cells_are_python(self):
        path = ROOT / "notebooks" / "colab_labyrinth_pipeline.ipynb"
        notebook = json.loads(path.read_bytes())
        code_cells = [(i, cell) for i, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        for i, cell in code_cells:
            with self.subTest(cell=i):
                ast.parse("".join(cell["source"]), filename=f"{path.name} cell {i}")
