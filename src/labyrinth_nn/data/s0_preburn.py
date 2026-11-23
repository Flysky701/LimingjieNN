"""Audit Pilot data, estimate S0 runtime costs, and update the supervision receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from labyrinth_nn.catalog.schema import FeatureCatalog
from labyrinth_nn.data import probe_simulator
from labyrinth_nn.data.sample_schema import OUTCOME_WIN, BattleSample, read_samples
from labyrinth_nn.data.sampler import QUEST_BUCKETS, HighScoreSampler, _existing_shards
from labyrinth_nn.data.sim_client import SimClient, SimulationError, SimulationTimeoutError

DEFAULT_PROFILE = Path("configs/sim_profile_round1_cn.json")
DEFAULT_SAMPLING = Path("configs/sampling_round1.json")
DEFAULT_CRITIC = Path("configs/critic_m2b_r1.json")
LOG_TAIL_BYTES = 4 * 1024 * 1024
CACHE_MISS_MARKERS = ("downloading from", "download failed", "cacheunavailable", "dependencyunavailable")
HISTORY_FIELDS = ("train_loss", "outcome_loss", "damage_loss", "damage_huber", "damage_nll",
                  "time_loss", "affinity_loss", "lr")


def _json_text(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            file.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _scratch_directory() -> Iterator[Path]:
    if sys.platform != "win32" or sys.version_info < (3, 13):
        with tempfile.TemporaryDirectory() as directory:
            yield Path(directory)
        return

    # Inherit the temp root's ACL so the Windows sandbox can use and remove the directory.
    directory = Path(tempfile.gettempdir()).resolve() / f"labyrinth-s0-{uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _read_shards(paths: Sequence[Path]) -> list[BattleSample]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Sample shard is not a file: {path}")
    return [sample for path in paths for sample in read_samples(path)]


def _nonwin_audit_rows(by_bucket: dict[str, list[BattleSample]], limit: int) -> Iterator[BattleSample]:
    buckets = (*QUEST_BUCKETS, *sorted(by_bucket.keys() - set(QUEST_BUCKETS)))
    queues = deque(iter(sorted(by_bucket[bucket], key=lambda s: s.sample_id))
                   for bucket in buckets if bucket in by_bucket)
    while queues and limit > 0:
        bucket_rows = queues.popleft()
        sample = next(bucket_rows, None)
        if sample is not None:
            yield sample
            limit -= 1
            queues.append(bucket_rows)


def verify_data(
    catalog_path: Path, sample_paths: Sequence[Path], *, db_path: Path | None = None,
    db_sha256: str | None = None, catalog_sha256: str | None = None,
    expected_shards: int | None = None, expected_unique: int | None = None,
    min_nonwin_audit: int = 200,
) -> dict[str, Any]:
    """Verify locked data and audit distinct non-win rows in bucket order."""
    if min_nonwin_audit < 0:
        raise ValueError("min_nonwin_audit must be nonnegative")
    catalog = FeatureCatalog.load(catalog_path)
    stored_catalog_hash = catalog.catalog_sha256.lower()
    computed_catalog_hash = catalog.compute_sha256().lower()
    stored_db_hash = catalog.db_sha256.lower()
    actual_db_hash = None
    db_present = db_path.is_file() if db_path is not None else None
    if db_present:
        with db_path.open("rb") as file:
            actual_db_hash = hashlib.file_digest(file, "sha256").hexdigest()

    missing_shards = []
    samples = []
    for path in sample_paths:
        if path.is_file():
            samples.extend(read_samples(path))
        else:
            missing_shards.append(str(path))
    sample_ids = {sample.sample_id for sample in samples}
    db_versions = {sample.db_version for sample in samples}
    duplicate_sample_ids = len(samples) - len(sample_ids)
    checks = {
        "db_file": db_path is None or bool(db_present),
        "db_sha256": db_sha256 is None or db_sha256.lower() == (
            actual_db_hash if db_path is not None else stored_db_hash
        ),
        "catalog_db_sha256": db_path is None or not stored_db_hash or stored_db_hash == actual_db_hash,
        "catalog_self_sha256": bool(stored_catalog_hash) and stored_catalog_hash == computed_catalog_hash,
        "catalog_sha256": catalog_sha256 is None or catalog_sha256.lower() == stored_catalog_hash,
        "sample_shard_files": not missing_shards,
        "expected_shards": expected_shards is None or len(sample_paths) == expected_shards,
        "expected_unique": expected_unique is None or len(sample_ids) == expected_unique,
        "no_duplicate_sample_ids": duplicate_sample_ids == 0,
        "samples_nonempty": bool(samples),
        "db_version": db_versions == {catalog.db_version},
    }

    corpus_nonwin_samples = corpus_nonwin_runs = corpus_clipped_runs = corpus_archived_runs = 0
    by_bucket: dict[str, list[BattleSample]] = defaultdict(list)
    seen_nonwin_ids: set[str] = set()
    for sample in samples:
        quest = catalog.quests.get(sample.quest_id)
        wave_hp = quest.wave_total_hp if quest else 0
        nonwin_runs = [run for run in sample.runs if run.outcome_class != OUTCOME_WIN]
        if nonwin_runs or sum(sample.outcome_counts[1:]) > 0:
            corpus_nonwin_samples += 1
            if sample.sample_id not in seen_nonwin_ids:
                seen_nonwin_ids.add(sample.sample_id)
                by_bucket[sample.bucket].append(sample)
        corpus_nonwin_runs += len(nonwin_runs)
        corpus_archived_runs += sum(bool(run.enemy_final_hps) for run in nonwin_runs)
        corpus_clipped_runs += sum(run.total_damage > wave_hp or run.damage_ratio >= 1.0 - 1e-9
                                   for run in nonwin_runs)

    audited_ids: list[str] = []
    audited_buckets: Counter[str] = Counter()
    archived_terminal_hp_count = proxy_only_runs = clipped_count = arith_mismatches = 0
    examples: list[dict[str, Any]] = []
    env_ids = set(catalog.environment_enemy_ids)

    def record_mismatch(sample: BattleSample, reason: str, **details: Any) -> None:
        nonlocal arith_mismatches
        arith_mismatches += 1
        if len(examples) < 20:
            examples.append({"sample_id": sample.sample_id, "quest_id": sample.quest_id,
                             "reason": reason, **details})

    for sample in _nonwin_audit_rows(by_bucket, min_nonwin_audit):
        audited_ids.append(sample.sample_id)
        audited_buckets[sample.bucket] += 1
        quest = catalog.quests.get(sample.quest_id)
        wave_hp = quest.wave_total_hp if quest else 0
        initial_hps = [catalog.enemies[eid].hp for eid in quest.enemy_ids
                       if eid not in env_ids and eid in catalog.enemies] if quest else []
        initial_hp = sum(initial_hps)
        nonwin_runs = [run for run in sample.runs if run.outcome_class != OUTCOME_WIN]
        if not nonwin_runs:
            record_mismatch(sample, "missing_nonwin_runs")
        for run in nonwin_runs:
            clipped_count += run.total_damage > wave_hp or run.damage_ratio >= 1.0 - 1e-9
            if run.enemy_final_hps:
                archived_terminal_hp_count += 1
                if len(run.enemy_final_hps) != len(initial_hps) or initial_hp <= 0:
                    record_mismatch(sample, "invalid_terminal_hp_accounting", seed=run.seed)
                    continue
                lost_hp = sum(max(0, start - final) for start, final in zip(initial_hps, run.enemy_final_hps))
                ratio = lost_hp / initial_hp
            else:
                proxy_only_runs += 1
                if wave_hp <= 0:
                    record_mismatch(sample, "invalid_wave_total_hp", seed=run.seed)
                    continue
                ratio = run.total_damage / wave_hp
            expected = max(0.0, min(1.0, ratio))
            finite_damage = math.isfinite(run.damage_ratio)
            finite_expected = math.isfinite(ratio)
            if not (finite_damage and finite_expected) or abs(run.damage_ratio - expected) > 1e-6:
                record_mismatch(sample, "damage_ratio_mismatch", seed=run.seed,
                                damage_ratio=run.damage_ratio if finite_damage else None,
                                expected=expected if finite_expected else None)

    checks["min_nonwin_audit"] = len(audited_ids) >= min_nonwin_audit
    checks["nonwin_arithmetic"] = arith_mismatches == 0
    has_terminal_hp = bool(archived_terminal_hp_count or corpus_archived_runs)
    return {
        "passed": all(checks.values()), "checks": checks,
        "db_present": db_present, "db_sha256": actual_db_hash, "catalog_db_sha256": stored_db_hash,
        "db_version": catalog.db_version, "sample_db_versions": sorted(db_versions),
        "catalog_sha256": stored_catalog_hash, "computed_catalog_sha256": computed_catalog_hash,
        "sample_shards": len(sample_paths), "missing_sample_shards": missing_shards, "sample_rows": len(samples),
        "unique_sample_ids": len(sample_ids), "duplicate_sample_ids": duplicate_sample_ids,
        "expected_shards": expected_shards, "expected_unique": expected_unique, "min_nonwin_audit": min_nonwin_audit,
        "corpus_nonwin_samples": corpus_nonwin_samples, "corpus_nonwin_runs": corpus_nonwin_runs,
        "corpus_clipped_runs": corpus_clipped_runs, "corpus_archived_terminal_hp_runs": corpus_archived_runs,
        "corpus_proxy_only_runs": corpus_nonwin_runs - corpus_archived_runs,
        "audited_distinct_nonwin_samples": len(audited_ids), "audited_nonwin_sample_ids": audited_ids,
        "audited_nonwin_by_bucket": dict(audited_buckets), "archived_terminal_hp_count": archived_terminal_hp_count,
        "proxy_only_runs": proxy_only_runs, "clipped_count": clipped_count, "arith_mismatches": arith_mismatches,
        "arith_mismatch_examples": examples,
        "damage_label_policy": ("independent_enemy_final_hp" if has_terminal_hp
                                else "truncated_aggregate_damage_proxy"),
        "damage_policy_restriction": (None if has_terminal_hp
                                      else "non_win_damage_is_truncated_aggregate_proxy_only_not_terminal_hp"),
    }


def _scan_sim_log(path: Path | None) -> dict[str, Any]:
    present = path is not None and path.is_file()
    count = 0
    examples = []
    if present:
        with path.open("rb") as file:
            start = max(0, file.seek(0, os.SEEK_END) - LOG_TAIL_BYTES)
            file.seek(start)
            tail = file.read(LOG_TAIL_BYTES).decode("utf-8", errors="replace")
        for line in tail.splitlines():
            if any(marker in line.lower() for marker in CACHE_MISS_MARKERS):
                count += 1
                if len(examples) < 20:
                    examples.append(line)
    return {"sim_log_present": present, "cache_miss_count": count, "cache_miss_examples": examples}


def run_cpu_cost_test(
    catalog_path: Path, *, profile_path: Path = DEFAULT_PROFILE, sampling_path: Path = DEFAULT_SAMPLING,
    url: str = "http://127.0.0.1:8080", probe_bench_count: int = 64, probe_out: Path | None = None,
    warmup_rows: int = 8, rows: int = 200, batch_size: int = 8, concurrency: int = 1,
    timeout_sec: float = 120.0, sim_log: Path | None = None, cpu_cc_per_hour: float | None = None,
    target_s1_rows: int = 125000, overhead_factor: float = 1.15,
    max_cc_per_100k: float = 1.15, s1_budget_cc: float = 2.5,
) -> dict[str, Any]:
    """Probe, warm up and time the simulator using a resumed sampling run."""
    for name, value, minimum in (
        ("rows", rows, 1), ("warmup_rows", warmup_rows, 0), ("batch_size", batch_size, 1),
        ("concurrency", concurrency, 1), ("probe_bench_count", probe_bench_count, 0),
        ("target_s1_rows", target_s1_rows, 0),
    ):
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    positive = {"timeout_sec": timeout_sec, "overhead_factor": overhead_factor}
    if cpu_cc_per_hour is not None:
        positive["cpu_cc_per_hour"] = cpu_cc_per_hour
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not all(math.isfinite(value) for value in (max_cc_per_100k, s1_budget_cc)):
        raise ValueError("CPU budget limits must be finite")

    catalog = FeatureCatalog.load(catalog_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    profile_override = dict(profile) | {"timeout_sec": timeout_sec, "max_retries": 0, "retry_on_timeout": False}
    config_override = dict(sampling) | {"batch_size": batch_size, "concurrency": concurrency,
                                       "chunk_size": max(1, warmup_rows + rows)}
    client = SimClient(url, catalog, profile_override)
    probe = probe_simulator.run_probe(client, bench_count=probe_bench_count)
    if probe_out is not None:
        _write_atomic(probe_out, _json_text(probe))
    report = {
        "probe": probe, "requested_rows": rows, "requested_warmup_rows": warmup_rows,
        "warmup_sec": 0.0, "warmup_rows": 0, "warmup_subruns": 0,
        "cpu_cc_per_hour": cpu_cc_per_hour, "target_s1_rows": target_s1_rows,
        "overhead_factor": overhead_factor, "max_cc_per_100k": max_cc_per_100k, "s1_budget_cc": s1_budget_cc,
    }
    if not probe["passed"]:
        return {**report, **_scan_sim_log(sim_log), "status": "probe_failed", "passed": False}

    try:
        with _scratch_directory() as directory:
            scratch_out = directory / "mini.jsonl"
            warmup_ids: set[str] = set()
            if warmup_rows:
                warmup = HighScoreSampler(catalog, profile_override, config_override).collect(
                    client, scratch_out, warmup_rows, resume=False,
                )
                report.update(warmup_sec=warmup["elapsed_sec"], warmup_rows=warmup["new_battles"],
                              warmup_subruns=warmup["new_subruns"])
                warmup_ids = {sample.sample_id for shard in _existing_shards(scratch_out).values()
                              for sample in read_samples(shard)}
            steady = HighScoreSampler(catalog, profile_override, config_override).collect(
                client, scratch_out, warmup_rows + rows, resume=True,
            )
            contexts = {sample.context_id or sample.physical_state_key()
                        for shard in _existing_shards(scratch_out).values() for sample in read_samples(shard)
                        if sample.sample_id not in warmup_ids}
    except SimulationTimeoutError as exc:
        return {**report, **_scan_sim_log(sim_log), "status": "timeout_restart_server_required",
                "passed": False, "error": str(exc)}
    except SimulationError as exc:
        return {**report, **_scan_sim_log(sim_log), "status": "gate_failed", "passed": False, "error": str(exc)}

    report.update(_scan_sim_log(sim_log))
    report.update({key: steady[key] for key in ("buckets_new", "guilds_new", "new_battles", "new_subruns",
                                                "repeat_rate", "elapsed_sec", "rows_per_sec", "subruns_per_sec")})
    all_buckets = all(steady["buckets_new"].get(bucket, 0) > 0 for bucket in QUEST_BUCKETS)
    report.update(unique_contexts_est=len(contexts), all_6_buckets_covered=all_buckets)
    report.update(avg_subruns_per_row=None, cc_per_100k_subruns=None, projected_s1_subruns=None,
                  projected_s1_wall_hours=None, projected_s1_cc_raw=None, projected_s1_cc_with_overhead=None,
                  gate_cc_per_100k_passed=None, gate_s1_budget_passed=None)
    speed, new_battles = steady["subruns_per_sec"], steady["new_battles"]
    if cpu_cc_per_hour is not None and speed > 0 and new_battles > 0:
        avg_subruns = steady["new_subruns"] / new_battles
        cc_per_100k = (100_000.0 / speed) / 3600.0 * cpu_cc_per_hour
        projected_subruns = target_s1_rows * avg_subruns
        wall_hours = (projected_subruns / speed) / 3600.0
        raw_cc = wall_hours * cpu_cc_per_hour
        total_cc = raw_cc * overhead_factor
        report.update(avg_subruns_per_row=avg_subruns, cc_per_100k_subruns=cc_per_100k,
                      projected_s1_subruns=projected_subruns, projected_s1_wall_hours=wall_hours,
                      projected_s1_cc_raw=raw_cc, projected_s1_cc_with_overhead=total_cc,
                      gate_cc_per_100k_passed=cc_per_100k <= max_cc_per_100k,
                      gate_s1_budget_passed=total_cc <= s1_budget_cc)
    passed = (new_battles == rows and report["cache_miss_count"] == 0 and (rows < 60 or all_buckets)
              and (cpu_cc_per_hour is None or bool(report["gate_cc_per_100k_passed"]
                                                  and report["gate_s1_budget_passed"])))
    return {**report, "status": "passed" if passed else "gate_failed", "passed": passed}


def run_t4_timing(
    catalog_path: Path, sample_paths: Sequence[Path], *, config_path: Path = DEFAULT_CRITIC,
    epochs: int = 2, device: str | None = None, amp: bool = False, pin_memory: bool = False,
    num_workers: int = 0, group_by: str = "context", t4_cc_per_hour: float | None = None,
    s2_target_samples: int = 150000, s2_runs: int = 11, s2_est_epochs: int = 25,
    s2_budget_cc: float = 3.5, overhead_factor: float = 1.15,
) -> dict[str, Any]:
    """Time a short Critic run and use its epoch wall time to project S2 cost."""
    for name, value in (
        ("s2_target_samples", s2_target_samples), ("s2_runs", s2_runs), ("s2_est_epochs", s2_est_epochs),
    ):
        if not math.isfinite(value) or value < 1:
            raise ValueError(f"{name} must be finite and at least 1")
    positive = {"s2_budget_cc": s2_budget_cc, "overhead_factor": overhead_factor}
    if t4_cc_per_hour is not None:
        positive["t4_cc_per_hour"] = t4_cc_per_hour
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    import torch

    from labyrinth_nn.models.critic import CriticConfig
    from labyrinth_nn.train.train_critic import train_critic

    start = perf_counter()
    samples = _read_shards(sample_paths)
    data_load_sec = perf_counter() - start
    catalog = FeatureCatalog.load(catalog_path)
    cfg = replace(CriticConfig.from_json(config_path, catalog), epochs=epochs)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    using_cuda = resolved_device.type == "cuda" and torch.cuda.is_available()
    if using_cuda:
        torch.cuda.reset_peak_memory_stats(resolved_device)

    history = []
    error = None
    with _scratch_directory() as directory:
        start = perf_counter()
        try:
            result = train_critic(cfg, catalog, samples, directory, seed=42, split_seed=42, device=device, amp=amp,
                                  num_workers=num_workers, pin_memory=pin_memory, group_by=group_by)
            history = result["history"]
        except FloatingPointError as exc:
            error = str(exc)
        train_wall_sec = perf_counter() - start

    epochs_completed = len(history)
    sec_per_epoch = train_wall_sec / max(1, epochs_completed)
    nonfinite_fields = []
    for i, row in enumerate(history, 1):
        fields = (*HISTORY_FIELDS, "val_select_nll") if row.get("val_select_nll") is not None else HISTORY_FIELDS
        for field in fields:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                nonfinite_fields.append({"epoch": i, "field": field})
    all_finite = error is None and not nonfinite_fields
    report = {
        "device": str(resolved_device), "amp": amp, "pin_memory": pin_memory, "num_workers": num_workers,
        "group_by": group_by, "sample_count": len(samples), "sample_shards": len(sample_paths),
        "data_load_sec": data_load_sec, "epochs_requested": epochs, "epochs_completed": epochs_completed,
        "train_wall_sec": train_wall_sec, "sec_per_epoch": sec_per_epoch,
        "samples_per_sec": (len(samples) * epochs_completed) / max(1e-9, train_wall_sec),
        "peak_cuda_mem_mb": torch.cuda.max_memory_allocated(resolved_device) / (1024 * 1024) if using_cuda else None,
        "all_finite": all_finite, "nonfinite_history_fields": nonfinite_fields,
        "t4_cc_per_hour": t4_cc_per_hour, "s2_target_samples": s2_target_samples, "s2_runs": s2_runs,
        "s2_est_epochs": s2_est_epochs, "s2_budget_cc": s2_budget_cc, "overhead_factor": overhead_factor,
        "s2_est_wall_sec": None, "s2_est_wall_hours": None, "s2_est_cc_raw": None,
        "s2_est_cc_with_overhead": None, "gate_s2_budget_passed": None,
    }
    if error is not None:
        return {**report, "status": "nonfinite_training", "passed": False, "error": error}
    if t4_cc_per_hour is not None:
        scale = s2_target_samples / max(1, len(samples))
        wall_sec = sec_per_epoch * scale * s2_runs * s2_est_epochs
        wall_hours = wall_sec / 3600.0
        raw_cc = wall_hours * t4_cc_per_hour
        total_cc = raw_cc * overhead_factor
        report.update(s2_est_wall_sec=wall_sec, s2_est_wall_hours=wall_hours, s2_est_cc_raw=raw_cc,
                      s2_est_cc_with_overhead=total_cc, gate_s2_budget_passed=total_cc <= s2_budget_cc)
    passed = all_finite and epochs_completed >= 1 and (t4_cc_per_hour is None or bool(report["gate_s2_budget_passed"]))
    return {**report, "status": "passed" if passed else "gate_failed", "passed": passed}


def update_status_receipt(
    receipt_path: Path, *, locked_commit: str | None = None, current_stage: str | None = None,
    stage_name: str | None = None, stage_status: str | None = None, stage_cc: float | None = None,
    observed_balance_cc: float | None = None, artifact_key: str | None = None, artifact_value: str | None = None,
    blocker: str | None = None, clear_blocker: bool = False, note: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Merge the supplied fields into the receipt and replace it atomically."""
    if (artifact_key is None) != (artifact_value is None):
        raise ValueError("artifact_key and artifact_value must be supplied together")
    if stage_name is None and (stage_status is not None or stage_cc is not None):
        raise ValueError("stage_status and stage_cc require stage_name")
    if blocker is not None and clear_blocker:
        raise ValueError("blocker and clear_blocker are mutually exclusive")

    now = datetime.now(UTC).isoformat()
    receipt = {
        "schema_version": 1, "updated_at_utc": now, "locked_commit": None, "current_stage": None,
        "stages": {}, "cc_ledger": {"per_stage": {}, "total_estimated_cc": 0.0, "hard_cap_cc": 30.0,
                                     "reserve_cc_untouched": 20.0, "last_observed_balance_cc": None},
        "artifacts": {}, "blocker": None, "notes": [],
    }
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        existing_ledger = existing.pop("cc_ledger", {})
        receipt.update(existing)
        receipt["cc_ledger"].update(existing_ledger)
    receipt["updated_at_utc"] = now
    for key, value in (("locked_commit", locked_commit), ("current_stage", current_stage), ("blocker", blocker)):
        if value is not None:
            receipt[key] = value
    if clear_blocker:
        receipt["blocker"] = None
    if artifact_key is not None:
        receipt["artifacts"][artifact_key] = artifact_value

    ledger = receipt["cc_ledger"]
    if stage_name is not None:
        stage = receipt["stages"].setdefault(stage_name, {})
        if stage_status is not None:
            stage["status"] = stage_status
        if stage_cc is not None:
            ledger["per_stage"][stage_name] = stage_cc
    ledger["total_estimated_cc"] = sum(ledger["per_stage"].values(), 0.0)
    if observed_balance_cc is not None:
        ledger["last_observed_balance_cc"] = observed_balance_cc
    if isinstance(note, str):
        note = [note]
    if note:
        receipt["notes"].extend({"at_utc": now, "text": text} for text in note)
        receipt["notes"] = receipt["notes"][-20:]

    _write_atomic(receipt_path, _json_text(receipt))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    verify = commands.add_parser("verify-data", help="verify Pilot shards and audit non-win damage")
    verify.set_defaults(run=verify_data)
    verify.add_argument("--db", dest="db_path", type=Path)
    verify.add_argument("--db-sha256")
    verify.add_argument("--catalog-sha256")
    verify.add_argument("--expected-shards", type=int)
    verify.add_argument("--expected-unique", type=int)
    verify.add_argument("--min-nonwin-audit", type=int, default=200)

    cpu = commands.add_parser("cpu-cost", help="time steady-state simulator sampling and project S1 cost")
    cpu.set_defaults(run=run_cpu_cost_test)
    cpu.add_argument("--profile", dest="profile_path", type=Path, default=DEFAULT_PROFILE)
    cpu.add_argument("--sampling", dest="sampling_path", type=Path, default=DEFAULT_SAMPLING)
    cpu.add_argument("--url", default="http://127.0.0.1:8080")
    cpu.add_argument("--probe-bench-count", type=int, default=64)
    cpu.add_argument("--probe-out", type=Path)
    cpu.add_argument("--warmup-rows", type=int, default=8)
    cpu.add_argument("--rows", type=int, default=200)
    cpu.add_argument("--batch-size", type=int, default=8)
    cpu.add_argument("--concurrency", type=int, default=1)
    cpu.add_argument("--timeout-sec", type=float, default=120.0)
    cpu.add_argument("--sim-log", type=Path)
    cpu.add_argument("--cpu-cc-per-hour", type=float)
    cpu.add_argument("--target-s1-rows", type=int, default=125000)
    cpu.add_argument("--max-cc-per-100k", type=float, default=1.15)
    cpu.add_argument("--s1-budget-cc", type=float, default=2.5)

    t4 = commands.add_parser("t4-timing", help="time Critic training and project S2 cost")
    t4.set_defaults(run=run_t4_timing)
    t4.add_argument("--config", dest="config_path", type=Path, default=DEFAULT_CRITIC)
    t4.add_argument("--epochs", type=int, default=2)
    t4.add_argument("--device")
    t4.add_argument("--amp", action="store_true")
    t4.add_argument("--pin-memory", action="store_true")
    t4.add_argument("--num-workers", type=int, default=0)
    t4.add_argument("--group-by", default="context", choices=("wave_team", "context", "wave"))
    t4.add_argument("--t4-cc-per-hour", type=float)
    t4.add_argument("--s2-target-samples", type=int, default=150000)
    t4.add_argument("--s2-runs", type=int, default=11)
    t4.add_argument("--s2-est-epochs", type=int, default=25)
    t4.add_argument("--s2-budget-cc", type=float, default=3.5)

    for command in (verify, cpu, t4):
        command.add_argument("--catalog", dest="catalog_path", type=Path, required=True)
        command.add_argument("--out", type=Path)
    for command in (verify, t4):
        command.add_argument("--samples", dest="sample_paths", type=Path, nargs="+", required=True)
    for command in (cpu, t4):
        command.add_argument("--overhead-factor", type=float, default=1.15)

    receipt = commands.add_parser("update-receipt", help="merge status and cost updates into the receipt")
    receipt.set_defaults(run=update_status_receipt)
    receipt.add_argument("--receipt", dest="receipt_path", type=Path, required=True)
    receipt.add_argument("--locked-commit")
    receipt.add_argument("--current-stage")
    receipt.add_argument("--stage-name")
    receipt.add_argument("--stage-status")
    receipt.add_argument("--stage-cc", type=float)
    receipt.add_argument("--observed-balance-cc", type=float)
    receipt.add_argument("--artifact-key")
    receipt.add_argument("--artifact-value")
    receipt.add_argument("--blocker")
    receipt.add_argument("--clear-blocker", action="store_true")
    receipt.add_argument("--note", action="append")

    options = vars(parser.parse_args(argv))
    run = options.pop("run")
    out = options.pop("out", None)
    try:
        report = run(**options)
        output = _json_text(report)
        passed = run is update_status_receipt or report["passed"]
    except (OSError, ValueError, ImportError, KeyError, TypeError, ArithmeticError, RuntimeError) as exc:
        output = _json_text({"status": "error", "passed": False, "error": str(exc)})
        passed = False
    if out is not None:
        try:
            _write_atomic(out, output)
        except OSError as exc:
            output = _json_text({"status": "error", "passed": False, "error": str(exc)})
            passed = False
    sys.stdout.write(output)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    raise SystemExit(main())
