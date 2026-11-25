"""Crash-safe, manifest-backed, zero-bisection executor for S3 active mining and S5 sealed panels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from labyrinth_nn.active_mining.identity import (
    S2HoldoutRegistry,
    read_jsonl,
    semantic_context_sha256,
    sha256_file,
    sha256_json,
    worker_for_context,
    worker_index_from_id,
    write_json_atomic,
)
from labyrinth_nn.active_mining.s3_planner import verify_s3_plan
from labyrinth_nn.active_mining.s5_reservation import S5Denylist
from labyrinth_nn.catalog.schema import FeatureCatalog
from labyrinth_nn.data.sample_schema import BattleSample, SingleRunResult, aggregate_runs, read_samples
from labyrinth_nn.data.sampler import (
    _existing_shards,
    _file_sha256,
    _shard_entry,
    _shard_path,
    _sync_file_atomic,
    _write_shard,
)
from labyrinth_nn.data.sim_client import SimClient, SimulationConfig, SimulationError, SimulationTimeoutError


def _append_journal_record(journal_path: Path, record: Mapping[str, Any]) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    with journal_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _append_failed_records(failed_path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with failed_path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_manifest_atomic(
    manifest_path: Path,
    entries: list[dict[str, Any]],
    total_unique: int,
    *,
    round_id: str,
    catalog: FeatureCatalog,
    extra: Mapping[str, Any] | None = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "shards": entries,
        "total_unique_battles": total_unique,
        "round_id": round_id,
        "db_version": catalog.db_version,
        "catalog_sha256": catalog.catalog_sha256 or catalog.compute_sha256(),
    }
    if extra:
        payload.update(extra)
    write_json_atomic(manifest_path, payload, indent=2)


def _sample_from_plan_row(row: Mapping[str, Any], runs: list[SingleRunResult] | None = None) -> BattleSample:
    payload = row["payload"]
    run_list = list(runs or [])
    counts, mean_dmg, mean_time = aggregate_runs(run_list)
    return BattleSample(
        sample_id=str(row["sample_id"]),
        round_id=str(row["round_id"]),
        source="sim",
        db_version=str(payload["db_version"]),
        quest_id=int(payload["quest_id"]),
        canonical_wave_sig=str(payload["canonical_wave_sig"]),
        bucket=str(payload["bucket"]),
        area=int(payload["area"]),
        difficulty=int(payload["difficulty"]),
        guild_id=int(payload["guild_id"]),
        team_unit_ids=[int(u) for u in payload["team_unit_ids"]],
        owned_unit_ids=[int(u) for u in payload["owned_unit_ids"]],
        owned_unit_count=int(payload["owned_unit_count"]),
        owned_role_counts=[int(x) for x in payload["owned_role_counts"]],
        owned_talent_counts=[int(x) for x in payload["owned_talent_counts"]],
        relic_ids=[int(r) for r in payload["relic_ids"]],
        mark_stacks=[int(m) for m in payload["mark_stacks"]],
        alpha_rupee=None if payload.get("alpha_rupee") is None else int(payload["alpha_rupee"]),
        revive_available=bool(payload["revive_available"]),
        runs=run_list,
        outcome_counts=counts,
        mean_damage_ratio=mean_dmg,
        mean_win_rem_time=mean_time,
        context_id=str(payload.get("context_id") or row["semantic_context_sha256"]),
    )


def _validate_run_results(plan_row: Mapping[str, Any], runs: Sequence[SingleRunResult]) -> None:
    expected_seeds = [int(s) for s in plan_row["run_seeds"]]
    if len(runs) != int(plan_row["repeat_count"]) or len(runs) != len(expected_seeds):
        raise ValueError(
            f"sample {plan_row['sample_id']}: expected {plan_row['repeat_count']} runs, got {len(runs)}"
        )
    actual_seeds = [int(r.seed) for r in runs]
    if actual_seeds != expected_seeds:
        raise ValueError(
            f"sample {plan_row['sample_id']}: seed echo/order mismatch {actual_seeds} != {expected_seeds}"
        )
    for idx, run in enumerate(runs):
        if run.result not in (0, 1, 2) or run.outcome_class not in (0, 1, 2):
            raise ValueError(f"sample {plan_row['sample_id']} run {idx}: invalid result/outcome")
        if run.battle_frames < 0 or run.limit_time <= 0 or run.total_damage < 0:
            raise ValueError(f"sample {plan_row['sample_id']} run {idx}: invalid frame/limit/damage")
        for name, val in (
            ("elapsed_sec", run.elapsed_sec),
            ("rem_time_ratio", run.rem_time_ratio),
            ("damage_ratio", run.damage_ratio),
        ):
            if not math.isfinite(val):
                raise ValueError(f"sample {plan_row['sample_id']} run {idx}: non-finite {name}")
        if not (0.0 <= run.rem_time_ratio <= 1.0 and 0.0 <= run.damage_ratio <= 1.0):
            raise ValueError(f"sample {plan_row['sample_id']} run {idx}: out-of-range ratio")


def _validate_manifest_and_shards(
    out_path: Path,
    manifest_path: Path,
    *,
    expected_round_id: str,
    catalog: FeatureCatalog,
) -> tuple[dict[int, Path], list[dict[str, Any]], dict[str, BattleSample]]:
    shards = _existing_shards(out_path)
    catalog_sha = catalog.catalog_sha256 or catalog.compute_sha256()
    if not shards and not manifest_path.is_file():
        return {}, [], {}
    if not manifest_path.is_file():
        raise ValueError(f"output shards exist without manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("round_id") != expected_round_id:
        raise ValueError(f"manifest round_id mismatch: {manifest.get('round_id')} != {expected_round_id}")
    if manifest.get("db_version") != catalog.db_version:
        raise ValueError(f"manifest db_version mismatch: {manifest.get('db_version')} != {catalog.db_version}")
    if manifest.get("catalog_sha256") != catalog_sha:
        raise ValueError("manifest catalog_sha256 mismatch")

    entries = list(manifest.get("shards") or [])
    shard_by_name = {p.name: p for p in shards.values()}
    if {e["filename"] for e in entries} != set(shard_by_name):
        raise ValueError(f"manifest shard filenames disagree with existing shards for {out_path}")

    committed: dict[str, BattleSample] = {}
    for entry in entries:
        shard_path = shard_by_name[entry["filename"]]
        actual_sha = _file_sha256(shard_path)
        if actual_sha.lower() != str(entry["sha256"]).lower():
            raise ValueError(f"shard sha256 mismatch for {shard_path}")
        rows = read_samples(shard_path)
        if len(rows) != int(entry["row_count"]):
            raise ValueError(f"shard row_count mismatch for {shard_path}: {len(rows)} != {entry['row_count']}")
        for s in rows:
            if s.round_id != expected_round_id or s.db_version != catalog.db_version:
                raise ValueError(f"sample {s.sample_id} in {shard_path} belongs to a different round/db")
            if s.sample_id in committed:
                raise ValueError(f"duplicate committed sample_id {s.sample_id} in {shard_path}")
            committed[s.sample_id] = s
    if len(committed) != int(manifest.get("total_unique_battles", -1)):
        raise ValueError("manifest total_unique_battles mismatch")
    return shards, entries, committed


def _verify_s5_rows(
    sidecar: Mapping[str, Any],
    worker_id: str,
    rows: Sequence[Mapping[str, Any]],
    catalog: FeatureCatalog,
    s5_denylist: S5Denylist | None,
) -> None:
    if s5_denylist is not None and sidecar.get("s5_denylist_sha256") != s5_denylist.denylist_sha256:
        raise ValueError("sidecar s5_denylist_sha256 mismatch")
    excluded_units = frozenset(int(u) for u in (sidecar.get("excluded_sim_team_units") or ()))
    num_workers = int(sidecar["num_workers"])
    w_idx = worker_index_from_id(worker_id, num_workers)
    seen_phys: set[str] = set()
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows):
        if row.get("worker_id") != worker_id:
            raise ValueError(f"row {idx} worker_id mismatch")
        payload = row["payload"]
        if excluded_units and (excluded_units & {int(u) for u in payload["team_unit_ids"]}):
            raise ValueError(f"S5 plan row {idx} contains excluded simulator team unit")
        ctx_sha = semantic_context_sha256(payload)
        if row.get("semantic_context_sha256") != ctx_sha or worker_for_context(ctx_sha, num_workers) != w_idx:
            raise ValueError(f"row {idx} context hash or worker assignment mismatch")
        sample = _sample_from_plan_row(row)
        phys = sample.physical_state_key()
        if row.get("physical_state_key") != phys or phys in seen_phys or sample.sample_id in seen_ids:
            raise ValueError(f"row {idx} invalid or duplicate physical_state_key / sample_id")
        if s5_denylist is not None and not s5_denylist.is_denied(payload):
            raise ValueError(f"S5 plan row {idx} missing from S5 denylist")
        seen_phys.add(phys)
        seen_ids.add(sample.sample_id)
        seeds = row.get("run_seeds") or []
        if len(seeds) != int(row["repeat_count"]) or len(set(seeds)) != len(seeds):
            raise ValueError(f"row {idx} invalid run_seeds")


def _simulate_batch_per_battle(
    client: SimClient,
    sim_configs: list[SimulationConfig],
    catalog: FeatureCatalog,
) -> tuple[dict[int, list[SingleRunResult]], dict[int, str]]:
    """Run one HTTP POST batch without retry/bisection and isolate HTTP 200 per-battle errors."""
    if "simulate_batch_single_runs" in getattr(client, "__dict__", {}):
        groups = client.simulate_batch_single_runs(sim_configs)
        return {idx: list(runs) for idx, runs in enumerate(groups)}, {}

    excluded = client.environment | set(catalog.environment_enemy_ids)
    excluded_unit_ids = excluded | {
        catalog.enemies[eid].unit_id for eid in excluded if eid in catalog.enemies
    }
    requests: list[dict[str, Any]] = []
    destinations: list[tuple[int, Any, dict[str, Any]]] = []
    for index, config in enumerate(sim_configs):
        repeat_count = config.runs
        run_seeds = config.run_seeds
        if repeat_count < 0 or (run_seeds is not None and len(run_seeds) != repeat_count):
            raise ValueError("runs must be nonnegative and match the number of explicit seeds")
        if not repeat_count:
            continue
        payload = client.build_request(config, catalog=catalog)
        request = dict(payload)
        enemy_ids = [eid for value in request["GuildEnemyList"] if (eid := int(value)) and eid not in excluded]
        if not enemy_ids:
            raise ValueError("a simulation requires at least one non-environment enemy")
        request.update(RepeatCount=1, GuildEnemyList=enemy_ids)
        for run_index in range(repeat_count):
            run_request = request
            if run_seeds is not None:
                run_request = request | {
                    "Settings": request["Settings"] | {
                        "UseFixedRandomSeed": True,
                        "RandomSeed": int(run_seeds[run_index]),
                    }
                }
            requests.append(run_request)
            destinations.append((index, config.quest, run_request))

    if not requests:
        return {idx: [] for idx in range(len(sim_configs))}, {}

    raw_responses = client._post(requests)
    if len(raw_responses) != len(requests):
        raise SimulationError(f"{client.url}: expected {len(requests)} response items, got {len(raw_responses)}")

    per_battle_items: dict[int, list[tuple[int, dict[str, Any], Any, dict[str, Any]]]] = defaultdict(list)
    for global_idx, (raw, (battle_idx, quest, req)) in enumerate(zip(raw_responses, destinations, strict=True)):
        per_battle_items[battle_idx].append((global_idx, raw, quest, req))

    ok_groups: dict[int, list[SingleRunResult]] = {}
    failed_errors: dict[int, str] = {}
    for battle_idx, config in enumerate(sim_configs):
        if config.runs == 0:
            ok_groups[battle_idx] = []
            continue
        items = per_battle_items[battle_idx]
        parsed_runs: list[SingleRunResult] = []
        last_global = -1
        last_quest: Any = config.quest.quest_id if config.quest is not None else "unknown"
        try:
            for global_idx, raw, quest, req in items:
                last_global = global_idx
                last_quest = req.get("LabyrinthQuestId", last_quest)
                parsed_runs.append(client._parse(raw, req, quest, catalog, excluded, excluded_unit_ids))
        except (AttributeError, KeyError, TypeError, ValueError, SimulationError) as exc:
            failed_errors[battle_idx] = f"response {last_global} (quest {last_quest}): {exc}"
        else:
            ok_groups[battle_idx] = parsed_runs
    return ok_groups, failed_errors


def run_worker_plan(
    *,
    sidecar_path: str | Path,
    worker_plan_path: str | Path,
    worker_id: str,
    catalog: FeatureCatalog,
    profile: Mapping[str, Any],
    client: SimClient,
    out_path: str | Path,
    reserve_plan_path: str | Path | None = None,
    allow_reserve: bool = False,
    s2_registry: S2HoldoutRegistry | None = None,
    s5_denylist: S5Denylist | None = None,
    existing_physical_states: Iterable[str] = (),
    target_battles: int | None = None,
    chunk_size: int = 500,
    max_subruns_per_post: int = 32,
    max_seconds: float | None = None,
    max_submitted_subruns: int | None = None,
    sync_dir: str | Path | None = None,
    resume: bool = False,
    progress_stderr: bool = False,
    before_post_hook: Callable[[dict[str, Any]], None] | None = None,
    after_parse_hook: Callable[[dict[str, Any], list[BattleSample]], None] | None = None,
    after_shard_hook: Callable[[Path, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one worker's manifest with durable pre-journal, strict seed verification, and no bisection."""
    if chunk_size < 1 or max_subruns_per_post < 1:
        raise ValueError("chunk_size and max_subruns_per_post must be positive")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if max_submitted_subruns is not None and max_submitted_subruns < 0:
        raise ValueError("max_submitted_subruns must be nonnegative")
    if getattr(client, "max_retries", 0) != 0 or getattr(client, "retry_on_timeout", False):
        raise ValueError("manifest executor requires client with max_retries=0 and retry_on_timeout=False")

    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    workers_meta = sidecar.get("workers") or {}
    if worker_id not in workers_meta:
        raise ValueError(f"worker_id {worker_id!r} missing from sidecar")
    w_meta = workers_meta[worker_id]
    expected_round_id = str(w_meta["round_id"])

    plan_path = Path(worker_plan_path)
    actual_plan_sha = sha256_file(plan_path)
    if actual_plan_sha.lower() != str(w_meta["plan_sha256"]).lower():
        raise ValueError(f"worker plan sha256 mismatch: {actual_plan_sha} != {w_meta['plan_sha256']}")
    plan_rows = read_jsonl(plan_path)
    if len(plan_rows) != int(w_meta["row_count"]):
        raise ValueError(f"worker plan row count {len(plan_rows)} != {w_meta['row_count']}")

    reserve_rows: list[dict[str, Any]] = []
    if allow_reserve:
        if reserve_plan_path is None or not w_meta.get("reserve_sha256"):
            raise ValueError("allow_reserve requires a valid reserve_plan_path pinned in sidecar")
        res_path = Path(reserve_plan_path)
        actual_res_sha = sha256_file(res_path)
        if actual_res_sha.lower() != str(w_meta["reserve_sha256"]).lower():
            raise ValueError(f"reserve plan sha256 mismatch: {actual_res_sha} != {w_meta['reserve_sha256']}")
        reserve_rows = read_jsonl(res_path)
        if len(reserve_rows) != int(w_meta.get("reserve_count", len(reserve_rows))):
            raise ValueError("reserve plan row count mismatch")

    plan_kind = sidecar.get("plan_kind", "s3_active_mining")
    if plan_kind == "s5_sealed_panel":
        _verify_s5_rows(sidecar, worker_id, plan_rows, catalog, s5_denylist)
    else:
        verify_s3_plan(
            sidecar=sidecar,
            worker_plans={worker_id: plan_rows},
            worker_reserves={worker_id: reserve_rows} if allow_reserve else None,
            catalog=catalog,
            s2_registry=s2_registry,
            s5_denylist=s5_denylist,
            existing_physical_states=existing_physical_states,
        )

    out = Path(out_path)
    if out.suffix not in {".jsonl", ".parquet"}:
        raise ValueError("--out must end in .jsonl or .parquet")
    manifest_path = out.with_name(f"{out.stem}.manifest.json")
    failed_path = out.with_name(f"{out.stem}.failed.jsonl")
    journal_path = out.with_name(f"{out.stem}.journal.jsonl")
    sync_root = Path(sync_dir) if sync_dir is not None else None

    shards, entries, committed_map = _validate_manifest_and_shards(
        out, manifest_path, expected_round_id=expected_round_id, catalog=catalog
    )
    if shards and not resume:
        raise FileExistsError(f"output already exists, use --resume: {out}")

    # Reconcile durable pre-journal against verified committed shards.
    journal_records = read_jsonl(journal_path)
    prepared_by_req: dict[str, dict[str, Any]] = {}
    resolved_sids_by_req: dict[str, set[str]] = defaultdict(set)
    attempted_sample_ids: set[str] = set(committed_map.keys())
    prior_submitted_subruns = 0
    prior_indeterminate_ids: set[str] = set()
    prior_quarantined_ids: set[str] = set()

    for entry in journal_records:
        if entry.get("worker_id") != worker_id or entry.get("plan_sha256") != actual_plan_sha:
            raise ValueError(
                f"conflicting journal metadata in {journal_path}: "
                f"worker={entry.get('worker_id')} plan_sha={entry.get('plan_sha256')}"
            )
        ev = entry.get("event")
        req_id = str(entry["request_id"])
        if ev == "prepared":
            prepared_by_req[req_id] = entry
            prior_submitted_subruns += int(entry["intended_subruns"])
            attempted_sample_ids.update(entry["sample_ids"])
        elif ev in ("committed", "reconciled_committed"):
            resolved_sids_by_req[req_id].update(str(sid) for sid in (entry.get("sample_ids") or ()))
        elif ev in ("timeout_indeterminate", "reconciled_indeterminate"):
            sids = [str(sid) for sid in (entry.get("sample_ids") or ())]
            resolved_sids_by_req[req_id].update(sids)
            prior_indeterminate_ids.update(sids)
        elif ev == "quarantined":
            sids = [str(sid) for sid in (entry.get("sample_ids") or ())]
            resolved_sids_by_req[req_id].update(sids)
            prior_quarantined_ids.update(sids)

    if failed_path.is_file():
        for failed_row in read_jsonl(failed_path):
            sid = failed_row.get("sample_id")
            if sid:
                attempted_sample_ids.add(str(sid))
                prior_quarantined_ids.add(str(sid))

    # Resolve any open prepared records from a crash (at sample granularity for partial-batch safety).
    for req_id, prep in prepared_by_req.items():
        resolved_sids = resolved_sids_by_req.get(req_id, set())
        req_sids = [str(sid) for sid in prep["sample_ids"] if str(sid) not in resolved_sids]
        if not req_sids:
            continue
        expected_seeds_map = prep["run_seeds_by_sample"]
        committed_now: list[str] = []
        indet_now: list[str] = []
        for sid in req_sids:
            if (
                sid in committed_map
                and [int(r.seed) for r in committed_map[sid].runs] == [int(x) for x in expected_seeds_map[sid]]
            ):
                committed_now.append(sid)
            else:
                indet_now.append(sid)
        if committed_now:
            _append_journal_record(
                journal_path,
                {
                    "event": "reconciled_committed",
                    "request_id": req_id,
                    "plan_sha256": actual_plan_sha,
                    "worker_id": worker_id,
                    "sample_ids": committed_now,
                    "intended_subruns": sum(len(expected_seeds_map[sid]) for sid in committed_now),
                },
            )
        if indet_now:
            prior_indeterminate_ids.update(indet_now)
            _append_journal_record(
                journal_path,
                {
                    "event": "reconciled_indeterminate",
                    "request_id": req_id,
                    "plan_sha256": actual_plan_sha,
                    "worker_id": worker_id,
                    "sample_ids": indet_now,
                    "intended_subruns": sum(len(expected_seeds_map[sid]) for sid in indet_now),
                },
            )

    existing_battles = len(committed_map)
    existing_subruns = sum(len(s.runs) for s in committed_map.values())
    goal_battles = len(plan_rows) if target_battles is None else int(target_battles)
    if goal_battles < 0:
        raise ValueError("target_battles must be nonnegative")

    started = time.monotonic()
    shard_index = max(shards, default=-1) + 1
    synced_shards = 0
    pending: list[BattleSample] = []
    pending_journals: list[dict[str, Any]] = []
    flushed_sample_ids: set[str] = set(committed_map.keys())
    written_battles = 0
    accepted_subruns = 0
    returned_subruns = 0
    session_submitted_subruns = 0
    quarantined_battles = 0
    reserve_used_battles = 0
    stopped_by_time_limit = False
    stopped_by_budget = False
    exhausted = False
    buckets_new: Counter[str] = Counter()

    def sync_artifacts() -> None:
        nonlocal synced_shards
        if sync_root is None:
            return
        while synced_shards < len(entries):
            shard_file = out.with_name(entries[synced_shards]["filename"])
            _sync_file_atomic(shard_file, sync_root)
            synced_shards += 1
        if manifest_path.is_file():
            _sync_file_atomic(manifest_path, sync_root)
        if failed_path.is_file():
            _sync_file_atomic(failed_path, sync_root)
        if journal_path.is_file():
            _sync_file_atomic(journal_path, sync_root)

    def flush(count: int) -> None:
        nonlocal shard_index, written_battles
        if count <= 0:
            return
        chunk = pending[:count]
        shard = _shard_path(out, shard_index)
        _write_shard(shard, chunk)
        shard_entry = _shard_entry(shard, count)
        entries.append(shard_entry)
        del pending[:count]
        written_battles += count
        shard_index += 1
        _write_manifest_atomic(
            manifest_path,
            entries,
            existing_battles + written_battles,
            round_id=expected_round_id,
            catalog=catalog,
            extra={
                "worker_id": worker_id,
                "plan_sha256": actual_plan_sha,
                "submitted_subruns": prior_submitted_subruns + session_submitted_subruns,
                "accepted_subruns": existing_subruns + accepted_subruns,
            },
        )
        sync_artifacts()
        if after_shard_hook is not None:
            after_shard_hook(shard, shard_entry)
        # Only mark requests committed after verified shard + manifest publish succeeds.
        flushed_sample_ids.update(s.sample_id for s in chunk)
        remaining_pending_journals: list[dict[str, Any]] = []
        for pj in pending_journals:
            if set(pj["sample_ids"]) <= flushed_sample_ids:
                _append_journal_record(
                    journal_path,
                    {
                        "event": "committed",
                        "request_id": pj["request_id"],
                        "plan_sha256": actual_plan_sha,
                        "worker_id": worker_id,
                        "sample_ids": pj["sample_ids"],
                        "shard_filename": shard_entry["filename"],
                        "shard_sha256": shard_entry["sha256"],
                        "intended_subruns": pj["intended_subruns"],
                    },
                )
            else:
                remaining_pending_journals.append(pj)
        pending_journals[:] = remaining_pending_journals
        if sync_root is not None and journal_path.is_file():
            _sync_file_atomic(journal_path, sync_root)
        if progress_stderr:
            print(
                f"[{worker_id}] Collected {existing_battles + written_battles}/{goal_battles} "
                f"({written_battles} new), wrote {shard.name}",
                file=sys.stderr,
                flush=True,
            )

    def sync_journal_only() -> None:
        if sync_root is not None:
            if failed_path.is_file():
                _sync_file_atomic(failed_path, sync_root)
            if journal_path.is_file():
                _sync_file_atomic(journal_path, sync_root)

    # Candidate queue: primary rows first, then reserve rows if authorized.
    primary_queue = [r for r in plan_rows if str(r["sample_id"]) not in attempted_sample_ids]
    reserve_queue = (
        [r for r in reserve_rows if str(r["sample_id"]) not in attempted_sample_ids]
        if allow_reserve
        else []
    )
    queue = primary_queue + reserve_queue
    q_idx = 0
    attempt_seq = len(prepared_by_req)
    consecutive_batch_errors = 0

    try:
        while existing_battles + written_battles + len(pending) < goal_battles:
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                stopped_by_time_limit = True
                break
            if q_idx >= len(queue):
                exhausted = True
                break

            needed_rows = goal_battles - (existing_battles + written_battles + len(pending))
            batch: list[dict[str, Any]] = []
            batch_subruns = 0
            while q_idx < len(queue) and len(batch) < needed_rows:
                cand = queue[q_idx]
                rep = int(cand["repeat_count"])
                if batch and batch_subruns + rep > max_subruns_per_post:
                    break
                if (
                    max_submitted_subruns is not None
                    and prior_submitted_subruns + session_submitted_subruns + batch_subruns + rep
                    > max_submitted_subruns
                ):
                    break
                batch.append(cand)
                batch_subruns += rep
                q_idx += 1

            if not batch:
                stopped_by_budget = True
                break

            attempt_seq += 1
            request_id = f"{worker_id}:{actual_plan_sha[:12]}:{attempt_seq}"
            sample_ids = [str(r["sample_id"]) for r in batch]
            run_seeds_by_sample = {str(r["sample_id"]): [int(s) for s in r["run_seeds"]] for r in batch}
            payload_sha256 = sha256_json(
                [
                    {
                        "sample_id": str(r["sample_id"]),
                        "run_seeds": [int(s) for s in r["run_seeds"]],
                        "payload": r["payload"],
                    }
                    for r in batch
                ]
            )
            prep_record = {
                "event": "prepared",
                "request_id": request_id,
                "plan_sha256": actual_plan_sha,
                "worker_id": worker_id,
                "sample_ids": sample_ids,
                "run_seeds_by_sample": run_seeds_by_sample,
                "payload_sha256": payload_sha256,
                "intended_subruns": batch_subruns,
                "attempt": attempt_seq,
                "is_reserve": any(bool(r.get("is_reserve")) for r in batch),
            }
            _append_journal_record(journal_path, prep_record)
            session_submitted_subruns += batch_subruns
            attempted_sample_ids.update(sample_ids)
            reserve_used_battles += sum(1 for r in batch if bool(r.get("is_reserve")))

            if before_post_hook is not None:
                before_post_hook(prep_record)

            sim_configs = [
                SimulationConfig(
                    quest=catalog.quests[int(r["payload"]["quest_id"])],
                    team_unit_ids=r["payload"]["team_unit_ids"],
                    owned_unit_ids=r["payload"]["owned_unit_ids"],
                    relic_ids=r["payload"]["relic_ids"],
                    alpha_rupee=r["payload"]["alpha_rupee"],
                    difficulty=int(r["payload"]["difficulty"]),
                    runs=int(r["repeat_count"]),
                    run_seeds=[int(s) for s in r["run_seeds"]],
                )
                for r in batch
            ]

            try:
                ok_groups, failed_errors = _simulate_batch_per_battle(client, sim_configs, catalog)
            except SimulationTimeoutError:
                _append_journal_record(
                    journal_path,
                    {
                        "event": "timeout_indeterminate",
                        "request_id": request_id,
                        "plan_sha256": actual_plan_sha,
                        "worker_id": worker_id,
                        "sample_ids": sample_ids,
                        "intended_subruns": batch_subruns,
                    },
                )
                if pending:
                    flush(len(pending))
                else:
                    sync_journal_only()
                raise
            except (SimulationError, KeyError, TypeError, ValueError) as exc:
                err_text = str(exc)
                is_conn_refused = any(
                    tok in err_text for tok in ("Connection refused", "Errno 111", "WinError 10061")
                )
                is_conn_drop = any(
                    tok in err_text for tok in ("Remote end closed connection", "Connection reset", "Broken pipe")
                )
                if is_conn_refused:
                    # Server is dead before accepting this batch; fail fast without poisoning unrun plan rows.
                    if pending:
                        flush(len(pending))
                    else:
                        sync_journal_only()
                    raise

                # Ambiguous disconnect or non-timeout transport / HTTP / payload error: quarantine this batch cleanly.
                _append_failed_records(
                    failed_path,
                    [
                        {
                            "sample_id": r["sample_id"],
                            "physical_state_key": r["physical_state_key"],
                            "request_id": request_id,
                            "reason": "quarantined_ambiguous_batch",
                            "error": err_text,
                        }
                        for r in batch
                    ],
                )
                _append_journal_record(
                    journal_path,
                    {
                        "event": "quarantined",
                        "request_id": request_id,
                        "plan_sha256": actual_plan_sha,
                        "worker_id": worker_id,
                        "sample_ids": sample_ids,
                        "intended_subruns": batch_subruns,
                        "error": err_text,
                    },
                )
                quarantined_battles += len(batch)
                if is_conn_drop:
                    if pending:
                        flush(len(pending))
                    else:
                        sync_journal_only()
                    raise
                consecutive_batch_errors += 1
                if consecutive_batch_errors >= 3:
                    if pending:
                        flush(len(pending))
                    else:
                        sync_journal_only()
                    raise RuntimeError(
                        "aborting worker: 3 consecutive batch transport/simulator failures"
                    ) from exc
                sync_journal_only()
                continue

            consecutive_batch_errors = 0
            if len(ok_groups) + len(failed_errors) != len(batch):
                raise ValueError(
                    f"battle group count {len(ok_groups) + len(failed_errors)} != batch size {len(batch)}"
                )
            returned_subruns += sum(len(g) for g in ok_groups.values())
            ok_plan_rows: list[dict[str, Any]] = []
            built_samples: list[BattleSample] = []
            failed_entries: list[tuple[dict[str, Any], str, str]] = []
            for idx, plan_row in enumerate(batch):
                if idx in failed_errors:
                    failed_entries.append((plan_row, "quarantined_battle_response_error", failed_errors[idx]))
                    continue
                runs = ok_groups[idx]
                try:
                    _validate_run_results(plan_row, runs)
                    built_samples.append(_sample_from_plan_row(plan_row, list(runs)))
                    ok_plan_rows.append(plan_row)
                except (ValueError, TypeError, KeyError) as exc:
                    failed_entries.append((plan_row, "quarantined_malformed_or_seed_mismatch", str(exc)))

            if failed_entries:
                _append_failed_records(
                    failed_path,
                    [
                        {
                            "sample_id": r["sample_id"],
                            "physical_state_key": r["physical_state_key"],
                            "request_id": request_id,
                            "reason": reason,
                            "error": err,
                        }
                        for r, reason, err in failed_entries
                    ],
                )
                _append_journal_record(
                    journal_path,
                    {
                        "event": "quarantined",
                        "request_id": request_id,
                        "plan_sha256": actual_plan_sha,
                        "worker_id": worker_id,
                        "sample_ids": [str(r["sample_id"]) for r, _, _ in failed_entries],
                        "intended_subruns": sum(int(r["repeat_count"]) for r, _, _ in failed_entries),
                        "errors_by_sample": {str(r["sample_id"]): err for r, _, err in failed_entries},
                        "error": failed_entries[0][2],
                    },
                )
                quarantined_battles += len(failed_entries)

            if built_samples:
                if after_parse_hook is not None:
                    after_parse_hook(prep_record, built_samples)
                pending.extend(built_samples)
                for s in built_samples:
                    accepted_subruns += len(s.runs)
                    buckets_new[s.bucket] += 1
                pending_journals.append(
                    {
                        "request_id": request_id,
                        "sample_ids": [str(r["sample_id"]) for r in ok_plan_rows],
                        "intended_subruns": sum(int(r["repeat_count"]) for r in ok_plan_rows),
                    }
                )
            elif failed_entries:
                sync_journal_only()
                continue

            while len(pending) >= chunk_size:
                flush(chunk_size)
            if failed_entries and pending:
                sync_journal_only()

        if pending:
            flush(len(pending))
        else:
            sync_artifacts()
    except KeyboardInterrupt:
        if pending:
            flush(len(pending))
        else:
            sync_artifacts()
        raise

    total_battles = existing_battles + written_battles
    total_submitted_subruns = prior_submitted_subruns + session_submitted_subruns
    total_accepted_subruns = existing_subruns + accepted_subruns
    completed_target = total_battles >= goal_battles
    elapsed = max(1e-9, time.monotonic() - started)
    result = {
        "worker_id": worker_id,
        "round_id": expected_round_id,
        "plan_sha256": actual_plan_sha,
        "existing_battles": existing_battles,
        "new_battles": written_battles,
        "total_battles": total_battles,
        "target_battles": goal_battles,
        "completed_target": completed_target,
        "exhausted": exhausted,
        "stopped_by_time_limit": stopped_by_time_limit,
        "stopped_by_budget": stopped_by_budget,
        "submitted_subruns": total_submitted_subruns,
        "session_submitted_subruns": session_submitted_subruns,
        "returned_subruns": returned_subruns,
        "accepted_subruns": total_accepted_subruns,
        "session_accepted_subruns": accepted_subruns,
        "unknown_subruns": max(0, total_submitted_subruns - total_accepted_subruns),
        "quarantined_battles": quarantined_battles + len(prior_quarantined_ids),
        "indeterminate_battles": len(prior_indeterminate_ids),
        "reserve_used_battles": reserve_used_battles,
        "elapsed_sec": round(elapsed, 4),
        "subruns_per_sec": round(accepted_subruns / elapsed, 4),
        "buckets_new": dict(sorted(buckets_new.items())),
    }
    if exhausted and not completed_target:
        raise RuntimeError(
            f"plan exhausted before reaching target_battles={goal_battles}: "
            f"total_battles={total_battles}, submitted_subruns={total_submitted_subruns}"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reserve-plan", type=Path)
    parser.add_argument("--allow-reserve", action="store_true")
    parser.add_argument("--s2-registry", type=Path)
    parser.add_argument("--s5-denylist", type=Path)
    parser.add_argument("--target-battles", type=int)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-subruns-per-post", type=int, default=32)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-submitted-subruns", type=int)
    parser.add_argument("--sync-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)

    catalog = FeatureCatalog.load(args.catalog)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    profile = dict(profile) | {"timeout_sec": args.timeout_sec, "max_retries": 0, "retry_on_timeout": False}
    client = SimClient(args.url, catalog, profile)
    s2_reg = S2HoldoutRegistry.load(args.s2_registry) if args.s2_registry else None
    s5_deny = S5Denylist.load(args.s5_denylist) if args.s5_denylist else None
    result = run_worker_plan(
        sidecar_path=args.sidecar,
        worker_plan_path=args.plan,
        worker_id=args.worker_id,
        catalog=catalog,
        profile=profile,
        client=client,
        out_path=args.out,
        reserve_plan_path=args.reserve_plan,
        allow_reserve=args.allow_reserve,
        s2_registry=s2_reg,
        s5_denylist=s5_deny,
        target_battles=args.target_battles,
        chunk_size=args.chunk_size,
        max_subruns_per_post=args.max_subruns_per_post,
        max_seconds=args.max_seconds,
        max_submitted_subruns=args.max_submitted_subruns,
        sync_dir=args.sync_dir,
        resume=args.resume,
        progress_stderr=args.progress,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
