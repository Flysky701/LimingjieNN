"""Offline S3 active-mining planner: S2 train-only anchors, TwoTowerEnsemble scoring, and worker manifests."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import comb
from pathlib import Path
from typing import TYPE_CHECKING, Any

from labyrinth_nn.active_mining.identity import (
    DEFAULT_NUM_WORKERS,
    UNSUPPORTED_SIM_TEAM_UNITS,
    S2HoldoutRegistry,
    semantic_context_sha256,
    sha256_file,
    sha256_json,
    worker_for_context,
    worker_id_for_index,
    worker_index_from_id,
    write_json_atomic,
    write_jsonl_atomic,
)
from labyrinth_nn.active_mining.s5_reservation import (
    S5Denylist,
    _draw_context_with_teams,
    _explicit_seeds,
    _sample_id_for_round,
)
from labyrinth_nn.catalog.schema import FeatureCatalog
from labyrinth_nn.data.dataset import BattleDataset, collate_battle_batch
from labyrinth_nn.data.sample_schema import BattleSample, OUTCOME_WIN
from labyrinth_nn.data.sampler import HighScoreSampler

if TYPE_CHECKING:
    from labyrinth_nn.models.critic import TwoTowerEnsemble

PLANNER_ALGORITHM_VERSION = "s3_active_mining_v1"


@dataclass(frozen=True)
class S3PlannerSpec:
    num_workers: int = DEFAULT_NUM_WORKERS
    per_worker_rows: int = 20000
    bucket_quotas_per_worker: Mapping[str, int] = field(
        default_factory=lambda: {"normal": 8000, "robot_ex": 5000, "boss": 3000, "broad": 4000}
    )
    repeat_tier_counts_per_worker: Mapping[int, int] = field(
        default_factory=lambda: {1: 10000, 6: 7000, 12: 3000}
    )
    reserve_rows_per_worker: int = 1000
    reserve_repeats: int = 6
    random_control_fraction: float = 0.08
    overconfident_win_prob_min: float = 0.70
    observed_nonwin_max_rate: float = 0.50
    disagreement_std_min: float = 0.06
    boundary_prob_low: float = 0.25
    boundary_prob_high: float = 0.75
    max_mutations_per_anchor: int = 4
    max_anchor_candidates_generated: int = 18
    max_anchors_per_stratum: int = 2500
    candidates_per_fresh_context: int = 6
    seed: int = 20261125
    round_id_prefix: str = "round2_s3"
    batch_size: int = 512
    num_workers_loader: int = 0
    device: str = "cpu"
    underfull_team_prob: float = 0.03
    excluded_team_units: frozenset[int] = UNSUPPORTED_SIM_TEAM_UNITS


def load_calibrated_ensemble(
    checkpoint_paths: Sequence[str | Path],
    catalog: FeatureCatalog,
    *,
    expected_sha256s: Sequence[str] | None = None,
    device: str = "cpu",
    lcb_z: float = 1.0,
) -> tuple[TwoTowerEnsemble, list[dict[str, Any]]]:
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint path is required")
    if expected_sha256s is not None and len(expected_sha256s) != len(checkpoint_paths):
        raise ValueError("expected_sha256s length must match checkpoint_paths")
    import torch
    from labyrinth_nn.models.critic import CriticConfig, TwoTowerBattleCritic, TwoTowerEnsemble

    catalog_sha = catalog.catalog_sha256 or catalog.compute_sha256()
    models: list[TwoTowerBattleCritic] = []
    meta: list[dict[str, Any]] = []
    for idx, raw_path in enumerate(checkpoint_paths):
        path = Path(raw_path)
        actual_sha = sha256_file(path)
        if expected_sha256s is not None and actual_sha.lower() != expected_sha256s[idx].lower():
            raise ValueError(f"checkpoint sha256 mismatch for {path}: {actual_sha} != {expected_sha256s[idx]}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("catalog_sha256") != catalog_sha:
            raise ValueError(f"checkpoint catalog_sha256 mismatch for {path}")
        config = CriticConfig.from_dict(state["config"], catalog)
        model = TwoTowerBattleCritic(config)
        model.load_state_dict(state["model"])
        model.to(device).eval()
        models.append(model)
        meta.append(
            {
                "path": str(path),
                "sha256": actual_sha,
                "ablation_mode": config.ablation_mode,
                "seed": state.get("seed"),
                "temperature": float(model.log_temperature.clamp(-4.0, 4.0).exp().item()),
            }
        )
    ensemble = TwoTowerEnsemble(models, lcb_z=lcb_z).to(device).eval()
    return ensemble, meta


def score_samples_with_ensemble(
    ensemble: TwoTowerEnsemble,
    catalog: FeatureCatalog,
    samples: Sequence[BattleSample],
    *,
    batch_size: int = 512,
    num_workers: int = 0,
    device: str = "cpu",
) -> list[dict[str, float]]:
    """Full forward BattleDataset scoring (no roster-cached shortcut)."""
    if not samples:
        return []
    import torch
    from torch.utils.data import DataLoader

    dataset = BattleDataset(
        catalog,
        samples,
        training=False,
        cache_eval=False,
        roster_max_len=ensemble.config.roster_max_len,
        relic_max_len=ensemble.config.relic_max_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_battle_batch,
    )
    scores: list[dict[str, float]] = []
    with torch.no_grad():
        for raw in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
            out = ensemble(batch)
            required = ("p_win_mean", "p_win_std", "lcb_win", "damage_mean", "win_rem_time")
            if any(key not in out for key in required):
                raise ValueError("ensemble output missing required prediction keys")
            cols = {k: out[k].detach().cpu().reshape(-1).tolist() for k in required}
            row_n = len(raw["sample_id"])
            if any(len(cols[k]) != row_n for k in required):
                raise ValueError("ensemble output length mismatch")
            for i in range(row_n):
                entry = {k: float(cols[k][i]) for k in required}
                if any(not math.isfinite(val) for val in entry.values()):
                    raise ValueError(f"non-finite ensemble prediction for sample {raw['sample_id'][i]}")
                if not (0.0 <= entry["p_win_mean"] <= 1.0 and 0.0 <= entry["lcb_win"] <= 1.0 and entry["p_win_std"] >= 0.0):
                    raise ValueError(f"out-of-range ensemble prediction for sample {raw['sample_id'][i]}")
                scores.append(entry)
    if len(scores) != len(samples):
        raise ValueError(f"scored row count {len(scores)} != input count {len(samples)}")
    return scores


def is_overconfident_loss_anchor(
    sample: BattleSample,
    score: Mapping[str, float],
    *,
    overconfident_win_prob_min: float = 0.70,
    observed_nonwin_max_rate: float = 0.50,
) -> bool:
    """Strictly separates high-p observed non-wins from low-p wins even if |error| is larger."""
    total = sum(sample.outcome_counts)
    if total <= 0:
        return False
    obs_win_rate = sample.outcome_counts[OUTCOME_WIN] / total
    return score["p_win_mean"] >= overconfident_win_prob_min and obs_win_rate < observed_nonwin_max_rate


def _validate_legal_mutation(parent: BattleSample, candidate_team: Sequence[int]) -> int:
    owned_set = set(parent.owned_unit_ids)
    cand_set = set(candidate_team)
    if len(cand_set) != len(candidate_team) or not cand_set <= owned_set:
        raise ValueError("candidate team must be duplicate-free subset of parent owned_unit_ids")
    parent_set = set(parent.team_unit_ids)
    if len(parent_set) == len(cand_set):
        added = cand_set - parent_set
        removed = parent_set - cand_set
        if len(added) != len(removed) or len(added) not in (1, 2):
            raise ValueError(f"same-size mutation must replace 1 or 2 slots, got added={len(added)}")
        return len(added)
    diff = len(parent_set ^ cand_set)
    if diff not in (1, 2):
        raise ValueError(f"mutation must change 1 or 2 slots, got symmetric_diff={diff}")
    return diff


def generate_one_two_slot_mutations(
    catalog: FeatureCatalog,
    parent: BattleSample,
    *,
    limit: int = 18,
    rng: random.Random | None = None,
    excluded_team_units: frozenset[int] = UNSUPPORTED_SIM_TEAM_UNITS,
) -> list[tuple[list[int], int]]:
    """Deterministically generate legal distinct 1-slot and 2-slot team replacements inside owned_unit_ids."""
    team = sorted(parent.team_unit_ids, key=lambda uid: (catalog.units[uid].search_area_width, uid))
    team_set = set(team)
    others = sorted(
        uid for uid in parent.owned_unit_ids if uid not in team_set and uid not in excluded_team_units
    )
    if not others or not team:
        return []
    one_slot: list[tuple[list[int], int]] = []
    for out_uid in team:
        for in_uid in others:
            mutated = sorted(
                (team_set - {out_uid}) | {in_uid},
                key=lambda uid: (catalog.units[uid].search_area_width, uid),
            )
            if excluded_team_units & set(mutated):
                continue
            one_slot.append((mutated, 1))
    two_slot: list[tuple[list[int], int]] = []
    if len(team) >= 2 and len(others) >= 2:
        for out_pair in combinations(team, 2):
            for in_pair in combinations(others, 2):
                mutated = sorted(
                    (team_set - set(out_pair)) | set(in_pair),
                    key=lambda uid: (catalog.units[uid].search_area_width, uid),
                )
                if excluded_team_units & set(mutated):
                    continue
                two_slot.append((mutated, 2))
                if len(two_slot) >= limit * 2:
                    break
            if len(two_slot) >= limit * 2:
                break
    if rng is not None:
        rng.shuffle(one_slot)
        rng.shuffle(two_slot)
    # Interleave 1-slot and 2-slot candidates deterministically up to limit.
    results: list[tuple[list[int], int]] = []
    seen_teams: set[tuple[int, ...]] = set()
    n1 = max(1, (limit * 2) // 3)
    for mutated, slots in one_slot[:n1] + two_slot[: max(0, limit - min(len(one_slot), n1))]:
        key = tuple(mutated)
        if key not in seen_teams:
            seen_teams.add(key)
            results.append((mutated, slots))
        if len(results) >= limit:
            break
    if len(results) < limit:
        for mutated, slots in one_slot[n1:] + two_slot:
            key = tuple(mutated)
            if key not in seen_teams:
                seen_teams.add(key)
                results.append((mutated, slots))
            if len(results) >= limit:
                break
    return results


def _stratum_for_bucket(bucket: str) -> str:
    if bucket == "normal":
        return "normal"
    if bucket == "robot_ex":
        return "robot_ex"
    if bucket.startswith("boss_"):
        return "boss"
    return "broad"


def _make_plan_row(
    *,
    worker_id: str,
    round_id: str,
    sample: BattleSample,
    arm: str,
    anchor_sample_id: str | None,
    reason: str,
    is_random_control: bool,
    is_reserve: bool,
    scores: Mapping[str, float],
    repeat_count: int,
    run_seeds: list[int],
    plan_index: int = 0,
) -> dict[str, Any]:
    ctx_sha = semantic_context_sha256(sample)
    phys = sample.physical_state_key()
    return {
        "plan_index": plan_index,
        "worker_id": worker_id,
        "round_id": round_id,
        "sample_id": sample.sample_id,
        "semantic_context_sha256": ctx_sha,
        "physical_state_key": phys,
        "arm": arm,
        "anchor_sample_id": anchor_sample_id,
        "reason": reason,
        "is_random_control": bool(is_random_control),
        "is_reserve": bool(is_reserve),
        "bucket": sample.bucket,
        "scores": {k: float(v) for k, v in scores.items()},
        "repeat_count": int(repeat_count),
        "run_seeds": list(run_seeds),
        "payload": {
            "db_version": sample.db_version,
            "quest_id": sample.quest_id,
            "canonical_wave_sig": sample.canonical_wave_sig,
            "bucket": sample.bucket,
            "area": sample.area,
            "difficulty": sample.difficulty,
            "guild_id": sample.guild_id,
            "team_unit_ids": list(sample.team_unit_ids),
            "owned_unit_ids": list(sample.owned_unit_ids),
            "owned_unit_count": sample.owned_unit_count,
            "owned_role_counts": list(sample.owned_role_counts),
            "owned_talent_counts": list(sample.owned_talent_counts),
            "relic_ids": list(sample.relic_ids),
            "mark_stacks": list(sample.mark_stacks),
            "alpha_rupee": sample.alpha_rupee,
            "revive_available": sample.revive_available,
            "context_id": ctx_sha,
        },
    }


def _assign_repeat_tiers(
    items: list[dict[str, Any]],
    tier_counts: Mapping[int, int],
    seed: int,
    rng_keywords: Sequence[str],
    catalog: FeatureCatalog,
) -> list[dict[str, Any]]:
    """Assign exact 1/6/12 repeat tiers across a worker's rows, keeping random_control stratified across tiers."""
    total = len(items)
    if sum(tier_counts.values()) != total:
        raise ValueError(f"repeat tier sum {sum(tier_counts.values())} != worker row count {total}")
    remaining = {int(k): int(v) for k, v in tier_counts.items()}

    # Partition into random controls and active candidates, both deterministically ordered.
    controls = [row for row in items if row["is_random_control"]]
    actives = [row for row in items if not row["is_random_control"]]
    assigned: dict[int, int] = {}

    if controls:
        for tier in sorted(remaining, reverse=True):
            share = min(remaining[tier], round(len(controls) * (tier_counts[tier] / max(1, total))))
            for row in controls:
                if share <= 0:
                    break
                if id(row) not in assigned:
                    assigned[id(row)] = tier
                    remaining[tier] -= 1
                    share -= 1
        for row in controls:
            if id(row) not in assigned:
                for tier in sorted(remaining):
                    if remaining[tier] > 0:
                        assigned[id(row)] = tier
                        remaining[tier] -= 1
                        break

    def priority_key(row: dict[str, Any]) -> tuple[Any, ...]:
        arm = row["arm"]
        arm_rank = {
            "overconfident_loss_mutation": 3,
            "ensemble_disagreement": 2,
            "near_boundary": 1,
            "broad_exploration": 0,
            "random_control": 0,
        }.get(arm, 0)
        qname = catalog.quests[row["payload"]["quest_id"]].quest_name
        has_rng_kw = int(any(kw in qname for kw in rng_keywords))
        scores = row["scores"]
        boundary_closeness = -abs(scores["p_win_mean"] - 0.5)
        return (-arm_rank, -has_rng_kw, -scores["p_win_std"], boundary_closeness, row["physical_state_key"])

    actives_sorted = sorted(actives, key=priority_key)
    idx = 0
    for tier in sorted(remaining, reverse=True):
        count = remaining[tier]
        for row in actives_sorted[idx : idx + count]:
            assigned[id(row)] = tier
        idx += count

    finalized: list[dict[str, Any]] = []
    # Sort finalized worker rows deterministically by physical_state_key and assign plan_index.
    for plan_idx, row in enumerate(sorted(items, key=lambda r: r["physical_state_key"])):
        tier = assigned[id(row)]
        seeds = _explicit_seeds("s3", seed, row["physical_state_key"], tier)
        updated = dict(row)
        updated["plan_index"] = plan_idx
        updated["repeat_count"] = tier
        updated["run_seeds"] = seeds
        finalized.append(updated)
    return finalized


def verify_s3_plan(
    *,
    sidecar: Mapping[str, Any],
    worker_plans: Mapping[str, Sequence[Mapping[str, Any]]],
    worker_reserves: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    catalog: FeatureCatalog,
    s2_registry: S2HoldoutRegistry | None = None,
    s5_denylist: S5Denylist | None = None,
    existing_physical_states: Iterable[str] = (),
) -> None:
    """Fail-closed invariant checker for S3 plan sidecar and per-worker manifests."""
    catalog_sha = catalog.catalog_sha256 or catalog.compute_sha256()
    if sidecar.get("catalog_sha256") != catalog_sha:
        raise ValueError("sidecar catalog_sha256 mismatch")
    if sidecar.get("db_version") != catalog.db_version:
        raise ValueError("sidecar db_version mismatch")
    if s2_registry is not None and sidecar.get("s2_denylist_sha256") != s2_registry.denylist_sha256:
        raise ValueError("sidecar s2_denylist_sha256 mismatch")
    if s5_denylist is not None and sidecar.get("s5_denylist_sha256") != s5_denylist.denylist_sha256:
        raise ValueError("sidecar s5_denylist_sha256 mismatch")

    num_workers = int(sidecar["num_workers"])
    excluded_units = frozenset(int(u) for u in (sidecar.get("excluded_sim_team_units") or ()))
    seen_phys = set(existing_physical_states)
    seen_sample_ids: set[str] = set()
    worker_contexts: dict[str, set[str]] = defaultdict(set)

    def check_rows(worker_id: str, rows: Sequence[Mapping[str, Any]], expected_reserve: bool) -> None:
        w_idx = worker_index_from_id(worker_id, num_workers)
        for idx, row in enumerate(rows):
            if row.get("worker_id") != worker_id:
                raise ValueError(f"row {idx} worker_id mismatch: {row.get('worker_id')} != {worker_id}")
            if bool(row.get("is_reserve")) != expected_reserve:
                raise ValueError(f"row {idx} unexpected is_reserve={row.get('is_reserve')}")
            payload = row["payload"]
            ctx_sha = semantic_context_sha256(payload)
            if row.get("semantic_context_sha256") != ctx_sha:
                raise ValueError(f"row {idx} semantic_context_sha256 mismatch")
            if worker_for_context(ctx_sha, num_workers) != w_idx:
                raise ValueError(f"row {idx} context assigned to wrong worker {worker_id}")
            if s2_registry is not None and s2_registry.is_holdout_sample(payload):
                raise ValueError(f"row {idx} violates S2 holdout denylist")
            if s5_denylist is not None and s5_denylist.is_denied(payload):
                raise ValueError(f"row {idx} violates S5 denylist")
            team = [int(u) for u in payload["team_unit_ids"]]
            owned = [int(u) for u in payload["owned_unit_ids"]]
            if len(set(team)) != len(team) or not set(team) <= set(owned):
                raise ValueError(f"row {idx} team_unit_ids outside owned_unit_ids")
            if excluded_units and (excluded_units & set(team)):
                raise ValueError(f"row {idx} team_unit_ids contains excluded simulator unit(s)")
            phys_expected = BattleSample(
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
                team_unit_ids=team,
                owned_unit_ids=owned,
                owned_unit_count=int(payload["owned_unit_count"]),
                owned_role_counts=[int(x) for x in payload["owned_role_counts"]],
                owned_talent_counts=[int(x) for x in payload["owned_talent_counts"]],
                relic_ids=[int(r) for r in payload["relic_ids"]],
                mark_stacks=[int(m) for m in payload["mark_stacks"]],
                alpha_rupee=None if payload.get("alpha_rupee") is None else int(payload["alpha_rupee"]),
                revive_available=bool(payload["revive_available"]),
            ).physical_state_key()
            if row.get("physical_state_key") != phys_expected:
                raise ValueError(f"row {idx} physical_state_key mismatch")
            if phys_expected in seen_phys:
                raise ValueError(f"duplicate physical_state_key {phys_expected}")
            if row["sample_id"] in seen_sample_ids:
                raise ValueError(f"duplicate sample_id {row['sample_id']}")
            seen_phys.add(phys_expected)
            seen_sample_ids.add(str(row["sample_id"]))
            seeds = row.get("run_seeds") or []
            if len(seeds) != int(row["repeat_count"]) or len(set(seeds)) != len(seeds):
                raise ValueError(f"row {idx} invalid run_seeds for repeat_count={row['repeat_count']}")
            worker_contexts[worker_id].add(ctx_sha)

    for worker_id, rows in sorted(worker_plans.items()):
        check_rows(worker_id, rows, expected_reserve=False)
    if worker_reserves is not None:
        for worker_id, rows in sorted(worker_reserves.items()):
            check_rows(worker_id, rows, expected_reserve=True)

    # Pairwise semantic context disjointness across all workers.
    workers_list = sorted(worker_contexts)
    for i, w1 in enumerate(workers_list):
        for w2 in workers_list[i + 1 :]:
            overlap = worker_contexts[w1] & worker_contexts[w2]
            if overlap:
                raise ValueError(f"workers {w1} and {w2} share semantic contexts: {len(overlap)}")


def plan_s3_mining(
    *,
    catalog: FeatureCatalog,
    profile: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
    samples: Sequence[BattleSample],
    s2_registry: S2HoldoutRegistry,
    s5_denylist: S5Denylist,
    checkpoint_paths: Sequence[str | Path],
    out_dir: str | Path,
    expected_checkpoint_sha256s: Sequence[str] | None = None,
    spec: S3PlannerSpec | None = None,
    source_zip_sha256: str = "",
    db_sha256: str = "",
    seal_sha256: str = "",
    extra_denied_contexts: Iterable[str] = (),
    extra_denied_physical_states: Iterable[str] = (),
) -> dict[str, Any]:
    """Deterministically plan S3 active mining manifests and sidecar without any simulator calls."""
    spec = spec or S3PlannerSpec()
    catalog_sha = catalog.catalog_sha256 or catalog.compute_sha256()
    if s2_registry.catalog_sha256 != catalog_sha or s5_denylist.catalog_sha256 != catalog_sha:
        raise ValueError("catalog_sha256 differs between catalog, S2 registry, and S5 denylist")
    if s2_registry.db_version != catalog.db_version or s5_denylist.db_version != catalog.db_version:
        raise ValueError("db_version differs between catalog, S2 registry, and S5 denylist")
    if sum(spec.bucket_quotas_per_worker.values()) != spec.per_worker_rows:
        raise ValueError("bucket_quotas_per_worker must sum to per_worker_rows")
    if sum(spec.repeat_tier_counts_per_worker.values()) != spec.per_worker_rows:
        raise ValueError("repeat_tier_counts_per_worker must sum to per_worker_rows")

    # Sort input samples deterministically so input file permutation never changes outputs.
    ordered_samples = sorted(samples, key=lambda s: (s.sample_id, s.physical_state_key()))
    existing_phys: set[str] = set()
    existing_sample_ids: set[str] = set()
    s2_all_contexts: set[str] = set()
    train_anchors: list[BattleSample] = []

    for s in ordered_samples:
        if s.db_version != catalog.db_version:
            raise ValueError(f"sample {s.sample_id} db_version {s.db_version!r} != {catalog.db_version!r}")
        if s.sample_id in existing_sample_ids:
            raise ValueError(f"duplicate sample_id in input corpus: {s.sample_id}")
        if s5_denylist.is_denied(s):
            raise ValueError(f"input corpus overlaps S5 reserved denylist on sample {s.sample_id}")
        existing_sample_ids.add(s.sample_id)
        existing_phys.add(s.physical_state_key())
        ctx_sha = semantic_context_sha256(s)
        s2_all_contexts.add(ctx_sha)
        fold = s2_registry.sample_folds.get(s.sample_id)
        if fold == "train" and not s2_registry.is_holdout_sample(s):
            train_anchors.append(s)

    ensemble, ckpt_meta = load_calibrated_ensemble(
        checkpoint_paths, catalog, expected_sha256s=expected_checkpoint_sha256s, device=spec.device
    )
    anchor_scores = score_samples_with_ensemble(
        ensemble, catalog, train_anchors, batch_size=spec.batch_size,
        num_workers=spec.num_workers_loader, device=spec.device,
    )

    denied_ctx_set = {str(x) for x in extra_denied_contexts}
    denied_phys_set = {str(x) for x in extra_denied_physical_states}
    used_phys = set(existing_phys) | denied_phys_set
    blocked_ctx = (
        set(s2_registry.holdout_semantic_contexts)
        | set(s5_denylist.reserved_semantic_contexts)
        | denied_ctx_set
    )
    rng_keywords = tuple(profile.get("adaptive_repeat", {}).get("high_rng_keywords", ()))

    # 1) Generate candidate mutations from S2 train anchors.
    eligible_anchors_by_arm: dict[str, list[tuple[tuple[Any, ...], BattleSample, Mapping[str, float]]]] = defaultdict(list)
    for sample, score in zip(train_anchors, anchor_scores, strict=True):
        total = sum(sample.outcome_counts)
        if total <= 0:
            continue
        obs_win = sample.outcome_counts[OUTCOME_WIN] / total
        p_win = score["p_win_mean"]
        p_std = score["p_win_std"]
        phys = sample.physical_state_key()
        if is_overconfident_loss_anchor(
            sample, score,
            overconfident_win_prob_min=spec.overconfident_win_prob_min,
            observed_nonwin_max_rate=spec.observed_nonwin_max_rate,
        ):
            eligible_anchors_by_arm["overconfident_loss_mutation"].append(
                ((-(p_win - obs_win), -p_win, phys), sample, score)
            )
        elif p_std >= spec.disagreement_std_min:
            eligible_anchors_by_arm["ensemble_disagreement"].append(
                ((-p_std, abs(p_win - 0.5), phys), sample, score)
            )
        elif spec.boundary_prob_low <= p_win <= spec.boundary_prob_high:
            eligible_anchors_by_arm["near_boundary"].append(
                ((abs(p_win - 0.5), -p_std, phys), sample, score)
            )

    # Bound anchor mutation scoring cost while remaining strictly deterministic.
    mutation_proposals: list[tuple[str, BattleSample, BattleSample, Mapping[str, float], int]] = []
    seen_proposed_phys = set(used_phys)
    for arm_name in ("overconfident_loss_mutation", "ensemble_disagreement", "near_boundary"):
        ranked_anchors = sorted(eligible_anchors_by_arm[arm_name], key=lambda item: item[0])
        per_stratum_count: Counter[str] = Counter()
        for _, parent, parent_score in ranked_anchors:
            stratum = _stratum_for_bucket(parent.bucket)
            if per_stratum_count[stratum] >= spec.max_anchors_per_stratum:
                continue
            per_stratum_count[stratum] += 1
            local_rng = random.Random(f"mut:{spec.seed}:{parent.physical_state_key()}")
            for mutated_team, slots_changed in generate_one_two_slot_mutations(
                catalog,
                parent,
                limit=spec.max_anchor_candidates_generated,
                rng=local_rng,
                excluded_team_units=spec.excluded_team_units,
            ):
                if spec.excluded_team_units & set(mutated_team):
                    continue
                _validate_legal_mutation(parent, mutated_team)
                ctx_sha = semantic_context_sha256(parent)
                worker_idx = worker_for_context(ctx_sha, spec.num_workers)
                worker_id = worker_id_for_index(worker_idx)
                round_id = f"{spec.round_id_prefix}_{worker_id}"
                cand = replace(
                    parent,
                    team_unit_ids=mutated_team,
                    round_id=round_id,
                    context_id=ctx_sha,
                    runs=[],
                    outcome_counts=[0, 0, 0],
                    mean_damage_ratio=0.0,
                    mean_win_rem_time=None,
                )
                phys = cand.physical_state_key()
                if phys in seen_proposed_phys:
                    continue
                seen_proposed_phys.add(phys)
                cand = replace(cand, sample_id=_sample_id_for_round(round_id, catalog, cand))
                mutation_proposals.append((arm_name, parent, cand, parent_score, slots_changed))

    mutation_scores = score_samples_with_ensemble(
        ensemble, catalog, [item[2] for item in mutation_proposals],
        batch_size=spec.batch_size, num_workers=spec.num_workers_loader, device=spec.device,
    )

    # Group scored anchor mutations by (worker_id, stratum) keeping up to max_mutations_per_anchor per anchor.
    worker_stratum_pool: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_anchor: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = defaultdict(list)
    for (arm_name, parent, cand, parent_score, slots_changed), cand_score in zip(
        mutation_proposals, mutation_scores, strict=True
    ):
        ctx_sha = semantic_context_sha256(cand)
        worker_id = worker_id_for_index(worker_for_context(ctx_sha, spec.num_workers))
        stratum = _stratum_for_bucket(cand.bucket)
        reason = (
            f"anchor_p_win={parent_score['p_win_mean']:.4f},obs_win={parent.outcome_counts[OUTCOME_WIN]}/{sum(parent.outcome_counts)},"
            f"slots_mutated={slots_changed}"
        )
        row = _make_plan_row(
            worker_id=worker_id,
            round_id=cand.round_id,
            sample=cand,
            arm=arm_name,
            anchor_sample_id=parent.sample_id,
            reason=reason,
            is_random_control=False,
            is_reserve=False,
            scores=cand_score,
            repeat_count=1,
            run_seeds=[],
        )
        sort_k = (
            -cand_score["p_win_mean"] if arm_name == "overconfident_loss_mutation" else -cand_score["p_win_std"],
            abs(cand_score["p_win_mean"] - 0.5),
            cand.physical_state_key(),
        )
        by_anchor[parent.sample_id].append((sort_k, (worker_id, stratum, row)))

    for anchor_id in sorted(by_anchor):
        picked = sorted(by_anchor[anchor_id], key=lambda item: item[0])[: spec.max_mutations_per_anchor]
        for _, (worker_id, stratum, row) in picked:
            worker_stratum_pool[(worker_id, stratum)].append(row)

    # Allocate selected anchor mutations into worker quotas (preserving room for random controls & broad).
    worker_selected: dict[str, list[dict[str, Any]]] = {
        worker_id_for_index(i): [] for i in range(spec.num_workers)
    }
    worker_stratum_counts: dict[tuple[str, str], int] = Counter()
    chosen_phys = set(used_phys)
    fresh_ctx_assigned_worker: dict[str, str] = {}

    for worker_id in sorted(worker_selected):
        for stratum, target_count in sorted(spec.bucket_quotas_per_worker.items()):
            control_reserve = round(target_count * spec.random_control_fraction)
            anchor_cap = max(0, target_count - control_reserve)
            candidates = sorted(
                worker_stratum_pool[(worker_id, stratum)],
                key=lambda r: (
                    0 if r["arm"] == "overconfident_loss_mutation" else (1 if r["arm"] == "ensemble_disagreement" else 2),
                    -r["scores"]["p_win_std"],
                    abs(r["scores"]["p_win_mean"] - 0.5),
                    r["physical_state_key"],
                ),
            )
            for row in candidates:
                if worker_stratum_counts[(worker_id, stratum)] >= anchor_cap:
                    break
                if row["physical_state_key"] in chosen_phys:
                    continue
                chosen_phys.add(row["physical_state_key"])
                worker_selected[worker_id].append(row)
                worker_stratum_counts[(worker_id, stratum)] += 1

    # 2) Fill remaining per-(worker, stratum) quotas & reserve pool with fresh unreserved contexts scored by ensemble.
    helper_cfg = dict(sampling_config)
    helper_cfg["seed"] = spec.seed
    helper_cfg["round_id"] = spec.round_id_prefix
    helper_cfg["paired_teams_per_context"] = 1
    helper = HighScoreSampler(catalog, profile, helper_cfg)
    playable_count = sum(1 for uid in catalog.units if uid not in spec.excluded_team_units)
    min_owned = next(
        (n for n in range(5, playable_count + 1) if comb(n, 5) >= spec.candidates_per_fresh_context),
        None,
    )
    if min_owned is None:
        raise ValueError(
            f"catalog has too few playable units ({playable_count}) for "
            f"candidates_per_fresh_context={spec.candidates_per_fresh_context}"
        )
    # Determine available buckets per stratum from catalog/helper.
    available_buckets = set(helper.quests.keys())

    def target_bucket_for_stratum(stratum: str, draw_idx: int) -> str:
        if stratum in ("normal", "robot_ex") and stratum in available_buckets:
            return stratum
        if stratum == "boss":
            choices = [b for b in ("boss_3f_d5", "boss_5f_d5") if b in available_buckets]
            return choices[draw_idx % len(choices)] if choices else next(iter(available_buckets))
        broad_choices = [b for b in ("special_dual_ex", "other_ex", "normal", "robot_ex") if b in available_buckets]
        return broad_choices[draw_idx % len(broad_choices)]

    used_fresh_ctx = set(blocked_ctx) | s2_all_contexts

    def deficit_for(worker_id: str, stratum: str) -> int:
        return spec.bucket_quotas_per_worker[stratum] - worker_stratum_counts[(worker_id, stratum)]

    # Draw batches of fresh contexts until all (worker, stratum) quotas are met.
    draw_counter = 0
    control_total_target = round(spec.per_worker_rows * spec.random_control_fraction)
    worker_control_counts: Counter[str] = Counter(
        {w: sum(1 for r in rows if r["is_random_control"]) for w, rows in worker_selected.items()}
    )

    while any(deficit_for(worker_id_for_index(i), s) > 0 for i in range(spec.num_workers) for s in spec.bucket_quotas_per_worker):
        # Gather a batch of fresh candidate rows for remaining needed strata.
        needed_strata = [
            s for s in sorted(spec.bucket_quotas_per_worker)
            if any(deficit_for(worker_id_for_index(i), s) > 0 for i in range(spec.num_workers))
        ]
        pending_contexts: list[tuple[str, str, BattleSample, list[BattleSample], bool]] = []
        batch_samples: list[BattleSample] = []
        batch_slices: list[tuple[int, int]] = []
        attempts = 0
        total_remaining = sum(
            max(0, deficit_for(worker_id_for_index(i), s))
            for i in range(spec.num_workers)
            for s in needed_strata
        )
        max_attempts = max(4000, total_remaining * 40)
        target_ctx_batch = min(256, max(1, total_remaining))

        while len(pending_contexts) < target_ctx_batch:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("exceeded max attempts drawing fresh unreserved contexts for S3")
            stratum = needed_strata[draw_counter % len(needed_strata)]
            bucket = target_bucket_for_stratum(stratum, draw_counter)
            draw_counter += 1
            base, teams = _draw_context_with_teams(
                helper,
                bucket,
                spec.candidates_per_fresh_context,
                spec.underfull_team_prob,
                min_owned,
                excluded_team_units=spec.excluded_team_units,
            )
            ctx_sha = semantic_context_sha256(base)
            if ctx_sha in used_fresh_ctx:
                continue
            worker_id = worker_id_for_index(worker_for_context(ctx_sha, spec.num_workers))
            if deficit_for(worker_id, stratum) <= 0:
                continue
            used_fresh_ctx.add(ctx_sha)
            round_id = f"{spec.round_id_prefix}_{worker_id}"
            ctx_cands: list[BattleSample] = []
            for t in teams:
                s_obj = replace(base, team_unit_ids=t, round_id=round_id, context_id=ctx_sha)
                if s_obj.physical_state_key() in chosen_phys:
                    continue
                s_obj = replace(s_obj, sample_id=_sample_id_for_round(round_id, catalog, s_obj))
                ctx_cands.append(s_obj)
            if not ctx_cands:
                continue
            is_ctrl = worker_control_counts[worker_id] < control_total_target
            start_i = len(batch_samples)
            batch_samples.extend(ctx_cands)
            end_i = len(batch_samples)
            pending_contexts.append((worker_id, stratum, base, ctx_cands, is_ctrl))
            batch_slices.append((start_i, end_i))

        batch_scores = score_samples_with_ensemble(
            ensemble, catalog, batch_samples, batch_size=spec.batch_size,
            num_workers=spec.num_workers_loader, device=spec.device,
        )
        for (worker_id, stratum, _, ctx_cands, is_ctrl), (start_i, end_i) in zip(
            pending_contexts, batch_slices, strict=True
        ):
            if deficit_for(worker_id, stratum) <= 0:
                continue
            scored_pairs = list(zip(ctx_cands, batch_scores[start_i:end_i], strict=True))
            if is_ctrl and worker_control_counts[worker_id] < control_total_target:
                chosen_cand, chosen_sc = scored_pairs[0]
                arm = "random_control"
                reason = "pre_registered_random_control"
                ctrl_flag = True
                worker_control_counts[worker_id] += 1
            else:
                # Choose candidate with highest disagreement or closest boundary, deterministically broken by physical key.
                best_pair = min(
                    scored_pairs,
                    key=lambda pair: (
                        -pair[1]["p_win_std"] if pair[1]["p_win_std"] >= spec.disagreement_std_min else 0.0,
                        abs(pair[1]["p_win_mean"] - 0.5),
                        -pair[1]["p_win_std"],
                        pair[0].physical_state_key(),
                    ),
                )
                chosen_cand, chosen_sc = best_pair
                if chosen_sc["p_win_std"] >= spec.disagreement_std_min:
                    arm = "ensemble_disagreement"
                    reason = f"fresh_disagreement_std={chosen_sc['p_win_std']:.4f}"
                elif spec.boundary_prob_low <= chosen_sc["p_win_mean"] <= spec.boundary_prob_high:
                    arm = "near_boundary"
                    reason = f"fresh_near_boundary_p={chosen_sc['p_win_mean']:.4f}"
                else:
                    arm = "broad_exploration"
                    reason = f"fresh_broad_p={chosen_sc['p_win_mean']:.4f}"
                ctrl_flag = False
            phys = chosen_cand.physical_state_key()
            if phys in chosen_phys:
                continue
            chosen_phys.add(phys)
            worker_selected[worker_id].append(
                _make_plan_row(
                    worker_id=worker_id,
                    round_id=chosen_cand.round_id,
                    sample=chosen_cand,
                    arm=arm,
                    anchor_sample_id=None,
                    reason=reason,
                    is_random_control=ctrl_flag,
                    is_reserve=False,
                    scores=chosen_sc,
                    repeat_count=1,
                    run_seeds=[],
                )
            )
            worker_stratum_counts[(worker_id, stratum)] += 1
            fresh_ctx_assigned_worker[semantic_context_sha256(chosen_cand)] = worker_id

    # Finalize repeat tiers for main worker plans.
    final_worker_plans: dict[str, list[dict[str, Any]]] = {}
    for worker_id in sorted(worker_selected):
        final_worker_plans[worker_id] = _assign_repeat_tiers(
            worker_selected[worker_id], spec.repeat_tier_counts_per_worker, spec.seed, rng_keywords, catalog
        )

    # 3) Generate independent per-worker reserve pools (never sharing contexts or physical states with main plan).
    worker_reserves: dict[str, list[dict[str, Any]]] = {
        worker_id_for_index(i): [] for i in range(spec.num_workers)
    }
    if spec.reserve_rows_per_worker > 0:
        reserve_samples: list[BattleSample] = []
        reserve_attempts = 0
        max_reserve_attempts = max(2000, spec.reserve_rows_per_worker * spec.num_workers * 40)
        reserve_buckets = sorted(available_buckets)
        while any(len(rows) < spec.reserve_rows_per_worker for rows in worker_reserves.values()):
            reserve_attempts += 1
            if reserve_attempts > max_reserve_attempts:
                raise RuntimeError("exceeded max attempts generating S3 reserve pool")
            bucket = reserve_buckets[reserve_attempts % len(reserve_buckets)]
            base, teams = _draw_context_with_teams(
                helper,
                bucket,
                1,
                spec.underfull_team_prob,
                min_owned,
                excluded_team_units=spec.excluded_team_units,
            )
            ctx_sha = semantic_context_sha256(base)
            if ctx_sha in used_fresh_ctx:
                continue
            worker_id = worker_id_for_index(worker_for_context(ctx_sha, spec.num_workers))
            if len(worker_reserves[worker_id]) >= spec.reserve_rows_per_worker:
                continue
            used_fresh_ctx.add(ctx_sha)
            round_id = f"{spec.round_id_prefix}_{worker_id}"
            cand = replace(base, team_unit_ids=teams[0], round_id=round_id, context_id=ctx_sha)
            phys = cand.physical_state_key()
            if phys in chosen_phys:
                continue
            chosen_phys.add(phys)
            cand = replace(cand, sample_id=_sample_id_for_round(round_id, catalog, cand))
            reserve_samples.append(cand)
            # Placeholder row until batch scored below.
            worker_reserves[worker_id].append({"_sample": cand})

        for worker_id in sorted(worker_reserves):
            worker_reserves[worker_id].sort(key=lambda x: x["_sample"].physical_state_key())
        flat_reserve = [row["_sample"] for w in sorted(worker_reserves) for row in worker_reserves[w]]
        reserve_scores = score_samples_with_ensemble(
            ensemble, catalog, flat_reserve, batch_size=spec.batch_size,
            num_workers=spec.num_workers_loader, device=spec.device,
        )
        score_by_sample_id = {
            cand.sample_id: sc for cand, sc in zip(flat_reserve, reserve_scores, strict=True)
        }
        for worker_id in sorted(worker_reserves):
            built: list[dict[str, Any]] = []
            for idx, entry in enumerate(worker_reserves[worker_id]):
                cand = entry["_sample"]
                sc = score_by_sample_id[cand.sample_id]
                phys = cand.physical_state_key()
                seeds = _explicit_seeds("s3_reserve", spec.seed, phys, spec.reserve_repeats)
                built.append(
                    _make_plan_row(
                        worker_id=worker_id,
                        round_id=cand.round_id,
                        sample=cand,
                        arm="reserve_pool",
                        anchor_sample_id=None,
                        reason="authorized_preplanned_reserve",
                        is_random_control=False,
                        is_reserve=True,
                        scores=sc,
                        repeat_count=spec.reserve_repeats,
                        run_seeds=seeds,
                        plan_index=idx,
                    )
                )
            worker_reserves[worker_id] = built

    # Build sidecar summary & verify all invariants before writing.
    arm_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    repeat_counts: Counter[int] = Counter()
    for rows in final_worker_plans.values():
        for row in rows:
            arm_counts[row["arm"]] += 1
            bucket_counts[row["bucket"]] += 1
            repeat_counts[row["repeat_count"]] += 1

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    workers_meta: dict[str, Any] = {}
    for worker_id in sorted(final_worker_plans):
        rows = final_worker_plans[worker_id]
        res_rows = worker_reserves[worker_id]
        plan_path = out_root / f"s3_plan_{worker_id}.jsonl"
        plan_sha = write_jsonl_atomic(plan_path, rows)
        res_path = out_root / f"s3_reserve_{worker_id}.jsonl"
        res_sha = write_jsonl_atomic(res_path, res_rows) if res_rows else None
        workers_meta[worker_id] = {
            "round_id": f"{spec.round_id_prefix}_{worker_id}",
            "plan_file": plan_path.name,
            "plan_sha256": plan_sha,
            "row_count": len(rows),
            "intended_subruns": sum(r["repeat_count"] for r in rows),
            "reserve_file": res_path.name if res_rows else None,
            "reserve_sha256": res_sha,
            "reserve_count": len(res_rows),
            "reserve_intended_subruns": sum(r["repeat_count"] for r in res_rows),
        }

    sidecar = {
        "schema_version": 1,
        "plan_kind": "s3_active_mining",
        "algorithm_version": PLANNER_ALGORITHM_VERSION,
        "db_version": catalog.db_version,
        "db_sha256": db_sha256 or catalog.db_sha256,
        "catalog_sha256": catalog_sha,
        "source_zip_sha256": source_zip_sha256,
        "seal_sha256": seal_sha256 or s2_registry.seal_sha256,
        "split_sha256": s2_registry.split_sha256,
        "s2_denylist_sha256": s2_registry.denylist_sha256,
        "s5_denylist_sha256": s5_denylist.denylist_sha256,
        "checkpoints": ckpt_meta,
        "seed": spec.seed,
        "num_workers": spec.num_workers,
        "excluded_sim_team_units": sorted(spec.excluded_team_units),
        "worker_assignment_rule": f"int(semantic_context_sha256, 16) % {spec.num_workers}",
        "thresholds": {
            "overconfident_win_prob_min": spec.overconfident_win_prob_min,
            "observed_nonwin_max_rate": spec.observed_nonwin_max_rate,
            "disagreement_std_min": spec.disagreement_std_min,
            "boundary_prob_low": spec.boundary_prob_low,
            "boundary_prob_high": spec.boundary_prob_high,
            "random_control_fraction": spec.random_control_fraction,
        },
        "per_worker_bucket_quotas": dict(spec.bucket_quotas_per_worker),
        "per_worker_repeat_tiers": {str(k): v for k, v in sorted(spec.repeat_tier_counts_per_worker.items())},
        "total_rows": sum(m["row_count"] for m in workers_meta.values()),
        "total_intended_subruns": sum(m["intended_subruns"] for m in workers_meta.values()),
        "arm_counts": dict(sorted(arm_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "repeat_counts": {str(k): v for k, v in sorted(repeat_counts.items())},
        "accounting_policy": {
            "max_retries": 0,
            "retry_on_timeout": False,
            "bisect_on_error": False,
            "count_submitted_before_transport": True,
            "reserve_requires_explicit_authorization": True,
        },
        "workers": workers_meta,
    }
    if denied_ctx_set:
        sidecar["extra_denied_contexts_count"] = len(denied_ctx_set)
    if denied_phys_set:
        sidecar["extra_denied_physical_states_count"] = len(denied_phys_set)
    verify_s3_plan(
        sidecar=sidecar,
        worker_plans=final_worker_plans,
        worker_reserves=worker_reserves,
        catalog=catalog,
        s2_registry=s2_registry,
        s5_denylist=s5_denylist,
        existing_physical_states=used_phys,
    )
    sidecar_sha = write_json_atomic(out_root / "s3_plan_sidecar.json", sidecar, indent=2)
    return {
        "sidecar_path": str(out_root / "s3_plan_sidecar.json"),
        "sidecar_sha256": sidecar_sha,
        "sidecar": sidecar,
    }
