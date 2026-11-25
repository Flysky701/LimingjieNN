"""Offline generation and cryptographic freezing of the S5 sealed evaluation panel and denylist."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any

from labyrinth_nn.active_mining.identity import (
    DEFAULT_NUM_WORKERS,
    UNSUPPORTED_SIM_TEAM_UNITS,
    semantic_context_sha256,
    sha256_json,
    worker_for_context,
    worker_id_for_index,
    write_json_atomic,
    write_jsonl_atomic,
)
from labyrinth_nn.catalog.schema import FeatureCatalog
from labyrinth_nn.data.sample_schema import BattleSample
from labyrinth_nn.data.sampler import GOLD_BOUNDARY, OWNED_BOUNDARY, HighScoreSampler, _choice, _preferred_probability

DEFAULT_S5_STRATIFIED_COUNTS: dict[str, int] = {
    "normal": 12000,
    "robot_ex": 4000,
    "special_dual_ex": 4000,
    "other_ex": 4000,
    "boss_3f_d5": 4000,
    "boss_5f_d5": 4000,
}

DEFAULT_S5_PAIRED_CONTEXTS: dict[str, int] = {
    "normal": 300,
    "robot_ex": 250,
    "boss_5f_d5": 120,
    "other_ex": 120,
    "special_dual_ex": 110,
    "boss_3f_d5": 100,
}


@dataclass(frozen=True)
class S5Denylist:
    schema_version: int
    db_version: str
    catalog_sha256: str
    seed: int
    reserved_semantic_contexts: frozenset[str]
    reserved_physical_states: frozenset[str]
    denylist_sha256: str

    def is_denied(self, sample: BattleSample | Mapping[str, Any]) -> bool:
        ctx_sha = semantic_context_sha256(sample)
        if ctx_sha in self.reserved_semantic_contexts:
            return True
        if isinstance(sample, BattleSample):
            return sample.physical_state_key() in self.reserved_physical_states
        identity = (
            str(sample.get("db_version", "")),
            int(sample["quest_id"]),
            int(sample["difficulty"]),
            int(sample["guild_id"]),
            sorted(int(u) for u in sample["team_unit_ids"]),
            sorted(int(u) for u in sample["owned_unit_ids"]),
            sorted(int(r) for r in sample["relic_ids"]),
            None if sample.get("alpha_rupee") is None else int(sample["alpha_rupee"]),
        )
        phys = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode("utf-8")).hexdigest()
        return phys in self.reserved_physical_states

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "db_version": self.db_version,
            "catalog_sha256": self.catalog_sha256,
            "seed": self.seed,
            "reserved_semantic_contexts": sorted(self.reserved_semantic_contexts),
            "reserved_physical_states": sorted(self.reserved_physical_states),
            "denylist_sha256": self.denylist_sha256,
        }

    def save(self, path: str | Path) -> str:
        return write_json_atomic(path, self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> S5Denylist:
        core = {
            "db_version": str(raw["db_version"]),
            "catalog_sha256": str(raw["catalog_sha256"]),
            "seed": int(raw["seed"]),
            "reserved_semantic_contexts": sorted(raw["reserved_semantic_contexts"]),
            "reserved_physical_states": sorted(raw["reserved_physical_states"]),
        }
        computed = sha256_json(core)
        if raw.get("denylist_sha256") != computed:
            raise ValueError("S5 denylist_sha256 mismatch (tampered or corrupt denylist)")
        return cls(
            schema_version=int(raw["schema_version"]),
            db_version=str(raw["db_version"]),
            catalog_sha256=str(raw["catalog_sha256"]),
            seed=int(raw["seed"]),
            reserved_semantic_contexts=frozenset(raw["reserved_semantic_contexts"]),
            reserved_physical_states=frozenset(raw["reserved_physical_states"]),
            denylist_sha256=computed,
        )

    @classmethod
    def load(cls, path: str | Path) -> S5Denylist:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class S5ReservationSpec:
    stratified_bucket_counts: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_S5_STRATIFIED_COUNTS)
    )
    stratified_paired_teams: int = 4
    stratified_repeats: int = 5
    paired_bucket_context_counts: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_S5_PAIRED_CONTEXTS)
    )
    paired_teams_per_context: int = 8
    paired_repeats: int = 11
    underfull_prob_stratified: float = 0.03
    underfull_prob_paired: float = 0.0
    seed: int = 20261155
    num_workers: int = DEFAULT_NUM_WORKERS
    round_id_prefix: str = "s5_sealed"
    excluded_team_units: frozenset[int] = UNSUPPORTED_SIM_TEAM_UNITS


def _sample_id_for_round(round_id: str, catalog: FeatureCatalog, sample: BattleSample) -> str:
    db_hash = catalog.db_sha256 or catalog.catalog_sha256
    identity = (
        round_id,
        db_hash,
        sample.quest_id,
        sample.difficulty,
        sample.guild_id,
        list(sample.team_unit_ids),
        list(sample.owned_unit_ids),
        list(sample.relic_ids),
        sample.alpha_rupee,
    )
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode("utf-8")).hexdigest()


def _explicit_seeds(namespace: str, seed: int, physical_key: str, repeat_count: int) -> list[int]:
    rng = random.Random(f"{namespace}:{seed}:{physical_key}")
    return rng.sample(range(1, 2**31), repeat_count)


def _draw_context_with_teams(
    helper: HighScoreSampler,
    bucket: str,
    teams_per_context: int,
    underfull_prob: float,
    min_owned_for_teams: int,
    *,
    excluded_team_units: frozenset[int] = UNSUPPORTED_SIM_TEAM_UNITS,
) -> tuple[BattleSample, list[list[int]]]:
    empty = bool(helper.early_quests[bucket]) and helper.rng.random() < min(
        1.0, helper.boundary["empty_relic_early"] / max(1e-9, helper.early_mass)
    )
    quest = helper._quest(bucket, empty)
    guild = helper.guilds[_choice(helper.rng, helper.guild_weights)]
    state = helper._ranges(quest.area)
    relic_count = helper.rng.randint(int(state.get("relic_min", 0)), int(state.get("relic_max", 8)))
    relic_ids = helper._relics(guild, relic_count, empty)
    owned_count = helper.rng.randint(int(state.get("owned_min", 8)), int(state.get("owned_max", 30)))
    if helper.rng.random() < _preferred_probability(
        helper.boundary["owned_25_boundary"], helper.owned_guild_mass, guild.guild_id in helper.owned_guilds
    ):
        owned_count = helper.rng.choice(OWNED_BOUNDARY)
    gold = helper.rng.randint(int(state.get("gold_min", 0)), int(state.get("gold_max", 12000)))
    if helper.rng.random() < _preferred_probability(
        helper.boundary["gold_10k_boundary"], helper.gold_guild_mass, guild.guild_id in helper.gold_guilds
    ):
        gold = helper.rng.choice(GOLD_BOUNDARY)

    first_mode = _choice(helper.rng, helper.team_weights)
    prior = [uid for uid in helper._prior(first_mode, quest) if uid not in excluded_team_units]
    required = (
        [guild.guild_master_id]
        if guild.guild_master_id in helper.catalog.units and guild.guild_master_id not in excluded_team_units
        else []
    )
    required = list(dict.fromkeys(required + prior))
    playable_pool = [uid for uid in helper.unit_ids if uid not in excluded_team_units]
    if len(playable_pool) < min_owned_for_teams:
        raise ValueError(
            f"catalog has too few playable units ({len(playable_pool)}) for min_owned_for_teams={min_owned_for_teams}"
        )
    owned_count = min(len(helper.unit_ids), max(min_owned_for_teams, len(required), owned_count))
    playable_needed = max(0, min_owned_for_teams - len(required))
    playable_picks = required + helper.rng.sample(
        [uid for uid in playable_pool if uid not in required], playable_needed
    )
    remaining_pool = [uid for uid in helper.unit_ids if uid not in playable_picks]
    owned = sorted(
        playable_picks
        + helper.rng.sample(remaining_pool, min(len(remaining_pool), max(0, owned_count - len(playable_picks))))
    )
    playable_owned = [uid for uid in owned if uid not in excluded_team_units]
    roles = Counter(helper.catalog.units[uid].role_id for uid in owned)
    talents = Counter(helper.catalog.units[uid].talent_id for uid in owned)
    teams: list[list[int]] = []
    seen_teams: set[tuple[int, ...]] = set()
    saved_underfull = helper.underfull_prob
    helper.underfull_prob = underfull_prob
    try:
        for team_idx in range(teams_per_context):
            mode = first_mode if team_idx == 0 else _choice(helper.rng, helper.team_weights)
            filtered_prior = [uid for uid in helper._prior(mode, quest) if uid not in excluded_team_units]
            team = helper._team(mode, quest, playable_owned, filtered_prior)
            if tuple(team) in seen_teams:
                size = len(team)
                if sum(len(t) == size for t in seen_teams) >= comb(len(playable_owned), size):
                    size = min(5, len(playable_owned))
                for _ in range(24):
                    candidate = helper._sort_team(helper.rng.sample(playable_owned, size))
                    if tuple(candidate) not in seen_teams:
                        team = candidate
                        break
                else:
                    team = list(
                        next(
                            t
                            for t in combinations(helper._sort_team(playable_owned), size)
                            if t not in seen_teams
                        )
                    )
            if excluded_team_units & set(team):
                raise ValueError(f"sampled team contains excluded unit(s): {sorted(excluded_team_units & set(team))}")
            seen_teams.add(tuple(team))
            teams.append(team)
    finally:
        helper.underfull_prob = saved_underfull

    base = BattleSample(
        sample_id="",
        round_id="",
        source="sim",
        db_version=helper.catalog.db_version,
        quest_id=quest.quest_id,
        canonical_wave_sig=quest.canonical_wave_sig,
        bucket=quest.bucket,
        area=quest.area,
        difficulty=5,
        guild_id=guild.guild_id,
        team_unit_ids=teams[0],
        owned_unit_ids=owned,
        owned_unit_count=owned_count,
        owned_role_counts=[roles[i] for i in range(1, 9)],
        owned_talent_counts=[talents[i] for i in range(1, 6)],
        relic_ids=relic_ids,
        mark_stacks=helper.catalog.mark_stacks_for_relics(relic_ids),
        alpha_rupee=gold,
        revive_available=any(helper.catalog.relics[rid].is_revive for rid in relic_ids),
        context_id="",
    )
    return base, teams


def build_s5_reservation(
    catalog: FeatureCatalog,
    profile: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
    *,
    denied_semantic_contexts: Iterable[str] = (),
    denied_physical_states: Iterable[str] = (),
    spec: S5ReservationSpec | None = None,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Generate frozen S5 stratified + paired evaluation plans and S5 denylist offline."""
    spec = spec or S5ReservationSpec()
    max_teams = max(spec.stratified_paired_teams, spec.paired_teams_per_context)
    playable_count = sum(1 for uid in catalog.units if uid not in spec.excluded_team_units)
    min_owned = next((n for n in range(5, playable_count + 1) if comb(n, 5) >= max_teams), None)
    if min_owned is None:
        raise ValueError(f"catalog has too few playable units ({playable_count}) for {max_teams} distinct teams/context")

    cfg = dict(sampling_config)
    cfg["seed"] = spec.seed
    cfg["round_id"] = spec.round_id_prefix
    cfg["paired_teams_per_context"] = 1
    helper = HighScoreSampler(catalog, profile, cfg)

    catalog_sha = catalog.catalog_sha256 or catalog.compute_sha256()
    blocked_ctx = set(denied_semantic_contexts)
    blocked_phys = set(denied_physical_states)
    s5_ctx: set[str] = set()
    s5_phys: set[str] = set()
    worker_rows: dict[str, list[dict[str, Any]]] = {
        worker_id_for_index(idx): [] for idx in range(spec.num_workers)
    }
    section_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()

    def append_panel(section: str, bucket: str, target_contexts: int, teams_per_ctx: int, repeats: int, underfull: float):
        accepted_ctx = 0
        attempts = 0
        max_attempts = max(2000, target_contexts * 50)
        while accepted_ctx < target_contexts:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"exceeded max attempts generating S5 section={section} bucket={bucket}")
            base, teams = _draw_context_with_teams(
                helper,
                bucket,
                teams_per_ctx,
                underfull,
                min_owned,
                excluded_team_units=spec.excluded_team_units,
            )
            ctx_sha = semantic_context_sha256(base)
            if ctx_sha in blocked_ctx or ctx_sha in s5_ctx:
                continue
            worker_idx = worker_for_context(ctx_sha, spec.num_workers)
            worker_id = worker_id_for_index(worker_idx)
            round_id = f"{spec.round_id_prefix}_{worker_id}"
            candidates: list[tuple[BattleSample, str]] = []
            collision = False
            for team in teams:
                if spec.excluded_team_units & set(team):
                    raise ValueError(f"S5 team contains excluded simulator unit(s): {team}")
                row_sample = replace(
                    base,
                    team_unit_ids=team,
                    round_id=round_id,
                    context_id=ctx_sha,
                )
                phys = row_sample.physical_state_key()
                if phys in blocked_phys or phys in s5_phys:
                    collision = True
                    break
                row_sample = replace(row_sample, sample_id=_sample_id_for_round(round_id, catalog, row_sample))
                candidates.append((row_sample, phys))
            if collision:
                continue
            s5_ctx.add(ctx_sha)
            for row_sample, phys in candidates:
                s5_phys.add(phys)
                seeds = _explicit_seeds("s5", spec.seed, phys, repeats)
                plan_row = {
                    "plan_index": len(worker_rows[worker_id]),
                    "worker_id": worker_id,
                    "round_id": round_id,
                    "sample_id": row_sample.sample_id,
                    "semantic_context_sha256": ctx_sha,
                    "physical_state_key": phys,
                    "arm": f"s5_{section}",
                    "panel_section": section,
                    "anchor_sample_id": None,
                    "reason": f"s5_reserved_{section}",
                    "is_random_control": section == "stratified_32k",
                    "is_reserve": False,
                    "bucket": row_sample.bucket,
                    "scores": None,
                    "repeat_count": repeats,
                    "run_seeds": seeds,
                    "payload": {
                        "db_version": row_sample.db_version,
                        "quest_id": row_sample.quest_id,
                        "canonical_wave_sig": row_sample.canonical_wave_sig,
                        "bucket": row_sample.bucket,
                        "area": row_sample.area,
                        "difficulty": row_sample.difficulty,
                        "guild_id": row_sample.guild_id,
                        "team_unit_ids": row_sample.team_unit_ids,
                        "owned_unit_ids": row_sample.owned_unit_ids,
                        "owned_unit_count": row_sample.owned_unit_count,
                        "owned_role_counts": row_sample.owned_role_counts,
                        "owned_talent_counts": row_sample.owned_talent_counts,
                        "relic_ids": row_sample.relic_ids,
                        "mark_stacks": row_sample.mark_stacks,
                        "alpha_rupee": row_sample.alpha_rupee,
                        "revive_available": row_sample.revive_available,
                        "context_id": ctx_sha,
                    },
                }
                worker_rows[worker_id].append(plan_row)
                section_counts[section] += 1
                bucket_counts[row_sample.bucket] += 1
            accepted_ctx += 1

    for bucket, row_target in sorted(spec.stratified_bucket_counts.items()):
        if row_target <= 0:
            continue
        if row_target % spec.stratified_paired_teams != 0:
            raise ValueError(
                f"stratified row target {row_target} for {bucket} must be divisible by "
                f"stratified_paired_teams={spec.stratified_paired_teams}"
            )
        append_panel(
            "stratified_32k",
            bucket,
            row_target // spec.stratified_paired_teams,
            spec.stratified_paired_teams,
            spec.stratified_repeats,
            spec.underfull_prob_stratified,
        )

    for bucket, ctx_target in sorted(spec.paired_bucket_context_counts.items()):
        if ctx_target <= 0:
            continue
        append_panel(
            "paired_1000x8",
            bucket,
            ctx_target,
            spec.paired_teams_per_context,
            spec.paired_repeats,
            spec.underfull_prob_paired,
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    denylist_core = {
        "db_version": catalog.db_version,
        "catalog_sha256": catalog_sha,
        "seed": spec.seed,
        "reserved_semantic_contexts": sorted(s5_ctx),
        "reserved_physical_states": sorted(s5_phys),
    }
    denylist = S5Denylist(
        schema_version=1,
        db_version=catalog.db_version,
        catalog_sha256=catalog_sha,
        seed=spec.seed,
        reserved_semantic_contexts=frozenset(s5_ctx),
        reserved_physical_states=frozenset(s5_phys),
        denylist_sha256=sha256_json(denylist_core),
    )
    denylist_path = out_root / "s5_denylist.json"
    denylist_file_sha = denylist.save(denylist_path)

    workers_meta: dict[str, Any] = {}
    for worker_id, rows in sorted(worker_rows.items()):
        plan_path = out_root / f"s5_plan_{worker_id}.jsonl"
        plan_sha = write_jsonl_atomic(plan_path, rows)
        workers_meta[worker_id] = {
            "round_id": f"{spec.round_id_prefix}_{worker_id}",
            "plan_file": plan_path.name,
            "plan_sha256": plan_sha,
            "row_count": len(rows),
            "intended_subruns": sum(r["repeat_count"] for r in rows),
            "reserve_file": None,
            "reserve_sha256": None,
            "reserve_count": 0,
        }

    sidecar = {
        "schema_version": 1,
        "plan_kind": "s5_sealed_panel",
        "algorithm_version": "s5_reservation_v1",
        "db_version": catalog.db_version,
        "catalog_sha256": catalog_sha,
        "seed": spec.seed,
        "num_workers": spec.num_workers,
        "excluded_sim_team_units": sorted(spec.excluded_team_units),
        "s5_denylist_file": denylist_path.name,
        "s5_denylist_sha256": denylist.denylist_sha256,
        "s5_denylist_file_sha256": denylist_file_sha,
        "total_rows": sum(m["row_count"] for m in workers_meta.values()),
        "total_intended_subruns": sum(m["intended_subruns"] for m in workers_meta.values()),
        "total_unique_contexts": len(s5_ctx),
        "section_counts": dict(sorted(section_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "workers": workers_meta,
    }
    sidecar_sha = write_json_atomic(out_root / "s5_plan_sidecar.json", sidecar, indent=2)
    return {
        "sidecar_path": str(out_root / "s5_plan_sidecar.json"),
        "sidecar_sha256": sidecar_sha,
        "denylist_path": str(denylist_path),
        "denylist_sha256": denylist.denylist_sha256,
        "sidecar": sidecar,
    }
