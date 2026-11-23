"""Plan and collect difficulty-five high-score battle samples."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from heapq import nlargest
from itertools import combinations, islice
from math import comb
from pathlib import Path
from typing import Any

from labyrinth_nn.catalog.schema import MARK_THRESHOLDS, FeatureCatalog, GuildEntry, QuestEntry
from labyrinth_nn.data.sample_schema import BattleSample, OUTCOME_WIN, aggregate_runs, read_samples, write_samples
from labyrinth_nn.data.sim_client import (
    SimClient, SimulationConfig, SimulationError, SimulationTimeoutError, environment_enemy_ids,
)

QUEST_BUCKETS = ("normal", "robot_ex", "special_dual_ex", "other_ex", "boss_3f_d5", "boss_5f_d5")
TEAM_MODES = ("prior", "counter_stratified", "perturb", "random")
OWNED_BOUNDARY = (24, 25, 26)
GOLD_BOUNDARY = (9999, 10000, 10001)
UNSUPPORTED_LINK = {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS}


def _identity_hash(identity: tuple) -> str:
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode("utf-8")).hexdigest()


def quest_stratum(quest: QuestEntry) -> str | None:
    if quest.quest_type in {1, 2} and quest.bucket in QUEST_BUCKETS[:4]:
        return quest.bucket
    if quest.quest_type == 3 and quest.is_difficulty5_boss:
        if quest.bucket == "boss_3f_d5":
            return quest.bucket
        if re.fullmatch(r"boss_\d+f_d5", quest.bucket):
            return "boss_5f_d5"
    return None


def _weights(keys: Iterable[Any], raw: Mapping[Any, Any]) -> dict[Any, float]:
    weights = {key: float(raw.get(key, 0) if raw else 1) for key in keys}
    if any(not 0 <= weight < float("inf") for weight in weights.values()):
        raise ValueError("sampling weights must be finite and nonnegative")
    total = sum(weights.values())
    if not 0 < total < float("inf"):
        raise ValueError("no eligible entries with a positive sampling weight")
    return {key: weight / total for key, weight in weights.items() if weight > 0}


def _choice(rng: random.Random, weights: Mapping[Any, float]) -> Any:
    return rng.choices(tuple(weights), tuple(weights.values()), k=1)[0]


def _preferred_probability(rate: float, preferred_mass: float, preferred: bool) -> float:
    if preferred:
        return min(1.0, rate / preferred_mass)
    return max(0.0, (rate - preferred_mass) / (1.0 - preferred_mass))


class HighScoreSampler:
    def __init__(
        self, catalog: FeatureCatalog, profile: Mapping[str, Any], config: Mapping[str, Any]
    ):
        self.catalog = catalog
        self.profile = dict(profile)
        self.config = dict(config)
        self.round_id = str(config.get("round_id", "round0_pilot"))
        self.seed = int(config.get("seed", 20261122))
        self.rng = random.Random(self.seed)
        if int(config.get("difficulty", profile.get("difficulty", 5))) != 5 or catalog.max_difficulty < 5:
            raise ValueError("high-score sampling requires difficulty 5")
        if len(catalog.units) < 5:
            raise ValueError("sampling requires at least five catalog units")
        self.unit_ids = catalog.sorted_unit_ids()
        self.paired_teams = int(config.get("paired_teams_per_context", 1))
        self.underfull_prob = float(config.get("underfull_team_prob", 0.0))
        self._pending_teams: list[tuple[BattleSample, str]] = []
        if self.paired_teams < 1 or not 0 <= self.underfull_prob <= 1:
            raise ValueError("invalid paired team count or underfull team probability")
        self.min_owned = next((n for n in range(5, len(self.unit_ids) + 1)
                               if comb(n, 5) >= self.paired_teams), None)
        if self.min_owned is None:
            raise ValueError("catalog cannot supply enough distinct five-unit teams per context")
        environment = environment_enemy_ids(catalog, profile)
        self.quest_enemies = {
            qid: [catalog.enemies[eid] for eid in quest.enemy_ids if eid not in environment and eid in catalog.enemies]
            for qid, quest in catalog.quests.items()
        }
        self.quests: dict[str, list[QuestEntry]] = defaultdict(list)
        for qid in catalog.sorted_quest_ids():
            quest = catalog.quests[qid]
            bucket = quest_stratum(quest)
            if bucket and quest.wave_total_hp > 0 and quest.limit_time > 0 and self.quest_enemies[qid]:
                self.quests[bucket].append(quest)
        self.quest_weights = _weights(
            (bucket for bucket in QUEST_BUCKETS if bucket in self.quests), config.get("quest_type_weights", {})
        )
        first_area = min(q.area for bucket in self.quest_weights for q in self.quests[bucket])
        self.early_quests = {
            bucket: [q for q in self.quests[bucket] if q.area == first_area] for bucket in self.quest_weights
        }
        self.early_mass = sum(self.quest_weights[b] for b, quests in self.early_quests.items() if quests)
        self.quest_decks: dict[tuple[str, bool], list[QuestEntry]] = {}
        self.team_weights = _weights(TEAM_MODES, config.get("team_mix_weights", {}))
        self.guilds = dict(sorted(catalog.guilds.items()))
        if not self.guilds:
            raise ValueError("sampling requires at least one catalog guild")
        self.guild_weights = _weights(
            self.guilds, {int(k): v for k, v in config.get("guild_weights", {}).items()}
        )
        self.guild_relics = {
            gid: guild.add_relic_id for gid, guild in self.guilds.items() if guild.add_relic_id in catalog.relics
        }
        guild_only = {guild.add_relic_id for guild in self.guilds.values()}
        self.relic_pool = [
            rid for rid in catalog.sorted_relic_ids()
            if rid not in guild_only and not (40000 < rid < 41000 or 50000 < rid < 51000)
        ]
        self.mark_relics = {
            mark: [rid for rid in self.relic_pool if catalog.relics[rid].relic_mark_id == mark
                   and catalog.relics[rid].relic_mark_count > 0]
            for mark in range(1, 6)
        }
        self.owned_guilds = self._scaling_guilds("owned_unit_scaling")
        self.gold_guilds = self._scaling_guilds("gold_scaling")
        self.owned_guild_mass = sum(self.guild_weights.get(gid, 0) for gid in self.owned_guilds)
        self.gold_guild_mass = sum(self.guild_weights.get(gid, 0) for gid in self.gold_guilds)
        self.boundary = {
            name: float(config.get("boundary_probe_probabilities", {}).get(name, default))
            for name, default in (
                ("mark_threshold_focus", 0.40), ("owned_25_boundary", 0.20),
                ("gold_10k_boundary", 0.25), ("empty_relic_early", 0.05),
            )
        }
        if any(not 0 <= prob <= 1 for prob in self.boundary.values()):
            raise ValueError("boundary probe probabilities must lie in [0, 1]")
        state = config.get("area_state_distributions", {})
        self.area_states = {int(area): distribution for area, distribution in state.items()}
        self.priors = config.get("team_priors", {})
        repeat = profile.get("adaptive_repeat", {})
        self.initial_runs = int(repeat.get("initial_runs", 1))
        self.borderline_runs = int(repeat.get("borderline_runs", 6))
        self.min_win_rate = float(repeat.get("borderline_min_win_rate", 0.15))
        self.max_win_rate = float(repeat.get("borderline_max_win_rate", 0.85))
        self.close_loss_damage = repeat.get("close_loss_damage_threshold")
        self.close_win_time = repeat.get("close_win_rem_time_threshold")
        self.random_repeat_prob = float(repeat.get("random_repeat_prob", 0.0))
        self.rng_keywords = tuple(str(word) for word in repeat.get(
            "high_rng_keywords", ("反物质兽", "狙击·遗物", "黄蜂女王", "美空")
        ) if word)
        if not (1 <= self.initial_runs <= self.borderline_runs
                and 0 <= self.min_win_rate <= self.max_win_rate <= 1
                and 0 <= self.random_repeat_prob <= 1):
            raise ValueError("invalid adaptive repeat settings")

    def _scaling_guilds(self, field: str) -> set[int]:
        return {
            gid for gid, rid in self.guild_relics.items()
            if getattr(self.catalog.relics[rid], field)
        }

    def _quest(self, bucket: str, empty: bool) -> QuestEntry:
        deck = self.quest_decks.setdefault((bucket, empty), [])
        if not deck:
            deck.extend(self.early_quests[bucket] if empty else self.quests[bucket])
            self.rng.shuffle(deck)
        return deck.pop()

    def _mark_probe(self, base: list[int]) -> tuple[int, tuple[int, ...]] | None:
        # Subset sums over distinct relics let us target an exact transition even
        # when each relic contributes several stacks.
        stacks = self.catalog.mark_stacks_for_relics(base)
        threshold = self.rng.choice(MARK_THRESHOLDS)
        offset = self.rng.choice((0, 0, 0, 0, -1, 1))
        for target in dict.fromkeys((threshold + offset, threshold)):
            for mark in self.rng.sample(range(1, 6), 5):
                missing = target - stacks[mark - 1]
                if missing < 0:
                    continue
                relics = self.mark_relics[mark].copy()
                self.rng.shuffle(relics)
                sums: dict[int, tuple[int, ...]] = {0: ()}
                for rid in relics:
                    width = self.catalog.relics[rid].relic_mark_count
                    for total, combo in tuple(sums.items()):
                        if total + width <= missing:
                            sums.setdefault(total + width, (*combo, rid))
                    if missing in sums:
                        break
                if missing in sums:
                    return mark, sums[missing]
        return None

    def _relics(self, guild: GuildEntry, count: int, empty: bool) -> list[int]:
        if empty:
            return []
        relics = [self.guild_relics[guild.guild_id]] if guild.guild_id in self.guild_relics else []
        optional = 40000 + guild.guild_id
        if optional in self.catalog.relics and count > len(relics) and self.rng.random() < 0.5:
            relics.append(optional)
        pool = self.relic_pool
        if self.rng.random() < self.boundary["mark_threshold_focus"]:
            probe = self._mark_probe(relics)
            if probe is not None:
                mark, picks = probe
                relics.extend(picks)
                pool = [rid for rid in pool if self.catalog.relics[rid].relic_mark_id != mark]
        relics.extend(self.rng.sample(pool, min(len(pool), max(0, count - len(relics)))))
        return sorted(relics)

    def _ranges(self, area: int) -> Mapping[str, Any]:
        if not self.area_states:
            return {}
        return self.area_states.get(area, self.area_states[min(self.area_states, key=lambda a: abs(a - area))])

    def _team_prior(self, quest: QuestEntry, owned: list[int], prior: list[int]) -> list[int]:
        units = self.catalog.units
        team = [uid for uid in prior if uid in owned][:5]
        if not team:
            team.append(max(owned, key=lambda uid: (
                units[uid].panel_def + units[uid].panel_magic_def, -units[uid].search_area_width
            )))
            sustain = [uid for uid in owned if uid not in team and units[uid].mech_flags.get("heal_or_regen")]
            if sustain:
                team.append(max(sustain, key=lambda uid: units[uid].panel_hp))
        enemies = self.quest_enemies[quest.quest_id]
        attack = "panel_atk" if sum(e.physical_def for e in enemies) <= sum(
            e.magic_def for e in enemies
        ) else "panel_magic_str"
        team.extend(sorted((uid for uid in owned if uid not in team),
                           key=lambda uid: getattr(units[uid], attack), reverse=True)[:5 - len(team)])
        return team

    def _team(self, mode: str, quest: QuestEntry, owned: list[int], prior: list[int]) -> list[int]:
        units = self.catalog.units
        if mode == "random":
            team = self.rng.sample(owned, 5)
        elif mode == "counter_stratified":
            team = self._team_prior(quest, owned, [])
            enemies = self.quest_enemies[quest.quest_id]
            flags = ["barrier_shield", "heal_or_regen", "dispel", "tp_charge"]
            if any(not e.has_dot_resist for e in enemies):
                flags.append("dot_slip")
            if any(not e.has_cc_resist for e in enemies):
                flags.append("cc_abnormal")
            if len(enemies) > 1:
                flags.append("aoe_dmg")
            candidates = [
                [uid for uid in owned if uid not in team and units[uid].mech_flags.get(flag)] for flag in flags
            ]
            available = [ids for ids in candidates if ids]
            if available:
                team[-1] = self.rng.choice(self.rng.choice(available))
        else:
            team = self._team_prior(quest, owned, prior)
            if mode == "perturb":
                others = [uid for uid in owned if uid not in team]
                count = min(len(others), self.rng.randint(1, 2))
                for slot, uid in zip(self.rng.sample(range(5), count), self.rng.sample(others, count)):
                    team[slot] = uid
        if self.underfull_prob and len(owned) >= 3 and self.rng.random() < self.underfull_prob:
            team = self.rng.sample(team, self.rng.choice((3, 4)))
        return self._sort_team(team)

    def _sort_team(self, team: Iterable[int]) -> list[int]:
        return sorted(team, key=lambda uid: (self.catalog.units[uid].search_area_width, uid))

    def _sample_id(self, context: tuple, team: list[int]) -> str:
        db_hash, quest_id, difficulty, guild_id, owned, relics, gold = context
        return _identity_hash((self.round_id, db_hash, quest_id, difficulty, guild_id, team, owned, relics, gold))

    def _prior(self, mode: str, quest: QuestEntry) -> list[int]:
        if mode not in {"prior", "perturb"}:
            return []
        ids = self.priors.get(str(quest.quest_id), self.priors.get(quest.bucket, ()))
        return list(dict.fromkeys(uid for value in ids if (uid := int(value)) in self.catalog.units))[:5]

    def _context_identity(self, quest_id, guild_id, owned, relics, gold) -> tuple:
        return (self.catalog.db_sha256 or self.catalog.catalog_sha256, quest_id, 5, guild_id, owned, relics, gold)

    def draw(self) -> tuple[BattleSample, str]:
        """Draw one state and report the selected team policy for planning statistics."""
        if self.paired_teams == 1:
            return self._draw_first_team()
        if not self._pending_teams:
            self._pending_teams = self._draw_context()[::-1]
        return self._pending_teams.pop()

    def _draw_first_team(self) -> tuple[BattleSample, str]:
        bucket = _choice(self.rng, self.quest_weights)
        empty = bool(self.early_quests[bucket]) and self.rng.random() < min(
            1.0, self.boundary["empty_relic_early"] / self.early_mass
        )
        quest = self._quest(bucket, empty)
        guild = self.guilds[_choice(self.rng, self.guild_weights)]
        mode = _choice(self.rng, self.team_weights)
        state = self._ranges(quest.area)
        relic_count = self.rng.randint(int(state.get("relic_min", 0)), int(state.get("relic_max", 8)))
        relic_ids = self._relics(guild, relic_count, empty)
        owned_count = self.rng.randint(int(state.get("owned_min", 8)), int(state.get("owned_max", 30)))
        if self.rng.random() < _preferred_probability(
            self.boundary["owned_25_boundary"], self.owned_guild_mass, guild.guild_id in self.owned_guilds
        ):
            owned_count = self.rng.choice(OWNED_BOUNDARY)
        gold = self.rng.randint(int(state.get("gold_min", 0)), int(state.get("gold_max", 12000)))
        if self.rng.random() < _preferred_probability(
            self.boundary["gold_10k_boundary"], self.gold_guild_mass, guild.guild_id in self.gold_guilds
        ):
            gold = self.rng.choice(GOLD_BOUNDARY)

        prior = self._prior(mode, quest)
        required = [guild.guild_master_id] if guild.guild_master_id in self.catalog.units else []
        required = list(dict.fromkeys(required + prior))
        owned_count = min(len(self.unit_ids), max(self.min_owned, len(required), owned_count))
        owned = sorted(required + self.rng.sample(
            [uid for uid in self.unit_ids if uid not in required], owned_count - len(required)
        ))
        team = self._team(mode, quest, owned, prior)
        roles = Counter(self.catalog.units[uid].role_id for uid in owned)
        talents = Counter(self.catalog.units[uid].talent_id for uid in owned)
        context = self._context_identity(quest.quest_id, guild.guild_id, owned, relic_ids, gold)
        return BattleSample(
            sample_id=self._sample_id(context, team), round_id=self.round_id, source="sim",
            db_version=self.catalog.db_version,
            quest_id=quest.quest_id, canonical_wave_sig=quest.canonical_wave_sig, bucket=quest.bucket,
            area=quest.area, difficulty=5, guild_id=guild.guild_id,
            team_unit_ids=team, owned_unit_ids=owned, owned_unit_count=owned_count,
            owned_role_counts=[roles[i] for i in range(1, 9)],
            owned_talent_counts=[talents[i] for i in range(1, 6)],
            relic_ids=relic_ids, mark_stacks=self.catalog.mark_stacks_for_relics(relic_ids),
            alpha_rupee=gold,
            revive_available=any(self.catalog.relics[rid].is_revive for rid in relic_ids),
            context_id=_identity_hash(context) if self.paired_teams > 1 else "",
        ), mode

    def _draw_context(self) -> list[tuple[BattleSample, str]]:
        first, mode = self._draw_first_team()
        group = [(first, mode)]
        quest = self.catalog.quests[first.quest_id]
        owned = first.owned_unit_ids
        context = self._context_identity(first.quest_id, first.guild_id, owned, first.relic_ids, first.alpha_rupee)
        teams = {tuple(first.team_unit_ids)}
        for _ in range(self.paired_teams - 1):
            mode = _choice(self.rng, self.team_weights)
            team = self._team(mode, quest, owned, self._prior(mode, quest))
            if tuple(team) in teams:
                mode = "random"
                size = len(team)
                # Preserve team size if that size still has unused combinations.
                if sum(len(t) == size for t in teams) == comb(len(owned), size):
                    size = 5
                for _ in range(16):
                    team = self._sort_team(self.rng.sample(owned, size))
                    if tuple(team) not in teams:
                        break
                else:
                    team = list(next(t for t in combinations(self._sort_team(owned), size) if t not in teams))
            teams.add(tuple(team))
            group.append((replace(first, team_unit_ids=team, sample_id=self._sample_id(context, team)), mode))
        return group

    def _unique(self, count: int, seen: set[str]) -> Iterator[tuple[BattleSample, str]]:
        duplicates = 0
        while count:
            sample, mode = self.draw()
            if sample.sample_id in seen:
                duplicates += 1
                if duplicates >= max(1000, len(seen) * 2):
                    raise ValueError("cannot find enough distinct battle states for the requested target")
                continue
            duplicates = 0
            seen.add(sample.sample_id)
            count -= 1
            yield sample, mode

    def dry_run(self, count: int) -> dict[str, Any]:
        if count < 0:
            raise ValueError("sample count must be nonnegative")
        strata: Counter[str] = Counter()
        buckets: Counter[str] = Counter()
        quests: Counter[int] = Counter()
        modes: Counter[str] = Counter()
        guilds: Counter[int] = Counter()
        marks: Counter[int] = Counter()
        mark_neighborhoods: Counter[int] = Counter()
        owned: Counter[int] = Counter()
        gold: Counter[int] = Counter()
        empty = 0
        for sample, mode in self._unique(count, set()):
            strata[quest_stratum(self.catalog.quests[sample.quest_id])] += 1
            buckets[sample.bucket] += 1
            quests[sample.quest_id] += 1
            modes[mode] += 1
            guilds[sample.guild_id] += 1
            stacks = set(sample.mark_stacks)
            marks.update(t for t in MARK_THRESHOLDS if t in stacks)
            mark_neighborhoods.update(t for t in MARK_THRESHOLDS if any(abs(v - t) <= 1 for v in stacks))
            owned[sample.owned_unit_count] += 1
            gold[sample.alpha_rupee] += 1
            empty += int(not sample.relic_ids)
        return {
            "round_id": self.round_id, "difficulty": 5, "planned_battles": count,
            "strata": dict(strata), "buckets": dict(buckets), "unique_quests": len(quests),
            "quest_counts": dict(sorted(quests.items())), "team_modes": dict(modes), "guilds": dict(guilds),
            "boundaries": {
                "mark_thresholds": {t: marks[t] for t in MARK_THRESHOLDS},
                "mark_neighborhoods": {t: mark_neighborhoods[t] for t in MARK_THRESHOLDS},
                "owned_25": {n: owned[n] for n in OWNED_BOUNDARY},
                "gold_10k": {n: gold[n] for n in GOLD_BOUNDARY},
                "empty_relic_early": empty,
            },
        }

    def _simulation_config(self, sample: BattleSample, start: int, stop: int) -> SimulationConfig:
        seeds = None
        if self.profile.get("use_fixed_random_seed", False):
            rng = random.Random(f"{self.profile.get('random_seed', 666)}:{sample.sample_id}")
            seeds = rng.sample(range(1, 2**31), self.borderline_runs)[start:stop]
        return SimulationConfig(
            self.catalog.quests[sample.quest_id], sample.team_unit_ids,
            owned_unit_ids=sample.owned_unit_ids, relic_ids=sample.relic_ids,
            alpha_rupee=sample.alpha_rupee, difficulty=sample.difficulty, runs=stop - start,
            run_seeds=seeds,
        )

    def _needs_repeat(self, sample: BattleSample) -> bool:
        names = [self.catalog.quests[sample.quest_id].quest_name]
        names.extend(enemy.name for enemy in self.quest_enemies[sample.quest_id])
        win_rate = sum(run.outcome_class == OUTCOME_WIN for run in sample.runs) / len(sample.runs)
        return self.min_win_rate <= win_rate <= self.max_win_rate or any(
            word in name for name in names for word in self.rng_keywords
        ) or any(
            (self.close_loss_damage is not None and run.outcome_class != OUTCOME_WIN
             and run.damage_ratio >= self.close_loss_damage)
            or (self.close_win_time is not None and run.outcome_class == OUTCOME_WIN
                and run.rem_time_ratio <= self.close_win_time)
            for run in sample.runs
        ) or (
            self.random_repeat_prob > 0
            and int.from_bytes(hashlib.sha256(sample.sample_id.encode("utf-8")).digest()[:8])
            < self.random_repeat_prob * 2**64
        )

    def simulate_samples(self, client: SimClient, samples: list[BattleSample]) -> list[BattleSample]:
        groups = client.simulate_batch_single_runs(
            [self._simulation_config(sample, 0, self.initial_runs) for sample in samples]
        )
        for sample, runs in zip(samples, groups, strict=True):
            sample.runs = list(runs)
        borderline = (
            [sample for sample in samples if self._needs_repeat(sample)]
            if self.initial_runs < self.borderline_runs else []
        )
        if borderline:
            extra = client.simulate_batch_single_runs([
                self._simulation_config(sample, self.initial_runs, self.borderline_runs)
                for sample in borderline
            ])
            for sample, runs in zip(borderline, extra, strict=True):
                sample.runs.extend(runs)
        for sample in samples:
            sample.outcome_counts, sample.mean_damage_ratio, sample.mean_win_rem_time = aggregate_runs(sample.runs)
        return samples

    def collect(
        self, client: SimClient, output: str | Path, target: int, *, resume: bool = False,
        failure_log: str | Path | None = None, sync_dir: str | Path | None = None,
        max_seconds: float | None = None, progress_stderr: bool = False,
    ) -> dict[str, Any]:
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        batch_size = int(self.config.get("batch_size", 32))
        chunk_size = int(self.config.get("chunk_size", 5000))
        concurrency = int(self.config.get("concurrency", 1))
        if target < 0 or min(batch_size, chunk_size, concurrency) < 1:
            raise ValueError("target must be nonnegative and batch_size, chunk_size, concurrency positive")
        path = Path(output)
        if path.suffix not in {".jsonl", ".parquet"}:
            raise ValueError("--out must end in .jsonl or .parquet")
        shards = _existing_shards(path)
        if shards and not resume:
            raise FileExistsError(f"output already exists, use --resume: {path}")
        isolate = bool(self.config.get("isolate_batch_errors", False))
        failed_path = Path(failure_log) if isolate and failure_log is not None else None
        seen: set[str] = set()
        entries = []
        existing_subruns = 0
        for shard in shards.values():
            rows = read_samples(shard)
            for sample in rows:
                if (sample.round_id, sample.db_version, sample.source) != (
                    self.round_id, self.catalog.db_version, "sim"
                ):
                    raise ValueError(f"sample {sample.sample_id} in {shard} belongs to a different run")
                seen.add(sample.sample_id)
                existing_subruns += len(sample.runs)
            entries.append(_shard_entry(shard, len(rows)))
        completed = len(seen)
        started = time.monotonic()
        stopped_by_time_limit = False
        new_subruns = repeated_battles = 0
        buckets_new: Counter[str] = Counter()
        guilds_new: Counter[int] = Counter()
        sync_root = Path(sync_dir) if sync_dir is not None else None
        manifest = path.with_name(f"{path.stem}.manifest.json")
        synced_shards = 0
        written = failed = 0
        index = max(shards, default=-1) + 1
        pending: list[BattleSample] = []
        if resume and failed_path is not None and failed_path.is_file():
            with failed_path.open(encoding="utf-8") as log:
                seen.update(json.loads(line)["sample_id"] for line in log if line.strip())
        remaining = max(0, target - completed)
        work = self._unique(remaining, seen)

        def batches() -> Iterator[list[BattleSample]]:
            while batch := [sample for sample, _ in islice(work, batch_size)]:
                yield batch

        def publish_manifest() -> None:
            nonlocal synced_shards
            self._write_manifest(path, entries, completed + written)
            if sync_root is not None:
                while synced_shards < len(entries):
                    shard = path.with_name(entries[synced_shards]["filename"])
                    _sync_file_atomic(shard, sync_root)
                    synced_shards += 1
                _sync_file_atomic(manifest, sync_root)
                if failed_path is not None and failed_path.is_file():
                    _sync_file_atomic(failed_path, sync_root)

        def flush(count: int) -> None:
            nonlocal index, written
            shard = _shard_path(path, index)
            _write_shard(shard, pending[:count])
            entries.append(_shard_entry(shard, count))
            del pending[:count]
            written += count
            index += 1
            publish_manifest()
            if progress_stderr:
                print(f"Collected {completed + written}/{target} battles ({written} new), wrote {shard.name}",
                      file=sys.stderr, flush=True)

        def simulate(batch: list[BattleSample]) -> tuple[list[BattleSample], list[dict[str, str]]]:
            try:
                return self.simulate_samples(client, batch), []
            except SimulationTimeoutError:
                raise
            except SimulationError as exc:
                if not isolate:
                    raise
                if len(batch) == 1:
                    return [], [{"sample_id": batch[0].sample_id, "error": str(exc)}]
                mid = len(batch) // 2
                left, left_errors = simulate(batch[:mid])
                right, right_errors = simulate(batch[mid:])
                return left + right, left_errors + right_errors

        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                stream = batches()
                while remaining:
                    if max_seconds is not None and time.monotonic() - started >= max_seconds:
                        stopped_by_time_limit = True
                        break
                    window = list(islice(stream, concurrency))
                    remaining -= sum(map(len, window))
                    futures = [pool.submit(simulate, batch) for batch in window]
                    for future in futures:
                        valid, errors = future.result()
                        pending.extend(valid)
                        for sample in valid:
                            runs = len(sample.runs)
                            new_subruns += runs
                            repeated_battles += runs > self.initial_runs
                            buckets_new[sample.bucket] += 1
                            guilds_new[sample.guild_id] += 1
                        failed += len(errors)
                        if errors and failed_path is not None:
                            failed_path.parent.mkdir(parents=True, exist_ok=True)
                            with failed_path.open("a", encoding="utf-8") as log:
                                for error in errors:
                                    log.write(json.dumps(error, ensure_ascii=False) + "\n")
                        while len(pending) >= chunk_size:
                            flush(chunk_size)
        except (SimulationError, KeyboardInterrupt):
            if pending:
                flush(len(pending))
            else:
                publish_manifest()
            raise
        if pending:
            flush(len(pending))
        else:
            publish_manifest()
        elapsed = max(1e-9, time.monotonic() - started)
        result = {
            "existing_battles": completed, "new_battles": written, "total_battles": completed + written,
            "new_shards": index - max(shards, default=-1) - 1,
            "existing_subruns": existing_subruns, "new_subruns": new_subruns,
            "total_subruns": existing_subruns + new_subruns,
            "repeated_battles": repeated_battles, "repeat_rate": repeated_battles / written if written else 0.0,
            "elapsed_sec": round(elapsed, 4), "rows_per_sec": round(written / elapsed, 4),
            "subruns_per_sec": round(new_subruns / elapsed, 4), "stopped_by_time_limit": stopped_by_time_limit,
            "buckets_new": dict(sorted(buckets_new.items())),
            "guilds_new": {str(k): v for k, v in sorted(guilds_new.items())},
        }
        if isolate:
            result["failed_battles"] = failed
        return result

    def _write_manifest(self, path: Path, entries: list[dict[str, Any]], total: int) -> None:
        manifest = path.with_name(f"{path.stem}.manifest.json")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "shards": entries, "total_unique_battles": total, "round_id": self.round_id,
            "db_version": self.catalog.db_version,
            "catalog_sha256": self.catalog.catalog_sha256 or self.catalog.compute_sha256(),
        }
        with tempfile.NamedTemporaryFile(dir=manifest.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")
            os.replace(temporary, manifest)
        finally:
            temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _shard_entry(path: Path, rows: int) -> dict[str, Any]:
    return {"filename": path.name, "row_count": rows, "sha256": _file_sha256(path)}


def _sync_file_atomic(src: Path, dst_dir: Path) -> bool:
    if not src.is_file() or src.parent.resolve() == dst_dir.resolve():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.is_file() and _file_sha256(src) == _file_sha256(dst):
        return False
    with tempfile.NamedTemporaryFile(dir=dst_dir, prefix=f".{src.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(src, temporary)
        os.replace(temporary, dst)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _shard_path(base: Path, index: int) -> Path:
    return base if index == 0 else base.with_name(f"{base.stem}.part-{index:05d}{base.suffix}")


def _existing_shards(base: Path) -> dict[int, Path]:
    shards = {0: base} if base.is_file() else {}
    if not base.parent.exists():
        return shards
    pattern = re.compile(rf"{re.escape(base.stem)}\.part-([0-9]{{5,}}){re.escape(base.suffix)}")
    for candidate in base.parent.iterdir():
        match = pattern.fullmatch(candidate.name)
        if match and candidate.is_file():
            index = int(match[1])
            if not index or index in shards:
                raise ValueError(f"ambiguous shard index in {candidate}")
            shards[index] = candidate
    return dict(sorted(shards.items()))


def _write_shard(path: Path, samples: list[BattleSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}-", suffix=path.suffix,
                                     dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write_samples(temporary, samples)
        # Publish the completed shard atomically and fail if another writer won the race.
        try:
            os.link(temporary, path)
        except OSError as exc:
            if exc.errno not in UNSUPPORTED_LINK:
                raise
            if path.exists():
                raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path) from exc
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def select_hard_negatives(
    samples: Iterable[BattleSample], predictions: Mapping[str, float] | None = None, *, limit: int = 100,
) -> list[BattleSample]:
    """Rank labeled battles by prediction error then uncertainty, or by mixed results and close losses."""
    def ranked_samples() -> Iterator[tuple[tuple[float, ...], BattleSample]]:
        for sample in samples:
            total = sum(sample.outcome_counts)
            if total:
                wins = sample.outcome_counts[OUTCOME_WIN]
            else:
                total = len(sample.runs)
                wins = sum(run.outcome_class == OUTCOME_WIN for run in sample.runs)
            if not total:
                continue
            win_rate = wins / total
            mixed = min(win_rate, 1.0 - win_rate)
            damage = sample.mean_damage_ratio
            close_loss = damage if win_rate < 1.0 and 0.0 < damage < 1.0 else 0.0
            prediction = predictions.get(sample.sample_id) if predictions is not None else None
            if prediction is not None:
                if not 0.0 <= prediction <= 1.0:
                    raise ValueError(f"invalid win probability for sample {sample.sample_id}")
                error = abs(win_rate - prediction)
                uncertainty = min(prediction, 1.0 - prediction)
                yield (1, error, uncertainty, mixed, close_loss), sample
            elif mixed or close_loss:
                yield (0, 0, 0, mixed, close_loss), sample

    return [sample for _, sample in nlargest(limit, ranked_samples(), key=lambda item: item[0])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path("configs/sim_profile_highscore_cn.json"))
    parser.add_argument("--sampling", "--config", type=Path, default=Path("configs/sampling_round0.json"))
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--target-battles", "--count", type=int)
    parser.add_argument("--seed", type=int, help="sampling seed for this worker")
    parser.add_argument("--round-id", help="round identifier for this worker")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--timeout-sec", type=float)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--sync-dir", type=Path)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--isolate-errors", action="store_true", help="quarantine samples that fail simulation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", type=int, metavar="N")
    args = parser.parse_args(argv)
    if args.dry_run is None and args.out is None:
        parser.error("--out is required when collecting battles")
    try:
        for name in ("batch_size", "concurrency", "timeout_sec", "max_seconds"):
            if (value := getattr(args, name)) is not None and not value > 0:
                raise ValueError(f"{name} must be positive")
        catalog = FeatureCatalog.load(args.catalog)
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        config = json.loads(args.sampling.read_text(encoding="utf-8"))
        for name in ("seed", "round_id", "chunk_size", "batch_size", "concurrency"):
            if (value := getattr(args, name)) is not None:
                config[name] = value
        if args.timeout_sec is not None:
            profile["timeout_sec"] = args.timeout_sec
        if args.isolate_errors:
            config["isolate_batch_errors"] = True
        sampler = HighScoreSampler(catalog, profile, config)
        if args.dry_run is not None:
            result = sampler.dry_run(args.dry_run)
        else:
            target = args.target_battles if args.target_battles is not None else int(config.get("target_battles", 25000))
            if target > int(config.get("max_full_target_battles", 150000)):
                raise ValueError("target exceeds max_full_target_battles")
            result = sampler.collect(
                SimClient(args.url, catalog, profile), args.out, target, resume=args.resume,
                failure_log=args.out.with_name(f"{args.out.stem}.failed.jsonl"),
                sync_dir=args.sync_dir, max_seconds=args.max_seconds, progress_stderr=args.progress,
            )
    except (ImportError, OSError, ValueError, SimulationError) as exc:
        parser.exit(1, f"sampling failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
