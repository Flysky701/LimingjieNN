"""Battle features shared by training, candidate search and ONNX fixtures."""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from labyrinth_nn.catalog.schema import (
    ATTR_BONUS_TABLE, ENEMY_STATIC_FEAT_NAMES, GLOBAL_FEAT_NAMES, MARK_THRESHOLDS,
    TEAM_POS_CHANNELS, TEAM_SYNERGY_FEAT_NAMES, UNIT_MECH_FLAG_NAMES,
    UNIT_STATIC_FEAT_NAMES, EnemyEntry, FeatureCatalog, UnitEntry,
)
from labyrinth_nn.data.sample_schema import BattleSample, NUM_OUTCOME_CLASSES, OUTCOME_WIN


def build_vocab_maps(catalog: FeatureCatalog) -> dict[str, dict[int, int]]:
    return dict(unit_to_idx=catalog.unit_to_idx(), enemy_to_idx=catalog.enemy_to_idx(),
                quest_to_idx=catalog.quest_to_idx(), relic_to_idx=catalog.relic_to_idx())


def _log(value: float) -> float:
    return math.log1p(max(0.0, value))


def unit_static_features(unit: UnitEntry | None) -> list[float]:
    if unit is None:
        return [0.0] * len(UNIT_STATIC_FEAT_NAMES)
    if len(unit.static_feats) == len(UNIT_STATIC_FEAT_NAMES):
        return list(unit.static_feats)
    panels = (unit.panel_hp, unit.panel_atk, unit.panel_magic_str, unit.panel_def,
              unit.panel_magic_def, unit.panel_dodge, unit.panel_life_steal,
              unit.panel_hp_recovery_rate, unit.panel_energy_recovery_rate,
              unit.panel_energy_reduce_rate)
    return [
        unit.search_area_width / 1000.0, float(unit.atk_type == 1), float(unit.atk_type == 2),
        float(unit.has_6star), float(unit.has_ue1), float(unit.has_ue2),
        float(unit.has_labyrinth_replace), *map(_log, panels),
        *(float(unit.mech_flags.get(name, False)) for name in UNIT_MECH_FLAG_NAMES),
    ]


def enemy_static_features(enemy: EnemyEntry | None) -> list[float]:
    if enemy is None:
        return [0.0] * len(ENEMY_STATIC_FEAT_NAMES)
    if len(enemy.static_feats) == len(ENEMY_STATIC_FEAT_NAMES):
        return list(enemy.static_feats)
    return [
        enemy.level / 100.0, _log(enemy.hp), _log(enemy.atk), _log(enemy.magic_str),
        _log(enemy.physical_def), _log(enemy.magic_def), _log(enemy.dodge),
        _log(enemy.accuracy), float(enemy.is_boss_unit), float(enemy.has_dot_resist),
        float(enemy.has_cc_resist),
    ]


def _pad(values: Sequence, length: int) -> list:
    result = list(values[:length])
    return result + [0] * (length - len(result))


def _position(catalog: FeatureCatalog, uid: int) -> int:
    unit = catalog.units.get(uid)
    return unit.search_area_width if unit is not None else 10**9


def _sorted_units(catalog: FeatureCatalog, ids: Sequence[int]) -> list[int]:
    return sorted(set(ids) - {0}, key=lambda uid: (_position(catalog, uid), uid))


def team_features(catalog: FeatureCatalog, team_ids: Sequence[int], relic_ids: Sequence[int],
                  gold_norm: float, owned_norm: float) -> dict[str, Any]:
    """Build only the candidate-dependent inputs. Slot order is shared with training."""
    ids = _sorted_units(catalog, team_ids)[:5]
    units = [catalog.units.get(uid) for uid in ids]
    positions = [_position(catalog, uid) for uid in ids]
    size = len(ids)
    matrix = []
    for i in range(5):
        row = []
        for j in range(5):
            if i < size and j < size:
                diff = float(positions[j] - positions[i])
                row.append([diff / 1000.0, float(abs(diff) <= 300), float(abs(diff) <= 400)])
            else:
                row.append([0.0] * TEAM_POS_CHANNELS)
        matrix.append(row)
    talents = [0] * 5
    physical = magic = 0
    for unit in units:
        if unit is None:
            continue
        if 1 <= unit.talent_id <= 5:
            talents[unit.talent_id - 1] += 1
        static = unit_static_features(unit)
        physical += unit.atk_type == 1 or static[1] > 0.5
        magic += unit.atk_type == 2 or static[2] > 0.5
    most = max(talents)
    front = units[0] if units else None
    relics = [catalog.relics[rid] for rid in relic_ids if rid in catalog.relics]
    synergy = {
        "team_size_norm": size / 5, "max_same_talent_norm": most / 5,
        "attr_bonus_value": ATTR_BONUS_TABLE[min(5, most)],
        "is_mono_5": float(size == most == 5),
        "phys_ratio": physical / max(1, size), "mag_ratio": magic / max(1, size),
        "front_gap_norm": (positions[1] - positions[0]) / 1000.0 if size >= 2 else 0.0,
        "front_pdef_x_relic": (front.panel_def / 1000.0 if front else 0.0)
                              * sum(r.front_pdef_to_atk_mult for r in relics),
        "front_mdef_x_relic": (front.panel_magic_def / 1000.0 if front else 0.0)
                              * sum(r.front_mdef_to_atk_mult for r in relics),
        "gold_x_gold_relic": gold_norm * sum(r.gold_scaling for r in relics),
        "owned_x_owned_relic": owned_norm * sum(r.owned_unit_scaling for r in relics),
        "element_amp_sum": float(sum(r.element_bonus_amp for r in relics)),
    }
    synergy.update({f"team_talent_{i}_norm": count / 5 for i, count in enumerate(talents, 1)})
    return {
        "team_unit_ids_sorted": _pad(ids, 5), "team_slot_mask": [i < size for i in range(5)],
        "team_pos_matrix": matrix, "team_synergy": [synergy[name] for name in TEAM_SYNERGY_FEAT_NAMES],
    }


def featurize_sample(
    catalog: FeatureCatalog, sample: BattleSample, unit_to_idx: dict[int, int],
    enemy_to_idx: dict[int, int], quest_to_idx: dict[int, int], relic_to_idx: dict[int, int],
    *, relic_max_len: int = 32, roster_max_len: int = 64, training: bool = False,
    unit_id_dropout: float = 0.0, relic_id_mask_prob: float = 0.0,
    relic_stack_noise_prob: float = 0.0, gold_mask_prob: float = 0.0,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    if roster_max_len < 1 or relic_max_len < 0:
        raise ValueError("roster_max_len must be positive and relic_max_len nonnegative")
    if not training:
        rng = None
    elif rng is None:
        rng = random.Random()

    def draw(probability: float) -> bool:
        return rng is not None and probability > 0.0 and rng.random() < probability

    visible_relic_ids = []
    stacks = _pad(sample.mark_stacks, 5)
    for rid in sample.relic_ids:
        if draw(relic_id_mask_prob):
            rel = catalog.relics.get(rid)
            if (rel is not None and 1 <= rel.relic_mark_id <= 5 and rel.relic_mark_count > 0
                    and draw(relic_stack_noise_prob)):
                slot = rel.relic_mark_id - 1
                stacks[slot] = max(0, stacks[slot] - rel.relic_mark_count)
        else:
            visible_relic_ids.append(rid)
    gold_known = sample.alpha_rupee is not None and not draw(gold_mask_prob)
    gold_norm = min(max(0.0, float(sample.alpha_rupee)), 10000.0) / 10000.0 if gold_known else 0.0
    owned_count = float(sample.owned_unit_count)
    owned_norm = min(owned_count, 25.0) / 25.0
    global_values = {
        "owned_unit_count_25_norm": owned_norm, "owned_unit_count_60_norm": owned_count / 60.0,
        "alpha_rupee_norm": gold_norm, "gold_known": float(gold_known),
        "difficulty_norm": sample.difficulty / max(1, catalog.max_difficulty),
        "area_norm": sample.area / max(1, catalog.max_area),
        "has_empty_relics": float(not visible_relic_ids), "revive_available": float(sample.revive_available),
    }
    for i, stack in enumerate(stacks, 1):
        global_values[f"mark_{i}_norm"] = stack / 15.0
        for threshold in MARK_THRESHOLDS:
            global_values[f"mark_{i}_ge{threshold}"] = float(stack >= threshold)
    for kind, counts, length in (("role", sample.owned_role_counts, 8),
                                 ("talent", sample.owned_talent_counts, 5)):
        for i, count in enumerate(_pad(counts, length), 1):
            global_values[f"owned_{kind}_{i}_norm"] = count / 20.0

    team = team_features(catalog, sample.team_unit_ids, visible_relic_ids, gold_norm, owned_norm)
    team_ids = [uid for uid in team["team_unit_ids_sorted"] if uid]
    if len(team_ids) > roster_max_len:
        raise ValueError("roster_max_len cannot hold every active team slot")
    selected = set(team_ids)
    others = [uid for uid in _sorted_units(catalog, sample.owned_unit_ids) if uid not in selected]
    roster = _pad(team_ids + others, roster_max_len)
    roster_pos = {uid: i for i, uid in enumerate(roster) if uid}
    quest = catalog.quests.get(sample.quest_id)
    enemies = _pad(quest.enemy_ids if quest else [], 5)
    relics = _pad(visible_relic_ids, relic_max_len)
    counts = sample.outcome_counts
    if len(counts) != NUM_OUTCOME_CLASSES or any(c < 0 for c in counts):
        raise ValueError("outcome_counts must contain three nonnegative counts")
    total = sum(counts)
    has_time = sample.mean_win_rem_time is not None and counts[OUTCOME_WIN] > 0
    return {
        **team, "team_roster_pos": [roster_pos.get(uid, 0) for uid in team["team_unit_ids_sorted"]],
        "roster_raw_ids": roster, "roster_mask": [bool(uid) for uid in roster],
        "roster_unit_ids": [0 if uid and draw(unit_id_dropout) else unit_to_idx.get(uid, 0) for uid in roster],
        "roster_unit_feats": [unit_static_features(catalog.units.get(uid)) for uid in roster],
        "quest_idx": quest_to_idx.get(sample.quest_id, 0), "area_idx": max(0, sample.area),
        "difficulty_idx": max(0, sample.difficulty), "enemy_ids": [enemy_to_idx.get(e, 0) for e in enemies],
        "enemy_feats": [enemy_static_features(catalog.enemies.get(e)) for e in enemies],
        "enemy_mask": [bool(e) for e in enemies], "relic_ids": [relic_to_idx.get(r, 0) for r in relics],
        "relic_mask": [bool(r) for r in relics], "global_feats": [global_values[n] for n in GLOBAL_FEAT_NAMES],
        "outcome_probs_target": [c / total for c in counts] if total else [0.0, 0.0, 1.0],
        "has_outcome_target": bool(total), "mean_damage_ratio": min(1.0, max(0.0, sample.mean_damage_ratio)),
        "mean_win_rem_time": min(1.0, max(0.0, sample.mean_win_rem_time)) if has_time else 0.0,
        "has_win_target": has_time, "sample_id": sample.sample_id, "quest_id": sample.quest_id,
        "canonical_wave_sig": sample.canonical_wave_sig, "team_sig": sample.canonical_team_sig(),
        "bucket": sample.bucket, "gold_known": gold_known,
    }


def split_samples(samples: Sequence[BattleSample], val_ratio: float = 0.15,
                  test_ratio: float = 0.15, seed: int = 42
                  ) -> tuple[list[BattleSample], list[BattleSample], list[BattleSample]]:
    if not (0 <= val_ratio and 0 <= test_ratio and val_ratio + test_ratio < 1):
        raise ValueError("split ratios must be nonnegative and sum to less than one")
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.canonical_wave_sig, sample.canonical_team_sig()].append(sample)

    def order(key: tuple[str, str]) -> tuple[str, tuple[str, str]]:
        digest = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode("utf-8")).hexdigest()
        return digest, key

    keys = sorted(groups, key=order)
    n = len(keys)
    min_val = int(n >= 3 and val_ratio > 0)
    min_test = int(n >= 3 and test_ratio > 0)
    n_test = min(max(min_test, int(n * test_ratio)), max(0, n - 1 - min_val))
    n_val = min(max(min_val, int(n * val_ratio)), max(0, n - 1 - n_test))
    cut = n - n_test - n_val

    def flatten(selected: list[tuple[str, str]]) -> list[BattleSample]:
        return [sample for key in selected for sample in groups[key]]

    return flatten(keys[:cut]), flatten(keys[cut:cut + n_val]), flatten(keys[cut + n_val:])


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("BattleDataset and tensor collation require PyTorch") from exc
    return torch


class BattleDataset:
    """Map-style PyTorch dataset. Noise is reproducible per epoch and sample index."""

    def __init__(self, catalog: FeatureCatalog, samples: Sequence[BattleSample], *,
                 seed: int = 42, training: bool = False, **feature_options: Any):
        _torch()
        self.catalog = catalog
        self.samples = samples
        self.maps = build_vocab_maps(catalog)
        self.options = feature_options
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(f"{self.seed}|{self.epoch}|{index}") if self.training else None
        return featurize_sample(self.catalog, self.samples[index], **self.maps, **self.options,
                                training=self.training, rng=rng)


_INT_FIELDS = {"quest_idx", "area_idx", "difficulty_idx", "enemy_ids", "relic_ids",
               "roster_unit_ids", "team_roster_pos"}
_BOOL_FIELDS = {"enemy_mask", "relic_mask", "roster_mask", "team_slot_mask",
                "has_outcome_target", "has_win_target"}
_FLOAT_FIELDS = {"roster_unit_feats", "enemy_feats", "global_feats", "team_pos_matrix",
                 "team_synergy", "outcome_probs_target", "mean_damage_ratio", "mean_win_rem_time"}


def collate_battle_batch(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Tensorize model inputs and labels, retaining raw IDs and group metadata as lists."""
    torch = _torch()
    if not rows:
        raise ValueError("cannot collate an empty battle batch")
    batch = {name: [row[name] for row in rows] for name in rows[0]}
    for fields, dtype in ((_INT_FIELDS, torch.long), (_BOOL_FIELDS, torch.bool),
                          (_FLOAT_FIELDS, torch.float32)):
        for name in fields & batch.keys():
            batch[name] = torch.tensor(batch[name], dtype=dtype)
    return batch
