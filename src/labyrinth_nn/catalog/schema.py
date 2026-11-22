"""Frozen catalog schema shared across catalog extraction, sampling, Two-Tower NN, and Path B planning."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Clean atomic binary mechanism flags extracted objectively from skill_action + labyrinth_skill_action_replace.
UNIT_MECH_FLAG_NAMES: tuple[str, ...] = (
    "dot_slip",
    "cc_abnormal",
    "barrier_shield",
    "taunt_decoy",
    "move_or_knock",
    "tp_charge",
    "tp_drain",
    "heal_or_regen",
    "dispel",
    "aoe_dmg",
)

# Clean objective panel + form + atomic mechanism features (no hand-crafted subjective scores).
UNIT_STATIC_FEAT_NAMES: tuple[str, ...] = (
    "search_area_width_norm",
    "is_physical",
    "is_magic",
    "has_6star",
    "has_ue1",
    "has_ue2",
    "has_labyrinth_replace",
    "log_hp_z",
    "log_atk_z",
    "log_magic_str_z",
    "def_z",
    "magic_def_z",
    "dodge_z",
    "life_steal_z",
    "hp_recovery_rate_z",
    "energy_recovery_rate_z",
    "energy_reduce_rate_z",
    *UNIT_MECH_FLAG_NAMES,
)

ENEMY_STATIC_FEAT_NAMES: tuple[str, ...] = (
    "level_norm",
    "log_hp_z",
    "log_atk_z",
    "log_magic_str_z",
    "def_z",
    "magic_def_z",
    "dodge_z",
    "accuracy_z",
    "is_boss_unit",
    "has_dot_resist",
    "has_cc_resist",
)

GLOBAL_FEAT_NAMES: tuple[str, ...] = (
    "mark_1_norm",
    "mark_2_norm",
    "mark_3_norm",
    "mark_4_norm",
    "mark_5_norm",
    "mark_1_ge4",
    "mark_2_ge4",
    "mark_3_ge4",
    "mark_4_ge4",
    "mark_5_ge4",
    "mark_1_ge9",
    "mark_2_ge9",
    "mark_3_ge9",
    "mark_4_ge9",
    "mark_5_ge9",
    "mark_1_ge15",
    "mark_2_ge15",
    "mark_3_ge15",
    "mark_4_ge15",
    "mark_5_ge15",
    "owned_unit_count_25_norm",
    "owned_unit_count_60_norm",
    "owned_role_1_norm",
    "owned_role_2_norm",
    "owned_role_3_norm",
    "owned_role_4_norm",
    "owned_role_5_norm",
    "owned_role_6_norm",
    "owned_role_7_norm",
    "owned_role_8_norm",
    "owned_talent_1_norm",
    "owned_talent_2_norm",
    "owned_talent_3_norm",
    "owned_talent_4_norm",
    "owned_talent_5_norm",
    "alpha_rupee_norm",
    "gold_known",
    "difficulty_norm",
    "area_norm",
    "has_empty_relics",
    "revive_available",
)

TEAM_SYNERGY_FEAT_NAMES: tuple[str, ...] = (
    "team_size_norm",
    "team_talent_1_norm",
    "team_talent_2_norm",
    "team_talent_3_norm",
    "team_talent_4_norm",
    "team_talent_5_norm",
    "max_same_talent_norm",
    "attr_bonus_value",
    "is_mono_5",
    "phys_ratio",
    "mag_ratio",
    "front_gap_norm",
    "front_pdef_x_relic",
    "front_mdef_x_relic",
    "gold_x_gold_relic",
    "owned_x_owned_relic",
    "element_amp_sum",
)

TEAM_POS_CHANNELS: int = 3
ATTR_BONUS_TABLE: tuple[float, ...] = (0.0, 0.0, 0.08, 0.16, 0.25, 0.75)
MARK_THRESHOLDS: tuple[int, ...] = (4, 9, 15)


@dataclass
class UnitEntry:
    unit_id: int
    name: str
    search_area_width: int
    atk_type: int
    talent_id: int
    role_id: int
    npc_enhance_id: int
    has_6star: bool
    has_ue1: bool
    has_ue2: bool
    has_labyrinth_replace: bool
    panel_hp: float
    panel_atk: float
    panel_magic_str: float
    panel_def: float
    panel_magic_def: float
    panel_dodge: float = 0.0
    panel_life_steal: float = 0.0
    panel_hp_recovery_rate: float = 0.0
    panel_energy_recovery_rate: float = 0.0
    panel_energy_reduce_rate: float = 0.0
    mech_flags: dict[str, bool] = field(default_factory=dict)
    static_feats: list[float] = field(default_factory=list)


@dataclass
class EnemyEntry:
    enemy_id: int
    unit_id: int
    name: str
    level: int
    hp: int
    virtual_hp: int
    atk: int
    magic_str: int
    physical_def: int
    magic_def: int
    dodge: int
    accuracy: int
    resist_status_id: int
    is_boss_unit: bool = False
    has_dot_resist: bool = False
    has_cc_resist: bool = False
    static_feats: list[float] = field(default_factory=list)


@dataclass
class QuestEntry:
    quest_id: int
    wave_group_id: int
    quest_type: int
    area: int
    limit_time: int
    reward_id: int
    enemy_ids: list[int]
    wave_total_hp: int
    canonical_wave_sig: str
    bucket: str
    is_difficulty5_boss: bool = False
    quest_name: str = ""


@dataclass
class RelicEntry:
    relic_id: int
    name: str
    rarity: int
    relic_mark_id: int
    relic_mark_count: int
    price: int
    effect_description: str = ""
    gold_scaling: bool = False
    owned_unit_scaling: bool = False
    front_pdef_to_atk_mult: float = 0.0
    front_mdef_to_atk_mult: float = 0.0
    element_bonus_amp: float = 0.0
    is_revive: bool = False
    is_shop_discount: bool = False
    shop_discount_rate: float = 0.0


@dataclass
class GuildEntry:
    guild_id: int
    guild_name: str
    guild_master_id: int
    add_relic_id: int


@dataclass
class EventChoiceEntry:
    event_id: int
    choice_id: int
    condition_type: int
    condition_value: int
    unit_id: int
    description: str
    result_ids: list[int] = field(default_factory=list)


@dataclass
class ScoreSettings:
    area_clear_pts: dict[int, int] = field(default_factory=dict)
    block_pts: dict[int, int] = field(default_factory=dict)
    unit_pts: int = 100
    relic_pts: int = 200
    normal_pts: int = 100
    ex_pts: int = 1000
    boss_pts: int = 1500
    score_rates: dict[int, int] = field(default_factory=dict)
    shop_refresh_costs: list[int] = field(default_factory=lambda: [300, 600, 1200, 2400, 4800, 9999])


@dataclass
class FeatureCatalog:
    schema_version: int
    db_version: str
    db_sha256: str
    catalog_sha256: str
    max_area: int
    max_difficulty: int
    units: dict[int, UnitEntry]
    enemies: dict[int, EnemyEntry]
    quests: dict[int, QuestEntry]
    relics: dict[int, RelicEntry]
    guilds: dict[int, GuildEntry]
    score_settings: ScoreSettings
    enhance_node_ids: list[int]
    environment_enemy_ids: list[int]
    ex_disambiguation_index: dict[str, list[int]]
    event_choices: dict[int, EventChoiceEntry] = field(default_factory=dict)
    max_account_training: dict[str, Any] = field(default_factory=dict)
    unit_feat_names: list[str] = field(default_factory=lambda: list(UNIT_STATIC_FEAT_NAMES))
    enemy_feat_names: list[str] = field(default_factory=lambda: list(ENEMY_STATIC_FEAT_NAMES))
    global_feat_names: list[str] = field(default_factory=lambda: list(GLOBAL_FEAT_NAMES))
    team_synergy_feat_names: list[str] = field(default_factory=lambda: list(TEAM_SYNERGY_FEAT_NAMES))

    def sorted_unit_ids(self) -> list[int]:
        return sorted(self.units.keys())

    def sorted_enemy_ids(self) -> list[int]:
        return sorted(self.enemies.keys())

    def sorted_quest_ids(self) -> list[int]:
        return sorted(self.quests.keys())

    def sorted_relic_ids(self) -> list[int]:
        return sorted(self.relics.keys())

    def unit_to_idx(self) -> dict[int, int]:
        return {uid: i + 1 for i, uid in enumerate(self.sorted_unit_ids())}

    def enemy_to_idx(self) -> dict[int, int]:
        return {eid: i + 1 for i, eid in enumerate(self.sorted_enemy_ids())}

    def quest_to_idx(self) -> dict[int, int]:
        return {qid: i + 1 for i, qid in enumerate(self.sorted_quest_ids())}

    def relic_to_idx(self) -> dict[int, int]:
        return {rid: i + 1 for i, rid in enumerate(self.sorted_relic_ids())}

    def mark_stacks_for_relics(self, relic_ids: list[int]) -> list[int]:
        stacks = [0, 0, 0, 0, 0]
        for rid in relic_ids:
            rel = self.relics.get(rid)
            if rel is not None and 1 <= rel.relic_mark_id <= 5 and rel.relic_mark_count > 0:
                stacks[rel.relic_mark_id - 1] += rel.relic_mark_count
        return stacks

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "db_version": self.db_version,
            "db_sha256": self.db_sha256,
            "catalog_sha256": self.catalog_sha256 if include_hash else "",
            "max_area": self.max_area,
            "max_difficulty": self.max_difficulty,
            "units": {str(k): asdict(v) for k, v in sorted(self.units.items())},
            "enemies": {str(k): asdict(v) for k, v in sorted(self.enemies.items())},
            "quests": {str(k): asdict(v) for k, v in sorted(self.quests.items())},
            "relics": {str(k): asdict(v) for k, v in sorted(self.relics.items())},
            "guilds": {str(k): asdict(v) for k, v in sorted(self.guilds.items())},
            "score_settings": {
                "area_clear_pts": {str(k): v for k, v in sorted(self.score_settings.area_clear_pts.items())},
                "block_pts": {str(k): v for k, v in sorted(self.score_settings.block_pts.items())},
                "unit_pts": self.score_settings.unit_pts,
                "relic_pts": self.score_settings.relic_pts,
                "normal_pts": self.score_settings.normal_pts,
                "ex_pts": self.score_settings.ex_pts,
                "boss_pts": self.score_settings.boss_pts,
                "score_rates": {str(k): v for k, v in sorted(self.score_settings.score_rates.items())},
                "shop_refresh_costs": list(self.score_settings.shop_refresh_costs),
            },
            "enhance_node_ids": list(self.enhance_node_ids),
            "environment_enemy_ids": list(self.environment_enemy_ids),
            "ex_disambiguation_index": self.ex_disambiguation_index,
            "event_choices": {str(k): asdict(v) for k, v in sorted(self.event_choices.items())},
            "max_account_training": self.max_account_training,
            "unit_feat_names": list(self.unit_feat_names),
            "enemy_feat_names": list(self.enemy_feat_names),
            "global_feat_names": list(self.global_feat_names),
            "team_synergy_feat_names": list(self.team_synergy_feat_names),
        }
        return payload

    def compute_sha256(self) -> str:
        canonical = json.dumps(self.to_dict(include_hash=False), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def finalize_hash(self) -> str:
        self.catalog_sha256 = self.compute_sha256()
        return self.catalog_sha256

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.finalize_hash()
        out.write_text(json.dumps(self.to_dict(include_hash=True), ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeatureCatalog":
        ss_raw = raw["score_settings"]
        score_settings = ScoreSettings(
            area_clear_pts={int(k): int(v) for k, v in ss_raw["area_clear_pts"].items()},
            block_pts={int(k): int(v) for k, v in ss_raw["block_pts"].items()},
            unit_pts=int(ss_raw["unit_pts"]),
            relic_pts=int(ss_raw["relic_pts"]),
            normal_pts=int(ss_raw["normal_pts"]),
            ex_pts=int(ss_raw["ex_pts"]),
            boss_pts=int(ss_raw["boss_pts"]),
            score_rates={int(k): int(v) for k, v in ss_raw["score_rates"].items()},
            shop_refresh_costs=[int(x) for x in ss_raw["shop_refresh_costs"]],
        )
        return cls(
            schema_version=int(raw["schema_version"]),
            db_version=str(raw.get("db_version", "")),
            db_sha256=str(raw.get("db_sha256", "")),
            catalog_sha256=str(raw.get("catalog_sha256", "")),
            max_area=int(raw["max_area"]),
            max_difficulty=int(raw["max_difficulty"]),
            units={int(k): UnitEntry(**v) for k, v in raw["units"].items()},
            enemies={int(k): EnemyEntry(**v) for k, v in raw["enemies"].items()},
            quests={int(k): QuestEntry(**v) for k, v in raw["quests"].items()},
            relics={int(k): RelicEntry(**v) for k, v in raw["relics"].items()},
            guilds={int(k): GuildEntry(**v) for k, v in raw["guilds"].items()},
            score_settings=score_settings,
            enhance_node_ids=[int(x) for x in raw["enhance_node_ids"]],
            environment_enemy_ids=[int(x) for x in raw["environment_enemy_ids"]],
            ex_disambiguation_index={str(k): [int(q) for q in v] for k, v in raw.get("ex_disambiguation_index", {}).items()},
            event_choices={int(k): EventChoiceEntry(**v) for k, v in raw.get("event_choices", {}).items()},
            max_account_training=dict(raw.get("max_account_training", {})),
            unit_feat_names=list(raw.get("unit_feat_names", UNIT_STATIC_FEAT_NAMES)),
            enemy_feat_names=list(raw.get("enemy_feat_names", ENEMY_STATIC_FEAT_NAMES)),
            global_feat_names=list(raw.get("global_feat_names", GLOBAL_FEAT_NAMES)),
            team_synergy_feat_names=list(raw.get("team_synergy_feat_names", TEAM_SYNERGY_FEAT_NAMES)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)
