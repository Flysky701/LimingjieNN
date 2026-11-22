"""Build the frozen feature catalog from a read-only CN master database."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from labyrinth_nn.catalog.schema import (
    UNIT_MECH_FLAG_NAMES,
    EnemyEntry,
    EventChoiceEntry,
    FeatureCatalog,
    GuildEntry,
    QuestEntry,
    RelicEntry,
    ScoreSettings,
    UnitEntry,
)

Row = dict[str, Any]
ENEMY_SLOTS = tuple(f"enemy_id_{slot}" for slot in range(1, 6))
PANEL_COLUMNS = (
    "hp", "atk", "magic_str", "def", "magic_def", "dodge", "life_steal",
    "hp_recovery_rate", "energy_recovery_rate", "energy_reduce_rate",
)
CC_AILMENT_ACTIONS = {8, 11, 12, 13, 61, 62, 69, 116}
STANDARD_DOT_AILMENT_IDS = {10, 11, 12, 13, 20, 28, 33, 35, 36, 43, 48, 53}
STANDARD_CC_AILMENT_IDS = (set(range(1, 10)) - {2}) | {
    14, 15, 16, 19, 23, 25, 26, 27, 30, 34, 37, 38, 50, 52, 55,
}


def _number(row: Row, key: str) -> int:
    return int(row.get(key) or 0)


def _rows(db: sqlite3.Connection, table: str, *, required: bool = False) -> list[Row]:
    # All table identifiers are internal literals.
    if not required and not db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone():
        return []
    return [dict(row) for row in db.execute(f'SELECT * FROM "{table}"')]


def _index(rows: Iterable[Row], key: str) -> dict[int, Row]:
    return {int(row[key]): row for row in rows}


def _highest(rows: Iterable[Row], key: str, rank: str) -> dict[int, Row]:
    selected: dict[int, Row] = {}
    for row in rows:
        item_id = int(row[key])
        if item_id not in selected or _number(row, rank) > _number(selected[item_id], rank):
            selected[item_id] = row
    return selected


def _log(value: float) -> float:
    return math.log1p(max(0.0, value))


def _normalizers(rows: Iterable[tuple[float, ...]]) -> list[tuple[float, float]]:
    return [(statistics.fmean(col), statistics.pstdev(col)) for col in zip(*rows)]


def _standardize(values: tuple[float, ...], norms: list[tuple[float, float]]) -> list[float]:
    if not norms:
        return [0.0] * len(values)
    return [(v - mean) / std if std else 0.0 for v, (mean, std) in zip(values, norms)]


def _action_flags(action: Row) -> tuple[bool, ...]:
    kind = _number(action, "action_type")
    detail = _number(action, "action_detail_1")
    desc = str(action.get("description") or "").upper()
    drain_description = (
        any(word in desc for word in ("减少", "降低", "削减"))
        and any(word in desc for word in ("技能值", "TP"))
        and not any(word in desc for word in ("降低量", "降低效果量"))
    )
    return (
        kind in {9, 79, 110, 128, 129},
        kind in {11, 12, 13, 61, 62, 69, 116},
        kind in {6, 73},
        kind in {20, 116},
        kind in {2, 3, 52, 55},
        (kind == 16 and detail != 2) or kind in {48, 96},
        (kind == 16 and detail == 2) or drain_description,
        kind in {4, 37, 48},
        kind in {49, 58},
        kind in {1, 36} and (_number(action, "target_range") > 0
                            or _number(action, "target_area") >= 2),
    )


def _read_units(db: sqlite3.Connection) -> dict[int, UnitEntry]:
    roster = _rows(db, "labyrinth_npc_unit_data", required=True)
    unit_data = _index(_rows(db, "unit_data", required=True), "unit_id")
    talents = _index(_rows(db, "unit_talent"), "unit_id")
    roles = _index(_rows(db, "unit_role_data"), "unit_id")
    rarity = _highest(_rows(db, "unit_rarity"), "unit_id", "rarity")
    promotion = _highest(_rows(db, "unit_promotion_status"), "unit_id", "promotion_level")
    baselines = _index(_rows(db, "labyrinth_npc_unit_enhance_data"), "enhance_id")
    max_level = max((_number(row, "unit_level") for row in _rows(db, "experience_unit")), default=0)
    equips: dict[int, set[int]] = defaultdict(set)
    for table in ("unit_unique_equipment", "unit_unique_equip"):
        for row in _rows(db, table):
            if _number(row, "equip_id"):
                equips[int(row["unit_id"])].add(_number(row, "equip_slot"))

    unit_skills = _index(_rows(db, "unit_skill_data"), "unit_id")
    skills = _index(_rows(db, "skill_data"), "skill_id")
    rf_skills: dict[int, set[int]] = defaultdict(set)
    for row in _rows(db, "unit_skill_data_rf"):
        skill_id, rf_skill_id = _number(row, "skill_id"), _number(row, "rf_skill_id")
        if skill_id > 0 and rf_skill_id > 0:
            rf_skills[skill_id].add(rf_skill_id)
    replacements = _index(_rows(db, "labyrinth_skill_action_replace"), "action_id")
    actions = _index(_rows(db, "skill_action"), "action_id") | replacements
    flags_by_action = {action_id: _action_flags(row) for action_id, row in actions.items()}

    entries: dict[int, UnitEntry] = {}
    panels: dict[int, tuple[float, ...]] = {}
    for npc in sorted(roster, key=lambda row: row["unit_id"]):
        uid = int(npc["unit_id"])
        unit = unit_data[uid]
        enhance_id = _number(npc, "enhance_id")
        baseline = baselines.get(enhance_id, {})
        form = rarity.get(uid, {})
        rank = promotion.get(uid, {})
        growth_level = (max_level or _number(baseline, "unit_level")) + (
            _number(rank, "promotion_level") or _number(baseline, "promotion_level")
        )
        panel = tuple(
            float(form.get(stat) or 0) + float(form.get(f"{stat}_growth") or 0) * growth_level
            + float(rank.get(stat) or 0)
            for stat in PANEL_COLUMNS
        )
        skill_ids = {
            skill_id
            for column, skill_id in unit_skills.get(uid, {}).items()
            if column != "unit_id" and isinstance(skill_id, int) and skill_id > 0
        }
        skill_ids |= {rf_id for skill_id in skill_ids for rf_id in rf_skills.get(skill_id, ())}
        action_ids = {
            action_id
            for skill_id in skill_ids
            for slot in range(1, 11)
            if (action_id := _number(skills.get(skill_id, {}), f"action_{slot}")) > 0
        }
        action_flags = [flags_by_action[aid] for aid in action_ids if aid in flags_by_action]
        flags = {
            name: any(values[i] for values in action_flags)
            for i, name in enumerate(UNIT_MECH_FLAG_NAMES)
        }
        entries[uid] = UnitEntry(
            unit_id=uid,
            name=str(unit.get("unit_name") or ""),
            search_area_width=_number(unit, "search_area_width"),
            atk_type=_number(unit, "atk_type"),
            talent_id=_number(talents.get(uid, {}), "talent_id"),
            role_id=_number(roles.get(uid, {}), "unit_role_id"),
            npc_enhance_id=enhance_id,
            has_6star=_number(form, "rarity") == 6,
            has_ue1=1 in equips.get(uid, ()),
            has_ue2=2 in equips.get(uid, ()),
            has_labyrinth_replace=not action_ids.isdisjoint(replacements),
            **{f"panel_{stat}": value for stat, value in zip(PANEL_COLUMNS, panel)},
            mech_flags=flags,
        )
        panels[uid] = (*map(_log, panel[:3]), *panel[3:])

    norms = _normalizers(panels.values())
    for uid, entry in entries.items():
        entry.static_feats = [
            entry.search_area_width / 1000.0, float(entry.atk_type == 1), float(entry.atk_type == 2),
            float(entry.has_6star), float(entry.has_ue1), float(entry.has_ue2),
            float(entry.has_labyrinth_replace), *_standardize(panels[uid], norms),
            *(float(entry.mech_flags[name]) for name in UNIT_MECH_FLAG_NAMES),
        ]
    return entries


def _read_enemies(db: sqlite3.Connection) -> dict[int, EnemyEntry]:
    ailments = _rows(db, "ailment_data")
    has_ailment_table = bool(ailments) or db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ailment_data'"
    ).fetchone() is not None
    if has_ailment_table:
        dot_ids = {int(r["ailment_id"]) for r in ailments if r["ailment_action"] in {9, 79}}
        cc_ids = {
            int(r["ailment_id"]) for r in ailments if r["ailment_action"] in CC_AILMENT_ACTIONS
            and (r["ailment_action"], _number(r, "ailment_detail_1")) != (8, 2)
        }
    else:
        dot_ids, cc_ids = STANDARD_DOT_AILMENT_IDS, STANDARD_CC_AILMENT_IDS
    resists = _index(_rows(db, "resist_data"), "resist_status_id")
    enemies: dict[int, EnemyEntry] = {}
    for row in _rows(db, "labyrinth_enemy_parameter", required=True):
        eid = int(row["enemy_id"])
        resist_id = _number(row, "resist_status_id")
        resist = resists.get(resist_id, {})
        enemies[eid] = EnemyEntry(
            enemy_id=eid,
            name=str(row.get("name") or ""),
            physical_def=_number(row, "def"),
            resist_status_id=resist_id,
            has_dot_resist=any(float(resist.get(f"ailment_{i}") or 0) > 0 for i in dot_ids),
            has_cc_resist=any(float(resist.get(f"ailment_{i}") or 0) > 0 for i in cc_ids),
            **{col: _number(row, col) for col in (
                "unit_id", "level", "hp", "virtual_hp", "atk", "magic_str", "magic_def", "dodge", "accuracy"
            )},
        )
    return enemies


def _normalize_enemies(enemies: dict[int, EnemyEntry], environment: set[int]) -> None:
    panels = {
        eid: (_log(e.hp), _log(e.atk), _log(e.magic_str), e.physical_def, e.magic_def, e.dodge, e.accuracy)
        for eid, e in enemies.items()
    }
    norms = _normalizers(panel for eid, panel in panels.items() if eid not in environment)
    for eid, e in enemies.items():
        e.static_feats = [
            e.level / 100.0, *_standardize(panels[eid], norms), float(e.is_boss_unit),
            float(e.has_dot_resist), float(e.has_cc_resist),
        ]


def _read_quests(
    db: sqlite3.Connection, enemies: dict[int, EnemyEntry], environment: set[int]
) -> tuple[dict[int, QuestEntry], dict[str, list[int]]]:
    waves = _index(_rows(db, "labyrinth_wave_group_data", required=True), "wave_group_id")
    displays = _index(_rows(db, "labyrinth_quest_data_display"), "quest_id")
    ex_displays = _index(_rows(db, "labyrinth_ex_quest_data_display"), "quest_id")
    quests: dict[int, QuestEntry] = {}
    ex_index: dict[str, list[int]] = defaultdict(list)
    boss_levels: dict[int, int] = {}
    area_max_level: dict[int, int] = defaultdict(int)
    for row in sorted(_rows(db, "labyrinth_quest_data", required=True), key=lambda r: r["quest_id"]):
        qid = int(row["quest_id"])
        wave_id = int(row["wave_group_id"])
        ids = [_number(waves.get(wave_id, {}), slot) for slot in ENEMY_SLOTS]
        ids = [eid for eid in ids if eid and eid in enemies]
        valid = [enemies[eid] for eid in ids if eid not in environment]
        if not valid:
            continue
        area = (qid // 100000) % 100
        quest_type = int(row["quest_type"])
        names = sorted(e.name for e in valid)
        composition = sorted((e.unit_id, e.name) for e in valid)
        signature = hashlib.sha256(
            json.dumps(composition, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if quest_type == 1:
            bucket = "normal"
        elif quest_type == 2:
            ex_index[f"{area}:{'|'.join(names)}"].append(qid)
            if any("黄蜂女王" in name or "美空" in name for name in names):
                bucket = "special_dual_ex"
            elif any("·遗物" in name for name in names):
                bucket = "robot_ex"
            else:
                bucket = "other_ex"
        elif quest_type == 3:
            bucket = "boss_other"
            boss_levels[qid] = max(e.level for e in valid)
            area_max_level[area] = max(area_max_level[area], boss_levels[qid])
            for enemy in valid:
                enemy.is_boss_unit = True
        else:
            continue
        name = ex_displays.get(qid, {}).get("quest_name") or displays.get(qid, {}).get("quest_name")
        quests[qid] = QuestEntry(
            quest_id=qid, wave_group_id=wave_id, quest_type=quest_type, area=area,
            limit_time=_number(row, "limit_time"), reward_id=_number(row, "reward_id"),
            enemy_ids=ids, wave_total_hp=sum(e.hp for e in valid), canonical_wave_sig=signature,
            bucket=bucket, quest_name=str(name or ""),
        )

    top_boss_area = max(area_max_level, default=0)
    for qid, level in boss_levels.items():
        quest = quests[qid]
        if level == area_max_level[quest.area]:
            quest.is_difficulty5_boss = True
            floor = 5 if quest.area == top_boss_area and quest.area != 3 else quest.area
            quest.bucket = f"boss_{floor}f_d5"
    return quests, dict(ex_index)


def _matched_number(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    return float(match[1]) if match else 0.0


def _read_relics(db: sqlite3.Connection) -> dict[int, RelicEntry]:
    relics: dict[int, RelicEntry] = {}
    for row in _rows(db, "labyrinth_relic", required=True):
        rid = int(row["relic_id"])
        effect = str(row.get("effect_description") or row.get("description") or "")
        description = effect + "\n" + str(row.get("description") or "")
        amp = {30001: 1.0, 40002: 0.5}.get(rid)
        discount = _matched_number(r"商店的贩卖价格\s*[－-]\s*(\d+(?:\.\d+)?)\s*[%％]", description)
        relics[rid] = RelicEntry(
            relic_id=rid, name=str(row.get("name") or ""), effect_description=effect,
            **{col: _number(row, col) for col in ("rarity", "relic_mark_id", "relic_mark_count", "price")},
            gold_scaling=("阿尔法金币持有数" in description
                          or ("金币" in description and ("持有" in description or "×" in description))),
            owned_unit_scaling="加入角色数" in description,
            front_pdef_to_atk_mult=_matched_number(r"前排[^\n×]*?物(?:理)?防(?:御力)?\s*×\s*(\d+)", description),
            front_mdef_to_atk_mult=_matched_number(r"前排[^\n×]*?魔(?:法)?防(?:御力)?\s*×\s*(\d+)", description),
            element_bonus_amp=amp if amp is not None else (
                _matched_number(r"属性加成\s*[＋+]\s*(\d+(?:\.\d+)?)\s*[%％]", description) / 100
            ),
            is_revive=rid == 50002 or "免除一次战斗败北" in description,
            is_shop_discount=rid == 40003 or discount > 0,
            shop_discount_rate=(discount or (10 if rid == 40003 else 0)) / 100,
        )
    return relics


def _read_scores(db: sqlite3.Connection) -> ScoreSettings:
    values = {int(row["setting_id"]): int(row["value"]) for row in _rows(db, "labyrinth_setting")}
    score = ScoreSettings()
    score.area_clear_pts = {area: values[200 + area] for area in range(1, 11) if 200 + area in values}
    score.block_pts = {area: values[210 + area] for area in range(1, 11) if 210 + area in values}
    for key, field in enumerate(("unit_pts", "relic_pts", "normal_pts", "ex_pts", "boss_pts"), 221):
        setattr(score, field, values.get(key, getattr(score, field)))
    score.shop_refresh_costs = [
        values.get(key, default) for key, default in enumerate(score.shop_refresh_costs, 101)
    ]
    score.score_rates = {
        int(row["difficulty"]): _number(row, "score_rate")
        for row in _rows(db, "labyrinth_difficulty_data")
    }
    return score


def _read_event_choices(db: sqlite3.Connection) -> dict[int, EventChoiceEntry]:
    choice_events = {
        int(choice): int(event["event_id"])
        for event in _rows(db, "labyrinth_event")
        for column, choice in event.items()
        if column.startswith("choice_id_") and choice
    }
    results: dict[int, list[int]] = defaultdict(list)
    for row in _rows(db, "labyrinth_event_result"):
        rid = int(row["event_result_id"])
        results[rid // 10].append(rid)
    return {
        cid: EventChoiceEntry(
            event_id=choice_events[cid], choice_id=cid, condition_type=_number(row, "condition_type"),
            condition_value=_number(row, "condition_value"), unit_id=_number(row, "unit_id"),
            description=str(row.get("description") or ""), result_ids=sorted(results[cid]),
        )
        for cid, row in _index(_rows(db, "labyrinth_event_choice"), "choice_id").items()
        if cid in choice_events
    }


def _read_training(db: sqlite3.Connection) -> dict[str, Any]:
    max_talent = max(
        (_number(row, "talent_level") for row in _rows(db, "experience_talent_level")), default=0
    )
    mastery_ids = {
        int(row["mastery_id"]) for row in _rows(db, "unit_role_mastery_id")
        if _number(row, "unit_role_id") != 0 and _number(row, "mastery_id") not in {0, 99999}
    }
    mastery: dict[str, list[int]] = {}
    for row in _rows(db, "unit_role_mastery_level"):
        mid = int(row["mastery_id"])
        if mid in mastery_ids:
            levels = mastery.setdefault(str(mid), [0, 0])
            levels[0] = max(levels[0], _number(row, "slot_level"))
            levels[1] = max(levels[1], _number(row, "enhance_level"))
    return {
        "talent_skill_levels": {
            str(row["talent_id"]): max_talent for row in _rows(db, "talent") if row["talent_id"]
        },
        "talent_skill_node_levels": {
            str(row["node_id"]): sum(
                bool(value) for col, value in row.items() if col.startswith("enhance_level_id_")
            )
            for row in _rows(db, "talent_skill_node")
        },
        "role_mastery": mastery,
    }


def _db_version(path: Path, digest: str, user_version: int) -> str:
    try:
        metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        metadata = {}
    if isinstance(metadata, dict) and metadata.get("fileSha256") in (None, digest):
        if metadata.get("version"):
            return str(metadata["version"])
    return str(user_version) if user_version else ""


def build_catalog(db_path: str | Path) -> FeatureCatalog:
    path = Path(db_path)
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        user_version = db.execute("PRAGMA user_version").fetchone()[0]
        units = _read_units(db)
        enemies = _read_enemies(db)
        environment = {
            eid for row in _rows(db, "labyrinth_difficulty_enemy_data") for slot in ENEMY_SLOTS
            if (eid := _number(row, slot)) != 0
        }
        quests, ex_index = _read_quests(db, enemies, environment)
        _normalize_enemies(enemies, environment)
        relics = _read_relics(db)
        guilds = {
            gid: GuildEntry(gid, str(row.get("guild_name") or ""),
                            _number(row, "guild_master_id"), _number(row, "add_relic_id"))
            for gid, row in _index(_rows(db, "labyrinth_enter_guild"), "guild_id").items()
        }
        score = _read_scores(db)
        nodes = sorted({int(row["node_id"]) for row in _rows(db, "labyrinth_enhance_node")})
        events = _read_event_choices(db)
        training = _read_training(db)

    with path.open("rb") as db_file:
        digest = hashlib.file_digest(db_file, "sha256").hexdigest()
    catalog = FeatureCatalog(
        schema_version=1, db_version=_db_version(path, digest, user_version), db_sha256=digest,
        catalog_sha256="", max_area=max((q.area for q in quests.values()), default=0),
        max_difficulty=max(score.score_rates, default=5), units=units, enemies=enemies, quests=quests,
        relics=relics, guilds=guilds, score_settings=score, enhance_node_ids=nodes,
        environment_enemy_ids=sorted(environment), ex_disambiguation_index=ex_index,
        event_choices=events, max_account_training=training,
    )
    catalog.finalize_hash()
    return catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    catalog = build_catalog(args.db)
    catalog.save(args.out)
    print(json.dumps({
        "out": str(args.out), "db_version": catalog.db_version,
        "units": len(catalog.units), "enemies": len(catalog.enemies),
        "quests": len(catalog.quests), "relics": len(catalog.relics),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
