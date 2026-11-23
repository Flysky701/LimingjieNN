"""Standard-library client for PCRCalculatorServer's batched /simulate API."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from labyrinth_nn.catalog.schema import EnemyEntry, FeatureCatalog, QuestEntry
from labyrinth_nn.data.sample_schema import (
    FRAMES_PER_SECOND,
    SingleRunResult,
    outcome_class_from_sim,
)

RELIC_EFFECT_ENEMY_ID = 770999901
DIFFICULTY_5_ENVIRONMENT_ENEMY_ID = 770900501
FINAL_HP_FIELDS = ("finalhp", "currenthp", "remaininghp", "currenthitpoint")
INITIAL_HP_FIELDS = ("initialhp", "maxhp", "starthp", "maxhitpoint")


class SimulationError(RuntimeError):
    """The server did not return a usable battle result."""


class SimulationTimeoutError(SimulationError):
    """The simulator HTTP request timed out while the server may still be running."""


def _is_timeout_error(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or (isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError))


@dataclass(frozen=True)
class SimulationConfig:
    quest: QuestEntry
    team_unit_ids: Sequence[int]
    owned_unit_ids: Sequence[int] = ()
    relic_ids: Sequence[int] = ()
    alpha_rupee: int | None = None
    difficulty: int | None = None
    runs: int = 1
    calc_stats: int = 1
    need_once_result: bool = True
    use_fixed_random_seed: bool | None = None
    random_seed: int | None = None
    run_seeds: Sequence[int] | None = None


def environment_enemy_ids(catalog: FeatureCatalog | None, profile: Mapping[str, Any]) -> set[int]:
    return (
        set(catalog.environment_enemy_ids if catalog is not None else ())
        | {int(eid) for eid in profile.get("exclude_environment_enemy_ids", ())}
        | {RELIC_EFFECT_ENEMY_ID, DIFFICULTY_5_ENVIRONMENT_ENEMY_ID}
    )


def _training(
    catalog: FeatureCatalog, profile: Mapping[str, Any], key: str, override: str
) -> dict[str, Any]:
    defaults = (
        catalog.max_account_training.get(key, {})
        if profile.get("fill_max_account_training_from_db", True) else {}
    )
    return {str(int(k)): v for k, v in defaults.items()} | {
        str(int(k)): v for k, v in profile.get(override, {}).items()
    }


def build_simulate_payload(
    catalog: FeatureCatalog,
    profile: Mapping[str, Any],
    quest_id: int | QuestEntry | None = None,
    team_unit_ids: Sequence[int] = (),
    *,
    quest: QuestEntry | None = None,
    owned_unit_ids: Sequence[int] | None = None,
    relic_ids: Sequence[int] | None = None,
    alpha_rupee: int | None = None,
    enhance_node_ids: Sequence[int] | None = None,
    difficulty: int | None = None,
    calc_stats: int = 1,
    need_once_result: bool = True,
    use_fixed_random_seed: bool | None = None,
    seed: int | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Build one wire request. Repetition is represented by separate requests."""
    if isinstance(quest_id, QuestEntry):
        if quest is not None and quest != quest_id:
            raise ValueError("conflicting quests")
        quest, quest_id = quest_id, None
    if quest is None:
        if quest_id is None:
            raise ValueError("quest_id or quest is required")
        quest = catalog.quests[int(quest_id)]
    elif quest_id is not None and int(quest_id) != quest.quest_id:
        raise ValueError("quest_id does not match quest")
    if seed is not None and random_seed is not None and int(seed) != int(random_seed):
        raise ValueError("seed and random_seed must agree")
    if seed is None:
        seed = random_seed if random_seed is not None else profile.get("random_seed", 666)
    if enhance_node_ids is None:
        enhance_node_ids = catalog.enhance_node_ids if profile.get("use_all_enhance_nodes", True) else ()

    environment = environment_enemy_ids(catalog, profile)
    team = list(map(int, team_unit_ids))
    enemies = [int(eid) for eid in quest.enemy_ids if eid and int(eid) not in environment]
    if not team or not enemies:
        raise ValueError("a simulation requires a team and at least one non-environment enemy")
    return {
        "PlayerList": [{"unitId": uid} for uid in team],
        "GuildEnemyList": enemies,
        "WaveGroupId": int(quest.wave_group_id),
        "IsGuildBattle": True,
        "IsSkipping": True,
        "BattleTime": int(quest.limit_time),
        "ToMaxUnitAttr": bool(profile.get("to_max_unit_attr", True)),
        "IsAutoMode": bool(profile.get("is_auto_mode", True)),
        "IsSetMode": bool(profile.get("is_set_mode", False)),
        "IsSemanMode": bool(profile.get("is_seman_mode", False)),
        "CalcStats": int(calc_stats),
        "NeedOnceResult": bool(need_once_result),
        "RepeatCount": 1,
        "Settings": {
            "UseFixedRandomSeed": bool(profile.get("use_fixed_random_seed", False)
                                       if use_fixed_random_seed is None else use_fixed_random_seed),
            "RandomSeed": int(seed),
            "ForceNoCritical_player": False,
            "ForceNoCritical_enemy": False,
            "ForceIgnoreDodge_player": False,
            "ForceIgnoreDodge_enemy": False,
            "ForceCritical_player": False,
            "randomSpecialDatas": [],
        },
        "TalentSkillLevels": {
            k: int(v) for k, v in _training(catalog, profile, "talent_skill_levels", "talent_skill_levels_override").items()
        },
        "TalentSkillNodeLevels": {
            k: int(v) for k, v in _training(
                catalog, profile, "talent_skill_node_levels", "talent_skill_node_levels_override"
            ).items()
        },
        "RoleMastery": {
            k: [int(slot), int(level)] for k, (slot, level) in _training(
                catalog, profile, "role_mastery", "role_mastery_override"
            ).items()
        },
        "LabyrinthEnhancedNodeIds": list(map(int, enhance_node_ids)),
        "LabyrinthRelicIds": list(map(int, relic_ids or ())),
        "LabyrinthOwnedUnitIds": list(dict.fromkeys([*map(int, owned_unit_ids or ()), *team])),
        "LabyrinthAlphaRupee": int(alpha_rupee if alpha_rupee is not None else 0),
        "LabyrinthQuestId": int(quest.quest_id),
        "LabyrinthDifficulty": int(profile.get("difficulty", 5) if difficulty is None else difficulty),
    }


build_simulation_request = build_simulate_payload


def _clip_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


class PcrCalculatorClient:
    def __init__(
        self, base_url: str, timeout_sec: float = 120.0, max_retries: int = 2, *,
        catalog: FeatureCatalog | None = None, profile: Mapping[str, Any] | None = None,
        retry_on_timeout: bool = False,
    ):
        self.url = f"{base_url.rstrip('/')}/simulate"
        self.catalog = catalog
        self.profile = dict(profile or {})
        self.timeout = float(timeout_sec)
        self.max_retries = int(max_retries)
        self.retry_on_timeout = bool(retry_on_timeout)
        if self.timeout <= 0 or self.max_retries < 0:
            raise ValueError("timeout_sec must be positive and max_retries nonnegative")
        self.environment = environment_enemy_ids(catalog, self.profile)

    def build_request(self, config: SimulationConfig, *, catalog: FeatureCatalog | None = None) -> dict[str, Any]:
        catalog = self.catalog if catalog is None else catalog
        if catalog is None:
            raise ValueError("a catalog is required to build a SimulationConfig request")
        return build_simulate_payload(
            catalog, self.profile, config.quest, config.team_unit_ids,
            owned_unit_ids=config.owned_unit_ids, relic_ids=config.relic_ids,
            alpha_rupee=config.alpha_rupee, difficulty=config.difficulty,
            calc_stats=config.calc_stats, need_once_result=config.need_once_result,
            use_fixed_random_seed=config.use_fixed_random_seed, random_seed=config.random_seed,
        )

    def _post(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        request = Request(
            self.url, method="POST", data=json.dumps(requests, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                break
            except HTTPError as exc:
                with exc:
                    if attempt == self.max_retries or not (exc.code in {408, 429} or exc.code >= 500):
                        raise SimulationError(f"{self.url}: HTTP {exc.code} {exc.reason}") from exc
            except (OSError, HTTPException) as exc:
                is_timeout = _is_timeout_error(exc)
                if (is_timeout and not self.retry_on_timeout) or attempt == self.max_retries:
                    error = SimulationTimeoutError if is_timeout else SimulationError
                    raise error(f"{self.url}: {exc}") from exc
            except ValueError as exc:
                raise SimulationError(f"{self.url}: invalid JSON response") from exc
        if not isinstance(data, list) or len(data) != len(requests):
            raise SimulationError(f"{self.url}: expected {len(requests)} response items")
        return data

    @staticmethod
    def _enemy_hps(
        items: list[dict[str, Any]], expected_ids: list[int], enemies: Mapping[int, EnemyEntry],
        excluded: set[int], excluded_unit_ids: set[int],
    ) -> tuple[list[int | None], list[int], bool]:
        initial_hps = [enemies[eid].hp if eid in enemies else None for eid in expected_ids]
        remaining = list(enumerate(expected_ids))
        finals: dict[int, int] = {}
        has_environment_hp = False
        rows = [{key.lower().replace("_", ""): value for key, value in item.items()} for item in items]
        # Use explicit enemy IDs before unit ID fallbacks and consume repeated enemy slots separately.
        rows.sort(key=lambda row: not row.get("enemyid"))
        for fields in rows:
            enemy_id = int(fields.get("enemyid") or 0)
            unit_id = int(fields.get("unitid") or 0)
            final_hp = next((fields[key] for key in FINAL_HP_FIELDS if fields.get(key) is not None), None)
            initial_hp = next((fields[key] for key in INITIAL_HP_FIELDS if fields.get(key) is not None), None)
            hp = fields.get("hp")
            if enemy_id in excluded or (not enemy_id and unit_id in excluded_unit_ids):
                has_environment_hp |= final_hp is not None or (hp is not None and (
                    initial_hp is not None or enemy_id in enemies
                ))
                continue
            if enemy_id:
                match = next((pair for pair in remaining if pair[1] == enemy_id), None)
            else:
                match = next((pair for pair in remaining if unit_id and (pair[1] == unit_id or (
                    pair[1] in enemies and enemies[pair[1]].unit_id == unit_id
                ))), None)
            if match is None:
                continue
            remaining.remove(match)
            slot = match[0]
            # In HP-bearing responses, plain HP can stand for either maximum or current HP.
            if final_hp is None and (initial_hp is not None or initial_hps[slot] is not None):
                final_hp = hp
            if initial_hps[slot] is None:
                initial_hp = initial_hp if initial_hp is not None else hp
                if initial_hp is not None:
                    initial_hps[slot] = max(0, int(initial_hp))
            if final_hp is not None:
                finals[slot] = max(0, int(final_hp))
        final_hps = [finals[i] for i in range(len(expected_ids))] if len(finals) == len(expected_ids) else []
        return initial_hps, final_hps, has_environment_hp

    def _parse(
        self, raw: dict[str, Any], request: dict[str, Any], quest: QuestEntry | None,
        catalog: FeatureCatalog | None, excluded: set[int], excluded_unit_ids: set[int],
    ) -> SingleRunResult:
        if raw.get("Success", True) is not True:
            raise SimulationError(str(raw.get("Message") or "server rejected the battle"))
        result = int(raw["Result"])
        frames = int(raw["BattleFrames"])
        seed = int(raw["CurrentSeed"])
        limit_time = int(request.get("BattleTime", 90))
        if result not in {0, 1, 2} or frames < 0 or limit_time <= 0:
            raise SimulationError(f"invalid result={result}, frames={frames}, limit={limit_time}")
        total = raw.get("TotalDamage")
        if total is None:
            total = (raw.get("Stats") or {}).get("TotalDamage")
        if total is None:
            total = (raw.get("OnceResult") or {}).get("currentDamage")
        enemy_ids = request["GuildEnemyList"]
        initial_hps, finals, has_environment_hp = self._enemy_hps(
            raw.get("EnemyList") or [], enemy_ids, catalog.enemies if catalog is not None else {},
            excluded, excluded_unit_ids,
        )
        known_hps = [hp for hp in initial_hps if hp is not None]
        if len(known_hps) == len(enemy_ids):
            initial_hp = sum(known_hps)
        elif catalog is not None and quest is not None and enemy_ids == [
            eid for eid in quest.enemy_ids if eid and eid not in catalog.environment_enemy_ids
        ]:
            initial_hp = quest.wave_total_hp
        else:
            raise SimulationError("missing initial HP for non-environment enemies in catalog and EnemyList")
        if finals:
            if len(known_hps) == len(enemy_ids):
                lost_hp = sum(max(0, hp - final) for hp, final in zip(known_hps, finals, strict=True))
            else:
                lost_hp = max(0, initial_hp - sum(finals))
        elif result == 0:
            finals = [0] * len(enemy_ids)
            lost_hp = initial_hp
        elif total is not None and not has_environment_hp:
            # v2.11.0 serializes configured units (including hidden effect units).
            # Its damage telemetry is tracked separately from those configurations.
            lost_hp = int(total)
        else:
            raise SimulationError("cannot account for non-environment damage without enemy HP or battle damage")
        elapsed = frames / FRAMES_PER_SECOND
        return SingleRunResult(
            seed=seed, result=result, outcome_class=outcome_class_from_sim(result, frames, limit_time),
            battle_frames=frames, limit_time=limit_time, elapsed_sec=elapsed,
            rem_time_ratio=_clip_ratio(1.0 - elapsed / limit_time),
            total_damage=int(total) if total is not None else lost_hp,
            damage_ratio=_clip_ratio(lost_hp / initial_hp) if initial_hp > 0 else 0.0,
            enemy_final_hps=finals,
        )

    def simulate_batch_single_runs(
        self, payloads_with_repeats: Iterable[tuple[Mapping[str, Any], int] | SimulationConfig], *,
        catalog: FeatureCatalog | None = None, environment_enemy_ids: Iterable[int] | None = None,
    ) -> list[list[SingleRunResult]]:
        """Post independent single-run requests and return one group per input item."""
        if catalog is None:
            catalog = self.catalog
        excluded = self.environment | set(catalog.environment_enemy_ids if catalog is not None else ())
        excluded.update(map(int, environment_enemy_ids or ()))
        excluded_unit_ids = excluded | {
            catalog.enemies[eid].unit_id for eid in excluded if catalog is not None and eid in catalog.enemies
        }
        requests: list[dict[str, Any]] = []
        destinations: list[tuple[int, QuestEntry | None]] = []
        groups: list[list[SingleRunResult]] = []
        for index, item in enumerate(payloads_with_repeats):
            groups.append([])
            config = item if isinstance(item, SimulationConfig) else None
            if config is None:
                payload, repeat_count = item
            else:
                repeat_count = config.runs
            run_seeds = config.run_seeds if config is not None else None
            if repeat_count < 0 or (run_seeds is not None and len(run_seeds) != repeat_count):
                raise ValueError("runs must be nonnegative and match the number of explicit seeds")
            if not repeat_count:
                continue
            if config is not None:
                payload = self.build_request(config, catalog=catalog)
            request = dict(payload)
            quest = config.quest if config is not None else (
                catalog.quests.get(int(request.get("LabyrinthQuestId") or 0)) if catalog is not None else None
            )
            enemy_ids = [eid for value in request["GuildEnemyList"] if (eid := int(value)) and eid not in excluded]
            if not enemy_ids:
                raise ValueError("a simulation requires at least one non-environment enemy")
            request.update(RepeatCount=1, GuildEnemyList=enemy_ids)
            for run_index in range(repeat_count):
                run_request = request
                if run_seeds is not None:
                    run_request = request | {
                        "Settings": request["Settings"] | {
                            "UseFixedRandomSeed": True, "RandomSeed": int(run_seeds[run_index]),
                        }
                    }
                requests.append(run_request)
                destinations.append((index, quest))
        if not requests:
            return groups
        for run_index, (raw, (index, quest), request) in enumerate(
            zip(self._post(requests), destinations, requests, strict=True)
        ):
            try:
                parsed = self._parse(raw, request, quest, catalog, excluded, excluded_unit_ids)
            except (AttributeError, KeyError, TypeError, ValueError, SimulationError) as exc:
                quest_id = request.get("LabyrinthQuestId", "unknown")
                raise SimulationError(f"response {run_index} (quest {quest_id}): {exc}") from exc
            groups[index].append(parsed)
        return groups


class SimClient(PcrCalculatorClient):
    """Compatibility client configured by the existing simulation profile."""

    def __init__(self, base_url: str, catalog: FeatureCatalog, profile: Mapping[str, Any]):
        super().__init__(base_url, profile.get("timeout_sec", 120.0), profile.get("max_retries", 2),
                         catalog=catalog, profile=profile,
                         retry_on_timeout=bool(profile.get("retry_on_timeout", False)))

    def simulate_batch_single_runs(
        self, payloads_with_repeats: Iterable[tuple[Mapping[str, Any], int] | SimulationConfig] | None = None, *,
        catalog: FeatureCatalog | None = None, environment_enemy_ids: Iterable[int] | None = None,
        configs: Iterable[SimulationConfig] | None = None,
    ) -> list[list[SingleRunResult]]:
        if configs is not None and payloads_with_repeats is not None:
            raise TypeError("provide either configs or payloads_with_repeats")
        inputs = configs if payloads_with_repeats is None else payloads_with_repeats
        if inputs is None:
            raise TypeError("a batch of simulation inputs is required")
        return super().simulate_batch_single_runs(inputs, catalog=catalog, environment_enemy_ids=environment_enemy_ids)
