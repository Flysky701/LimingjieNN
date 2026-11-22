"""Check a PCRCalculatorServer against a catalog and measure batch throughput."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from labyrinth_nn.catalog.schema import FeatureCatalog
from labyrinth_nn.data.sample_schema import FRAMES_PER_SECOND, SingleRunResult, outcome_class_from_sim
from labyrinth_nn.data.sampler import GOLD_BOUNDARY, OWNED_BOUNDARY, QUEST_BUCKETS, quest_stratum
from labyrinth_nn.data.sim_client import (
    DIFFICULTY_5_ENVIRONMENT_ENEMY_ID,
    PcrCalculatorClient,
    SimClient,
    SimulationConfig,
    SimulationError,
    build_simulate_payload,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run_probe(client: PcrCalculatorClient, bench_count: int = 64) -> dict[str, Any]:
    if bench_count < 0:
        raise ValueError("bench_count must be nonnegative")
    catalog = client.catalog
    if catalog is None:
        raise ValueError("the simulator probe requires a catalog")
    checks: list[dict[str, Any]] = []
    cases: dict[str, SimulationConfig] = {}
    candidates = [
        q for q in catalog.quests.values() if q.wave_total_hp > 0 and q.limit_time > 0
        and any(eid and eid not in client.environment for eid in q.enemy_ids)
    ]
    team = catalog.sorted_unit_ids()[:5]
    for bucket in QUEST_BUCKETS:
        matches = [q for q in candidates if quest_stratum(q) == bucket]
        if matches and team:
            quest = min(matches, key=lambda q: q.quest_id)
            cases[bucket] = SimulationConfig(quest, team, difficulty=5)

    def check(name: str, run: Callable[[], dict[str, Any]]) -> bool:
        try:
            details = run()
        except (OSError, ValueError, SimulationError) as exc:
            checks.append({"name": name, "passed": False, "error": str(exc)})
            return False
        checks.append({"name": name, "passed": True, "details": details})
        return True

    def strata() -> dict[str, Any]:
        missing = [bucket for bucket in QUEST_BUCKETS if bucket not in cases]
        _require(not missing, f"missing difficulty-5 strata: {', '.join(missing)}")
        return {"quest_ids": {bucket: config.quest.quest_id for bucket, config in cases.items()}}

    check("difficulty5_quest_strata", strata)
    baselines = []
    fixtures = [
        replace(case, runs=2 if i == 0 else 1,
                run_seeds=(12301, 12302) if i == 0 else (12302 + i,))
        for i, case in enumerate(cases.values())
    ]

    def ordered_batch() -> dict[str, Any]:
        _require(bool(fixtures), "no battle fixtures available")
        results = client.simulate_batch_single_runs(fixtures)
        seeds = [[r.seed for r in runs] for runs in results]
        _require(seeds == [list(c.run_seeds) for c in fixtures], "batch response order or run count differs")
        baselines.extend(results)
        return {"run_counts": [len(runs) for runs in results], "seeds": seeds}

    connected = check("batch_order_and_results", ordered_batch)
    baseline = next(iter(cases.values()), None)
    fixed = replace(baseline, use_fixed_random_seed=True, random_seed=12301) if baseline else None

    def summary(run: SingleRunResult) -> dict[str, Any]:
        return {"seed": run.seed, "result": run.result, "battle_frames": run.battle_frames,
                "total_damage": run.total_damage, "damage_ratio": run.damage_ratio}

    def single_runs(*payloads: dict[str, Any]) -> list[SingleRunResult]:
        groups = client.simulate_batch_single_runs((payload, 1) for payload in payloads)
        _require(all(len(group) == 1 for group in groups), "expected one result per request")
        runs = [group[0] for group in groups]
        _require([run.seed for run in runs] == [p["Settings"]["RandomSeed"] for p in payloads],
                 "fixed seeds did not match the requests")
        return runs

    def damage_accounting(payload: dict[str, Any], run: SingleRunResult) -> dict[str, Any]:
        enemy_ids = payload["GuildEnemyList"]
        _require(client.environment.isdisjoint(enemy_ids), "environment enemy present in GuildEnemyList")
        _require(all(eid in catalog.enemies for eid in enemy_ids), "catalog lacks enemy HP for comparison")
        initial_hps = [catalog.enemies[eid].hp for eid in enemy_ids]
        initial_hp = sum(initial_hps)
        _require(initial_hp > 0, "non-environment enemies must have positive total HP")
        finals = run.enemy_final_hps
        if finals:
            _require(len(finals) == len(enemy_ids), "final HP count differs from visible enemies")
            lost_hp = sum(max(0, hp - final) for hp, final in zip(initial_hps, finals, strict=True))
        else:
            lost_hp = initial_hp if run.result == 0 else run.total_damage
        expected = max(0.0, min(1.0, lost_hp / initial_hp))
        _require(0.0 <= run.damage_ratio <= 1.0 and run.damage_ratio == expected,
                 "damage ratio differs from non-environment enemy HP accounting")
        return {"enemy_ids": enemy_ids, "non_environment_initial_hp": initial_hp,
                "enemy_final_hps": finals, "used_aggregate_damage": not finals and run.result != 0,
                "damage_ratio": run.damage_ratio}

    def fixed_replay() -> dict[str, Any]:
        replay = client.simulate_batch_single_runs([fixed, fixed])
        _require(replay[0][0] == replay[1][0] == baselines[0][0], "fixed-seed replay differs")
        return {"seed": replay[0][0].seed, "result": replay[0][0].result}

    def independent_runs() -> dict[str, Any]:
        config = replace(baseline, runs=4, use_fixed_random_seed=False)
        runs, = client.simulate_batch_single_runs([config])
        seeds = [run.seed for run in runs]
        _require(len(seeds) == len(set(seeds)) == config.runs, "independent runs returned duplicate seeds")
        return {"seeds": seeds}

    def environment_hp() -> dict[str, Any]:
        injected = replace(fixed.quest, enemy_ids=[*fixed.quest.enemy_ids, *sorted(client.environment)])
        config = replace(fixed, quest=injected)
        request = client.build_request(config)
        _require(request == client.build_request(fixed), "an environment enemy changed the request")
        replay, = client.simulate_batch_single_runs([config])
        _require(replay[0] == baselines[0][0], "environment enemies changed the parsed battle")
        reported_hp = 0
        aggregate_only = 0
        for fixture, runs in zip(fixtures, baselines, strict=True):
            size = len(client.build_request(fixture)["GuildEnemyList"])
            for run in runs:
                _require(len(run.enemy_final_hps) in {0, size}, "final HP count differs from visible enemies")
                if run.enemy_final_hps:
                    reported_hp += int(run.result != 0)
                elif run.result != 0:
                    aggregate_only += 1
        return {"excluded_enemy_ids": sorted(client.environment),
                "non_win_runs_with_final_hp": reported_hp, "non_win_runs_with_aggregate_only": aggregate_only}

    def outcome_and_damage() -> dict[str, Any]:
        outcomes = []
        for runs in baselines:
            for r in runs:
                _require(
                    r.outcome_class == outcome_class_from_sim(r.result, r.battle_frames, r.limit_time)
                    and r.elapsed_sec == r.battle_frames / FRAMES_PER_SECOND
                    and 0 <= r.rem_time_ratio <= 1 and 0 <= r.damage_ratio <= 1 and r.total_damage >= 0,
                    "outcome, frames or damage is out of bounds",
                )
                outcomes.append({"result": r.result, "frames": r.battle_frames,
                                 "total_damage": r.total_damage, "damage_ratio": r.damage_ratio})
        return {"outcomes": outcomes}

    def account_training() -> dict[str, Any]:
        payload = client.build_request(fixed)
        lean_profile = client.profile | {
            "use_all_enhance_nodes": False, "fill_max_account_training_from_db": False,
            "talent_skill_levels_override": {}, "talent_skill_node_levels_override": {}, "role_mastery_override": {},
        }
        lean_client = SimClient(client.url.removesuffix("/simulate"), catalog, lean_profile)
        lean_payload = lean_client.build_request(fixed)
        expected_nodes = catalog.enhance_node_ids if client.profile.get("use_all_enhance_nodes", True) else []
        _require(payload["LabyrinthEnhancedNodeIds"] == expected_nodes
                 and lean_payload["LabyrinthEnhancedNodeIds"] == [], "enhancement node payload differs")
        for field, key in (("TalentSkillLevels", "talent_skill_levels"),
                           ("TalentSkillNodeLevels", "talent_skill_node_levels"), ("RoleMastery", "role_mastery")):
            expected = (catalog.max_account_training.get(key, {})
                        if client.profile.get("fill_max_account_training_from_db", True) else {})
            expected = {str(int(k)) for k in expected} | {
                str(int(k)) for k in client.profile.get(f"{key}_override", {})
            }
            _require(set(payload[field]) == expected and not lean_payload[field], f"{field} keys differ")
        forbidden = {"MasterSkillNodeLevels", "EnemyMasterSkillNodeLevels"}
        _require(forbidden.isdisjoint(payload) and forbidden.isdisjoint(lean_payload),
                 "unexpected master skill field in request")
        runs, = lean_client.simulate_batch_single_runs([fixed])
        return {"nodes": len(expected_nodes), "talents": len(payload["TalentSkillLevels"]),
                "talent_nodes": len(payload["TalentSkillNodeLevels"]), "role_masteries": len(payload["RoleMastery"]),
                "cleared_training_result": runs[0].result}

    def inventory_boundaries() -> dict[str, Any]:
        gold_relic = next((r.relic_id for r in catalog.relics.values() if r.gold_scaling), None)
        owned_relic = next((r.relic_id for r in catalog.relics.values() if r.owned_unit_scaling), None)
        _require(gold_relic is not None and owned_relic is not None, "catalog lacks gold or roster scaling relics")
        roster = catalog.sorted_unit_ids()
        _require(len(roster) >= max(OWNED_BOUNDARY), "catalog lacks enough units for 24/25/26 probes")
        configs = [fixed]
        configs.extend(replace(fixed, relic_ids=(gold_relic,), alpha_rupee=gold) for gold in GOLD_BOUNDARY)
        configs.extend(replace(fixed, relic_ids=(owned_relic,), owned_unit_ids=roster[:count])
                       for count in OWNED_BOUNDARY)
        requests = [client.build_request(config) for config in configs]
        _require(requests[0]["LabyrinthRelicIds"] == [] and requests[0]["LabyrinthAlphaRupee"] == 0,
                 "empty inventory did not serialize as zero relics and zero gold")
        _require([r["LabyrinthAlphaRupee"] for r in requests[1:4]] == list(GOLD_BOUNDARY),
                 "gold boundary values changed in transit")
        _require([len(r["LabyrinthOwnedUnitIds"]) for r in requests[4:]] == list(OWNED_BOUNDARY),
                 "owned-unit boundary counts changed in transit")
        _require(all(set(team) <= set(r["LabyrinthOwnedUnitIds"]) and r["LabyrinthRelicIds"] == list(c.relic_ids)
                     for r, c in zip(requests, configs, strict=True)), "team ownership or relic payload differs")
        runs = client.simulate_batch_single_runs(configs)
        return {"gold_values": list(GOLD_BOUNDARY), "owned_counts": list(OWNED_BOUNDARY),
                "gold_relic_id": gold_relic, "owned_relic_id": owned_relic,
                "damage_by_case": [group[0].total_damage for group in runs]}

    def npc_rarities() -> dict[str, Any]:
        payloads = []
        units = [catalog.units[uid] for uid in catalog.sorted_unit_ids()]
        for rarity, enhance_id in ((3, 1001), (6, 1002)):
            unit = next((u for u in units if u.npc_enhance_id == enhance_id), None)
            if unit is None:
                unit = next((u for u in units if u.npc_enhance_id not in (1001, 1002)
                             and u.has_6star == (rarity == 6)), None)
            _require(unit is not None, f"catalog lacks a {rarity}-star NPC")
            payload = client.build_request(replace(fixed, team_unit_ids=(unit.unit_id,)))
            # ToMaxUnitAttr would overwrite the requested rarity with the unit's maximum rarity.
            payload["ToMaxUnitAttr"] = False
            payload["PlayerList"][0].update(rarity=rarity, level=1, skillLevel=[1, 1, 1, 1])
            payloads.append(payload)
        runs = single_runs(*payloads)
        return {f"npc_{p['PlayerList'][0]['rarity']}star": {"unit_id": p["PlayerList"][0]["unitId"], **summary(run)}
                for p, run in zip(payloads, runs, strict=True)}

    def skill_replacement() -> dict[str, Any]:
        unit_id = next((uid for uid in catalog.sorted_unit_ids() if catalog.units[uid].has_labyrinth_replace), None)
        _require(unit_id is not None, "catalog lacks a unit with Labyrinth skill-action replacements")
        replacement_team = [unit_id, *(uid for uid in catalog.sorted_unit_ids() if uid != unit_id)][:5]
        run, = single_runs(client.build_request(replace(fixed, team_unit_ids=replacement_team)))
        return {"unit_id": unit_id, "team_unit_ids": replacement_team, **summary(run)}

    def auto_vs_set() -> dict[str, Any]:
        payloads = [
            build_simulate_payload(
                catalog, client.profile | {"is_auto_mode": auto, "is_set_mode": not auto, "is_seman_mode": False},
                fixed.quest, fixed.team_unit_ids, difficulty=5, use_fixed_random_seed=True, seed=fixed.random_seed,
            )
            for auto in (True, False)
        ]
        auto_run, set_run = single_runs(*payloads)
        return {"auto": summary(auto_run), "set": summary(set_run)}

    def difficulty5_environment() -> dict[str, Any]:
        enemy_id = DIFFICULTY_5_ENVIRONMENT_ENEMY_ID
        quest = replace(fixed.quest, enemy_ids=[*fixed.quest.enemy_ids, enemy_id])
        payload = client.build_request(replace(fixed, quest=quest))
        _require(payload["LabyrinthDifficulty"] == 5 and enemy_id not in payload["GuildEnemyList"],
                 "difficulty-5 environment enemy present in request")
        _require(payload == client.build_request(fixed), "difficulty-5 environment enemy changed the request")
        run, = single_runs(payload)
        _require(run == baselines[0][0], "difficulty-5 environment enemy changed the battle result")
        return {"excluded_enemy_id": enemy_id, **damage_accounting(payload, run)}

    def gold_relic() -> dict[str, Any]:
        relic = catalog.relics.get(50003)
        _require(relic is not None and relic.gold_scaling, "catalog lacks gold-scaling relic 50003")
        gold_values = (0, 10000)
        payloads = [client.build_request(replace(fixed, relic_ids=(50003,), alpha_rupee=gold)) for gold in gold_values]
        _require(all(p["LabyrinthRelicIds"] == [50003] and p["LabyrinthAlphaRupee"] == gold
                     for p, gold in zip(payloads, gold_values, strict=True)), "gold relic comparison changed in transit")
        return {"relic_id": 50003, "cases": [{"alpha_rupee": gold, **summary(run)}
                                             for gold, run in zip(gold_values, single_runs(*payloads), strict=True)]}

    def owned_relic() -> dict[str, Any]:
        relic = catalog.relics.get(50001)
        _require(relic is not None and relic.owned_unit_scaling, "catalog lacks roster-scaling relic 50001")
        roster = catalog.sorted_unit_ids()
        _require(len(roster) >= 25, "catalog lacks enough units for the 5/25 owned-unit comparison")
        owned_counts = (5, 25)
        payloads = [client.build_request(replace(fixed, relic_ids=(50001,), owned_unit_ids=roster[:count]))
                    for count in owned_counts]
        _require(all(p["LabyrinthRelicIds"] == [50001] and p["LabyrinthOwnedUnitIds"] == roster[:count]
                     for p, count in zip(payloads, owned_counts, strict=True)), "owned relic comparison changed in transit")
        return {"relic_id": 50001, "cases": [{"owned_unit_count": count, **summary(run)}
                                             for count, run in zip(owned_counts, single_runs(*payloads), strict=True)]}

    def boss_damage_ratio() -> dict[str, Any]:
        boss = max((q for q in candidates if q.bucket == "boss_5f_d5" and q.is_difficulty5_boss),
                   key=lambda q: (q.wave_total_hp, -q.quest_id), default=None)
        _require(boss is not None, "catalog lacks a boss_5f_d5 quest")
        payload = client.build_request(replace(fixed, quest=boss))
        run, = single_runs(payload)
        return {"quest_id": boss.quest_id, "result": run.result, **damage_accounting(payload, run)}

    def order_swap() -> dict[str, Any]:
        seeds = (12401, 12402)
        payloads = [client.build_request(replace(fixed, random_seed=seed)) for seed in seeds]
        forward = single_runs(*payloads)
        reverse = single_runs(*reversed(payloads))

        def signature(run: SingleRunResult) -> tuple[int, int, int, int]:
            return run.seed, run.result, run.battle_frames, run.total_damage

        signatures = [signature(run) for run in forward]
        reversed_signatures = [signature(run) for run in reverse]
        _require(signatures == reversed_signatures[::-1],
                 f"battle results changed when request order was swapped: {signatures} vs {reversed_signatures[::-1]}")
        return {"forward": signatures, "reverse": reversed_signatures}

    for name, callback in (
        ("fixed_seed_replay", fixed_replay),
        ("independent_single_runs", independent_runs),
        ("non_environment_hp", environment_hp),
        ("outcome_damage_frames", outcome_and_damage),
        ("enhancement_and_account_training", account_training),
        ("relic_owned_gold_boundaries", inventory_boundaries),
        ("npc_3star_vs_6star", npc_rarities),
        ("labyrinth_skill_action_replace", skill_replacement),
        ("auto_vs_set", auto_vs_set),
        ("env_enemy_770900501", difficulty5_environment),
        ("gold_relic_50003", gold_relic),
        ("owned_relic_50001", owned_relic),
        ("boss_5f_d5_damage_ratio", boss_damage_ratio),
        ("order_swap_invariance", order_swap),
    ):
        if connected:
            check(name, callback)
        else:
            checks.append({"name": name, "passed": False, "skipped": True,
                           "error": "batch order check did not return verified battle results"})

    benchmark: dict[str, Any] = {"requested_runs": bench_count}
    if not bench_count or not connected:
        benchmark["status"] = "skipped"
        benchmark["reason"] = "bench-count is zero" if not bench_count else "no verified baseline batch"
    else:
        options = list(cases.values())
        start = perf_counter()
        try:
            runs = client.simulate_batch_single_runs([
                options[i % len(options)] for i in range(bench_count)
            ])
        except SimulationError as exc:
            benchmark.update(status="failed", error=str(exc))
        else:
            elapsed = perf_counter() - start
            count = sum(map(len, runs))
            benchmark.update(status="passed", completed_runs=count, elapsed_sec=elapsed, sims_per_sec=count / elapsed)
    return {
        "url": client.url, "db_version": catalog.db_version, "checks": checks, "benchmark": benchmark,
        "passed": all(c["passed"] for c in checks) and benchmark["status"] != "failed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path("configs/sim_profile_highscore_cn.json"))
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--bench-count", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        catalog = FeatureCatalog.load(args.catalog)
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        report = run_probe(SimClient(args.url, catalog, profile), args.bench_count)
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
