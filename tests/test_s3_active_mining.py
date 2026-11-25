"""Offline acceptance tests for S3 active mining, S4 frozen splits, S5 reservation, and crash recovery."""
from __future__ import annotations

import errno
import json
import shutil
import tempfile
import threading
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import replace
from functools import cached_property
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from labyrinth_nn.active_mining.identity import (
    build_s2_holdout_registry,
    semantic_context_sha256,
    sha256_file,
    worker_for_context,
    worker_id_for_index,
    write_jsonl_atomic,
)
from labyrinth_nn.active_mining.s3_executor import run_worker_plan
from labyrinth_nn.active_mining.s3_planner import (
    S3PlannerSpec,
    generate_one_two_slot_mutations,
    is_overconfident_loss_anchor,
    load_calibrated_ensemble,
    plan_s3_mining,
    score_samples_with_ensemble,
    verify_s3_plan,
)
from labyrinth_nn.active_mining.s4_frozen_train import (
    evaluate_s4_distillation_gate,
    partition_frozen_samples,
)
from labyrinth_nn.active_mining.s5_evaluator import quest_only_baseline_records
from labyrinth_nn.active_mining.s5_reservation import (
    S5Denylist,
    S5ReservationSpec,
    build_s5_reservation,
)
from labyrinth_nn.catalog.schema import (
    EnemyEntry,
    FeatureCatalog,
    GuildEntry,
    QuestEntry,
    RelicEntry,
    ScoreSettings,
    UnitEntry,
)
from labyrinth_nn.data.sample_schema import BattleSample, read_samples
from labyrinth_nn.data.sampler import _sync_file_atomic
from labyrinth_nn.data.sim_client import SimClient, SimulationTimeoutError

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from labyrinth_nn.models.critic import CriticConfig, TwoTowerBattleCritic, TwoTowerEnsemble


@contextmanager
def mock_sim_server(catalog: FeatureCatalog, respond=None):
    requests: list[list[dict]] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/simulate":
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append(payload)
                batch_no = len(requests)
                result = []
                for idx, query in enumerate(payload):
                    seed = int(query["Settings"]["RandomSeed"])
                    enemies = [
                        {"enemyid": eid, "currentHp": catalog.enemies[eid].hp // 2}
                        for eid in query["GuildEnemyList"]
                    ]
                    row = {
                        "Success": True,
                        "EnemyList": enemies,
                        "Result": 0 if idx % 2 == 0 else 2,
                        "BattleFrames": 1800 if idx % 2 == 0 else 5399,
                        "CurrentSeed": seed,
                        "Stats": {"TotalDamage": 10000},
                    }
                    result.append(respond(query, row, batch_no, idx) if respond is not None else row)
            encoded = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            thread.join()


def build_synthetic_catalog() -> FeatureCatalog:
    units = {}
    for i in range(18):
        uid = 1001 + i
        units[uid] = UnitEntry(
            unit_id=uid,
            name=f"unit-{uid}",
            search_area_width=100 + i * 45,
            atk_type=1 + (i % 2),
            talent_id=1 + (i % 5),
            role_id=1 + (i % 8),
            npc_enhance_id=0,
            has_6star=True,
            has_ue1=True,
            has_ue2=(i % 2 == 0),
            has_labyrinth_replace=(i % 3 == 0),
            panel_hp=12000.0 + i * 100.0,
            panel_atk=1500.0 + i * 50.0,
            panel_magic_str=1200.0 + i * 40.0,
            panel_def=300.0 + i * 10.0,
            panel_magic_def=250.0 + i * 10.0,
            mech_flags={"heal_or_regen": i % 4 == 0, "barrier_shield": i % 5 == 0},
        )
    enemies = {
        2001: EnemyEntry(2001, 3001, "slime", 80, 30000, 30000, 500, 200, 100, 100, 10, 100, 0),
        2002: EnemyEntry(2002, 3002, "robot", 120, 45000, 45000, 700, 300, 150, 150, 10, 100, 0),
        2003: EnemyEntry(2003, 3003, "dual", 140, 55000, 55000, 800, 400, 200, 200, 10, 100, 0),
        2004: EnemyEntry(2004, 3004, "other", 150, 60000, 60000, 850, 450, 220, 220, 10, 100, 0),
        2005: EnemyEntry(2005, 3005, "boss3", 200, 80000, 80000, 900, 500, 250, 250, 10, 100, 0, is_boss_unit=True),
        2006: EnemyEntry(2006, 3006, "boss5", 250, 100000, 100000, 999, 600, 300, 300, 10, 100, 0, is_boss_unit=True),
    }
    quests = {
        770110101: QuestEntry(770110101, 1, 1, 1, 90, 1, [2001], 30000, "w-norm", "normal"),
        770120101: QuestEntry(770120101, 2, 2, 2, 90, 2, [2002], 45000, "w-rob", "robot_ex"),
        770120201: QuestEntry(770120201, 3, 2, 3, 90, 2, [2003], 55000, "w-dual", "special_dual_ex"),
        770120301: QuestEntry(770120301, 4, 2, 4, 90, 2, [2004], 60000, "w-oth", "other_ex"),
        770330101: QuestEntry(770330101, 5, 3, 3, 90, 3, [2005], 80000, "w-b3", "boss_3f_d5", True),
        770530101: QuestEntry(770530101, 6, 3, 5, 90, 3, [2006], 100000, "w-b5", "boss_5f_d5", True),
    }
    relics = {
        50001: RelicEntry(50001, "guild1", 3, 1, 1, 0, owned_unit_scaling=True),
        50002: RelicEntry(50002, "guild2", 3, 2, 1, 0, is_revive=True),
        50003: RelicEntry(50003, "guild3", 3, 3, 1, 0, gold_scaling=True),
        31001: RelicEntry(31001, "mark1", 1, 1, 4, 100),
        31002: RelicEntry(31002, "mark2", 1, 2, 4, 100),
        31003: RelicEntry(31003, "mark3", 1, 3, 4, 100),
        31004: RelicEntry(31004, "mark4", 1, 4, 4, 100),
        31005: RelicEntry(31005, "mark5", 1, 5, 4, 100),
    }
    guilds = {
        1: GuildEntry(1, "G1", 1001, 50001),
        2: GuildEntry(2, "G2", 1002, 50002),
        3: GuildEntry(3, "G3", 1003, 50003),
    }
    cat = FeatureCatalog(
        schema_version=1,
        db_version="202611131613",
        db_sha256="db-mini",
        catalog_sha256="",
        max_area=5,
        max_difficulty=5,
        units=units,
        enemies=enemies,
        quests=quests,
        relics=relics,
        guilds=guilds,
        score_settings=ScoreSettings(),
        enhance_node_ids=[11],
        environment_enemy_ids=[],
        ex_disambiguation_index={},
    )
    cat.finalize_hash()
    return cat


def make_synthetic_samples(catalog: FeatureCatalog) -> list[BattleSample]:
    unit_ids = catalog.sorted_unit_ids()
    quest_list = sorted(catalog.quests.values(), key=lambda q: q.quest_id)
    samples: list[BattleSample] = []
    # Create 24 distinct contexts x 2 legal teams = 48 rows + 1 legacy blank-context Pilot + 1 dup phys state.
    for ctx_i in range(24):
        quest = quest_list[ctx_i % len(quest_list)]
        guild_id = 1 + (ctx_i % 3)
        owned = unit_ids[:10] if ctx_i % 2 == 0 else unit_ids[2:12]
        if catalog.guilds[guild_id].guild_master_id not in owned:
            owned = sorted({catalog.guilds[guild_id].guild_master_id, *owned})
        relics = [50000 + guild_id, 31001]
        gold = 2000 + ctx_i * 350
        for t_idx in range(2):
            team = sorted(owned[t_idx : t_idx + 5])
            is_legacy_blank = ctx_i == 0 and t_idx == 0
            round_id = "round0_pilot" if is_legacy_blank else "round1_scale"
            sid = f"s-{ctx_i:02d}-{t_idx}"
            base = BattleSample(
                sample_id=sid,
                round_id=round_id,
                source="sim",
                db_version=catalog.db_version,
                quest_id=quest.quest_id,
                canonical_wave_sig=quest.canonical_wave_sig,
                bucket=quest.bucket,
                area=quest.area,
                difficulty=5,
                guild_id=guild_id,
                team_unit_ids=team,
                owned_unit_ids=owned,
                owned_unit_count=len(owned),
                owned_role_counts=[2] * 8,
                owned_talent_counts=[2] * 5,
                relic_ids=relics,
                mark_stacks=catalog.mark_stacks_for_relics(relics),
                alpha_rupee=gold,
                revive_available=(guild_id == 2),
                outcome_counts=[2, 0, 0] if (ctx_i + t_idx) % 3 != 0 else [0, 1, 1],
                mean_damage_ratio=1.0 if (ctx_i + t_idx) % 3 != 0 else 0.78,
                mean_win_rem_time=0.4 if (ctx_i + t_idx) % 3 != 0 else None,
                context_id="",
            )
            if not is_legacy_blank:
                base = replace(base, context_id=semantic_context_sha256(base))
            samples.append(base)
    # Add a duplicate physical state with a different round_id / sample_id.
    dup = replace(samples[1], sample_id="s-dup-round0", round_id="round0_pilot")
    samples.append(dup)
    return samples


def save_dummy_checkpoints(root: Path, catalog: FeatureCatalog, count: int = 3) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    cat_sha = catalog.catalog_sha256 or catalog.compute_sha256()
    cfg = CriticConfig(
        num_units=len(catalog.units) + 1,
        num_enemies=len(catalog.enemies) + 1,
        num_quests=len(catalog.quests) + 1,
        num_relics=len(catalog.relics) + 1,
        ablation_mode="m2b_r1",
        d_model=32,
        n_heads=4,
        n_context_layers=1,
        n_team_layers=1,
        relic_max_len=16,
        roster_max_len=24,
        batch_size=32,
        epochs=2,
    )
    paths: list[Path] = []
    for seed in range(42, 42 + count):
        torch.manual_seed(seed)
        model = TwoTowerBattleCritic(cfg)
        ckpt = {
            "config": cfg.to_dict(),
            "catalog_sha256": cat_sha,
            "seed": seed,
            "model": model.state_dict(),
        }
        p = root / f"ckpt_{seed}.pt"
        torch.save(ckpt, p)
        paths.append(p)
    return paths


class S3ActiveMiningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if HAS_TORCH:
            cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
            torch.set_num_threads(1)

    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / f"labyrinth-s3-test-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.catalog = build_synthetic_catalog()
        self.catalog_path = self.root / "feature_pack.json"
        self.catalog.save(self.catalog_path)
        self.profile = {
            "difficulty": 5,
            "timeout_sec": 5.0,
            "max_retries": 0,
            "retry_on_timeout": False,
            "use_fixed_random_seed": True,
            "adaptive_repeat": {"initial_runs": 2, "borderline_runs": 6, "high_rng_keywords": ["robot"]},
        }
        self.profile_path = self.root / "profile.json"
        self.profile_path.write_text(json.dumps(self.profile), encoding="utf-8")
        self.sampling = {
            "difficulty": 5,
            "quest_type_weights": {
                "normal": 0.3,
                "robot_ex": 0.25,
                "special_dual_ex": 0.11,
                "other_ex": 0.12,
                "boss_3f_d5": 0.10,
                "boss_5f_d5": 0.12,
            },
            "team_mix_weights": {"prior": 0.35, "counter_stratified": 0.25, "perturb": 0.25, "random": 0.15},
            "guild_weights": {"1": 0.4, "2": 0.3, "3": 0.3},
            "area_state_distributions": {
                "1": {"owned_min": 8, "owned_max": 12, "relic_min": 1, "relic_max": 4, "gold_min": 500, "gold_max": 4000},
                "5": {"owned_min": 10, "owned_max": 14, "relic_min": 2, "relic_max": 5, "gold_min": 1000, "gold_max": 8000},
            },
            "boundary_probe_probabilities": {
                "mark_threshold_focus": 0.2,
                "owned_25_boundary": 0.0,
                "gold_10k_boundary": 0.1,
                "empty_relic_early": 0.0,
            },
        }
        self.samples = make_synthetic_samples(self.catalog)
        self.s2_reg, self.s2_parts = build_s2_holdout_registry(
            self.samples, catalog_sha256=self.catalog.catalog_sha256, split_seed=42, group_by="context"
        )
        self.s2_reg_path = self.root / "s2_holdout_folds.json"
        self.s2_reg.save(self.s2_reg_path)
        s5_out = self.root / "s5_res"
        s5_res = build_s5_reservation(
            self.catalog,
            self.profile,
            self.sampling,
            denied_semantic_contexts={semantic_context_sha256(s) for s in self.samples},
            denied_physical_states={s.physical_state_key() for s in self.samples},
            spec=S5ReservationSpec(
                stratified_bucket_counts={
                    "normal": 4,
                    "robot_ex": 4,
                    "special_dual_ex": 4,
                    "other_ex": 4,
                    "boss_3f_d5": 4,
                    "boss_5f_d5": 4,
                },
                stratified_paired_teams=4,
                stratified_repeats=5,
                paired_bucket_context_counts={"normal": 2, "robot_ex": 2},
                paired_teams_per_context=6,
                paired_repeats=11,
                seed=9911,
                num_workers=5,
            ),
            out_dir=s5_out,
        )
        self.s5_denylist_path = Path(s5_res["denylist_path"])
        self.s5_denylist = S5Denylist.load(self.s5_denylist_path)

    @cached_property
    def ckpts(self) -> list[Path]:
        return save_dummy_checkpoints(self.root / "ckpts", self.catalog, count=3)

    def _plan_small(self, out_dir: Path, samples: list[BattleSample] | None = None) -> dict:
        return plan_s3_mining(
            catalog=self.catalog,
            profile=self.profile,
            sampling_config=self.sampling,
            samples=samples if samples is not None else self.samples,
            s2_registry=self.s2_reg,
            s5_denylist=self.s5_denylist,
            checkpoint_paths=self.ckpts,
            out_dir=out_dir,
            spec=S3PlannerSpec(
                num_workers=5,
                per_worker_rows=5,
                bucket_quotas_per_worker={"normal": 2, "robot_ex": 1, "boss": 1, "broad": 1},
                repeat_tier_counts_per_worker={1: 2, 6: 2, 12: 1},
                reserve_rows_per_worker=2,
                reserve_repeats=6,
                overconfident_win_prob_min=0.45,
                disagreement_std_min=0.01,
                seed=777,
                batch_size=32,
            ),
        )

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_1_identity_determinism_disjointness_and_mutations(self):
        dir_a = self.root / "plan_a"
        dir_b = self.root / "plan_b"
        reversed_samples = list(reversed(self.samples))
        res_a = self._plan_small(dir_a, self.samples)
        res_b = self._plan_small(dir_b, reversed_samples)

        self.assertEqual(res_a["sidecar_sha256"], res_b["sidecar_sha256"])
        for i in range(5):
            wid = worker_id_for_index(i)
            self.assertEqual(
                sha256_file(dir_a / f"s3_plan_{wid}.jsonl"),
                sha256_file(dir_b / f"s3_plan_{wid}.jsonl"),
            )
            self.assertEqual(
                sha256_file(dir_a / f"s3_reserve_{wid}.jsonl"),
                sha256_file(dir_b / f"s3_reserve_{wid}.jsonl"),
            )

        sidecar = res_a["sidecar"]
        worker_ctx_sets = []
        legacy_blank_ctx = semantic_context_sha256(self.samples[0])
        for wid, meta in sidecar["workers"].items():
            rows = [json.loads(line) for line in (dir_a / meta["plan_file"]).read_text(encoding="utf-8").splitlines()]
            worker_ctx_sets.append({r["semantic_context_sha256"] for r in rows})
            for r in rows:
                self.assertEqual(worker_id_for_index(worker_for_context(r["semantic_context_sha256"], 5)), wid)
                # Ensure legacy blank-context Pilot row context is blocked if it fell in holdout.
                if legacy_blank_ctx in self.s2_reg.holdout_semantic_contexts:
                    self.assertNotEqual(r["semantic_context_sha256"], legacy_blank_ctx)
                # Verify any anchor mutation stays inside parent roster and mutates 1 or 2 slots.
                if r["anchor_sample_id"] is not None:
                    parent = next(s for s in self.samples if s.sample_id == r["anchor_sample_id"])
                    cand_team = set(r["payload"]["team_unit_ids"])
                    parent_team = set(parent.team_unit_ids)
                    self.assertTrue(cand_team <= set(parent.owned_unit_ids))
                    self.assertIn(len(cand_team - parent_team), (1, 2))
        for i in range(len(worker_ctx_sets)):
            for j in range(i + 1, len(worker_ctx_sets)):
                self.assertEqual(worker_ctx_sets[i] & worker_ctx_sets[j], set())

    def test_2_overconfident_loss_anchor(self):
        base = self.samples[2]
        high_p_loss = replace(base, sample_id="high-p-loss", outcome_counts=[0, 4, 0])
        low_p_win = replace(base, sample_id="low-p-win", outcome_counts=[4, 0, 0])
        self.assertTrue(
            is_overconfident_loss_anchor(
                high_p_loss, {"p_win_mean": 0.75}, overconfident_win_prob_min=0.70, observed_nonwin_max_rate=0.50
            )
        )
        # |1.0 - 0.05| = 0.95 > |0.0 - 0.75| = 0.75, yet low-p win MUST be excluded.
        self.assertFalse(
            is_overconfident_loss_anchor(
                low_p_win, {"p_win_mean": 0.05}, overconfident_win_prob_min=0.70, observed_nonwin_max_rate=0.50
            )
        )

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_2_nonfinite_score_rejection(self):
        base = self.samples[2]
        ensemble, _ = load_calibrated_ensemble(self.ckpts, self.catalog)
        with patch.object(
            TwoTowerEnsemble,
            "forward",
            return_value={
                "p_win_mean": torch.tensor([float("nan")]),
                "p_win_std": torch.tensor([0.0]),
                "lcb_win": torch.tensor([0.5]),
                "damage_mean": torch.tensor([0.5]),
                "win_rem_time": torch.tensor([0.5]),
            },
        ):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                score_samples_with_ensemble(ensemble, self.catalog, [base], batch_size=1)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_3_exact_repeats_1_6_12_and_seed_validation(self):
        plan_dir = self.root / "plan_exec"
        res = self._plan_small(plan_dir)
        sidecar_path = Path(res["sidecar_path"])
        wid = "w01"
        plan_path = plan_dir / f"s3_plan_{wid}.jsonl"
        # Pick 1 row per repeat tier (1, 6, 12) -> 3 rows, 19 subruns total.
        all_rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
        one_per_tier = [
            next(r for r in all_rows if r["repeat_count"] == 1),
            next(r for r in all_rows if r["repeat_count"] == 6),
            next(r for r in all_rows if r["repeat_count"] == 12),
        ]
        sub_plan = plan_dir / "tiny_3_tiers.jsonl"
        sub_sha = write_jsonl_atomic(sub_plan, one_per_tier)
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["workers"][wid]["plan_file"] = sub_plan.name
        sidecar["workers"][wid]["plan_sha256"] = sub_sha
        sidecar["workers"][wid]["row_count"] = 3
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        out_path = self.root / "exec_ok" / "w01.jsonl"
        with mock_sim_server(self.catalog) as (url, requests):
            client = SimClient(url, self.catalog, self.profile)
            summary = run_worker_plan(
                sidecar_path=sidecar_path,
                worker_plan_path=sub_plan,
                worker_id=wid,
                catalog=self.catalog,
                profile=self.profile,
                client=client,
                out_path=out_path,
                s2_registry=self.s2_reg,
                s5_denylist=self.s5_denylist,
                existing_physical_states={s.physical_state_key() for s in self.samples},
                chunk_size=10,
                max_subruns_per_post=64,
            )
        self.assertEqual(summary["total_battles"], 3)
        self.assertEqual(summary["submitted_subruns"], 19)
        self.assertEqual(summary["accepted_subruns"], 19)
        sent_seeds = [q["Settings"]["RandomSeed"] for req in requests for q in req]
        self.assertEqual(len(sent_seeds), 19)
        self.assertEqual(len(set(sent_seeds)), 19)
        rows = read_samples(out_path)
        self.assertEqual(sorted(len(s.runs) for s in rows), [1, 6, 12])

        # Verify restart is idempotent without extra requests.
        with mock_sim_server(self.catalog) as (url, requests2):
            client2 = SimClient(url, self.catalog, self.profile)
            resumed = run_worker_plan(
                sidecar_path=sidecar_path,
                worker_plan_path=sub_plan,
                worker_id=wid,
                catalog=self.catalog,
                profile=self.profile,
                client=client2,
                out_path=out_path,
                s2_registry=self.s2_reg,
                s5_denylist=self.s5_denylist,
                existing_physical_states={s.physical_state_key() for s in self.samples},
                resume=True,
            )
        self.assertEqual(len(requests2), 0)
        self.assertEqual(resumed["existing_battles"], 3)
        self.assertEqual(resumed["new_battles"], 0)

        # Seed echo mismatch & missing run & non-finite value must quarantine with zero accepted rows.
        def bad_seed_respond(query, row, batch_no, idx):
            return row | {"CurrentSeed": int(row["CurrentSeed"]) + 1}

        bad_out = self.root / "exec_bad_seed" / "w01.jsonl"
        with mock_sim_server(self.catalog, respond=bad_seed_respond) as (url, _):
            client_bad = SimClient(url, self.catalog, self.profile)
            with self.assertRaisesRegex(RuntimeError, "plan exhausted"):
                run_worker_plan(
                    sidecar_path=sidecar_path,
                    worker_plan_path=sub_plan,
                    worker_id=wid,
                    catalog=self.catalog,
                    profile=self.profile,
                    client=client_bad,
                    out_path=bad_out,
                    s2_registry=self.s2_reg,
                    s5_denylist=self.s5_denylist,
                    existing_physical_states={s.physical_state_key() for s in self.samples},
                )
        self.assertFalse(bad_out.exists())
        self.assertEqual(len([l for l in bad_out.with_name("w01.failed.jsonl").read_text(encoding="utf-8").splitlines() if l]), 3)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_4_crash_timeout_sync_and_reserve_accounting(self):
        plan_dir = self.root / "plan_fault"
        res = self._plan_small(plan_dir)
        sidecar_path = Path(res["sidecar_path"])
        wid = "w01"
        plan_path = plan_dir / f"s3_plan_{wid}.jsonl"
        reserve_path = plan_dir / f"s3_reserve_{wid}.jsonl"
        sync_dir = self.root / "sync_fault"
        sync_dir.mkdir()
        existing_phys = {s.physical_state_key() for s in self.samples}

        # (a) Crash AFTER prepared journal BEFORE POST -> reconciled_indeterminate on resume, never replayed!
        out_a = self.root / "fault_a" / "w01.jsonl"

        def crash_before_post(_prep):
            raise RuntimeError("injected crash after prepare")

        with mock_sim_server(self.catalog) as (url, reqs_a):
            client = SimClient(url, self.catalog, self.profile)
            with self.assertRaisesRegex(RuntimeError, "injected crash after prepare"):
                run_worker_plan(
                    sidecar_path=sidecar_path,
                    worker_plan_path=plan_path,
                    worker_id=wid,
                    catalog=self.catalog,
                    profile=self.profile,
                    client=client,
                    out_path=out_a,
                    s2_registry=self.s2_reg,
                    s5_denylist=self.s5_denylist,
                    existing_physical_states=existing_phys,
                    target_battles=2,
                    max_subruns_per_post=12,
                    before_post_hook=crash_before_post,
                )
            self.assertEqual(len(reqs_a), 0)
            # Resume with reserve authorized fills target=5 without replaying the indeterminate row!
            summary_a = run_worker_plan(
                sidecar_path=sidecar_path,
                worker_plan_path=plan_path,
                worker_id=wid,
                catalog=self.catalog,
                profile=self.profile,
                client=client,
                out_path=out_a,
                reserve_plan_path=reserve_path,
                allow_reserve=True,
                s2_registry=self.s2_reg,
                s5_denylist=self.s5_denylist,
                existing_physical_states=existing_phys,
                target_battles=5,
                max_subruns_per_post=12,
                resume=True,
                sync_dir=sync_dir,
            )
        self.assertGreater(summary_a["indeterminate_battles"], 0)
        self.assertGreater(summary_a["reserve_used_battles"], 0)
        self.assertGreater(summary_a["submitted_subruns"], summary_a["accepted_subruns"])
        self.assertEqual(summary_a["total_battles"], 5)

        # (b) Timeout after 1 committed shard: flushes prior commits, never bisects/retries, not replayed on resume.
        out_b = self.root / "fault_b" / "w01.jsonl"
        sync_b = self.root / "sync_b"
        sync_b.mkdir()
        call_counter = 0

        def timeout_second_post(_configs):
            nonlocal call_counter
            call_counter += 1
            if call_counter == 1:
                # Use normal mock server for first batch.
                return orig_sim(_configs)
            raise SimulationTimeoutError("post timed out")

        with mock_sim_server(self.catalog) as (url, _):
            client_b = SimClient(url, self.catalog, self.profile)
            orig_sim = client_b.simulate_batch_single_runs
            with patch.object(client_b, "simulate_batch_single_runs", side_effect=timeout_second_post):
                with self.assertRaises(SimulationTimeoutError):
                    run_worker_plan(
                        sidecar_path=sidecar_path,
                        worker_plan_path=plan_path,
                        worker_id=wid,
                        catalog=self.catalog,
                        profile=self.profile,
                        client=client_b,
                        out_path=out_b,
                        s2_registry=self.s2_reg,
                        s5_denylist=self.s5_denylist,
                        existing_physical_states=existing_phys,
                        target_battles=5,
                        chunk_size=1,
                        max_subruns_per_post=12,
                        sync_dir=sync_b,
                    )
        self.assertEqual(call_counter, 2)
        self.assertTrue((sync_b / "w01.manifest.json").is_file())
        journal_events = [json.loads(l)["event"] for l in out_b.with_name("w01.journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertIn("timeout_indeterminate", journal_events)

        # (c) Crash AFTER verified shard/manifest BEFORE journal commit -> reconciled_committed once on resume!
        out_c = self.root / "fault_c" / "w01.jsonl"

        def crash_after_shard(_shard, _entry):
            raise RuntimeError("crash before journal commit")

        with mock_sim_server(self.catalog) as (url, reqs_c):
            client_c = SimClient(url, self.catalog, self.profile)
            with self.assertRaisesRegex(RuntimeError, "crash before journal commit"):
                run_worker_plan(
                    sidecar_path=sidecar_path,
                    worker_plan_path=plan_path,
                    worker_id=wid,
                    catalog=self.catalog,
                    profile=self.profile,
                    client=client_c,
                    out_path=out_c,
                    s2_registry=self.s2_reg,
                    s5_denylist=self.s5_denylist,
                    existing_physical_states=existing_phys,
                    target_battles=2,
                    chunk_size=1,
                    max_subruns_per_post=1,
                    after_shard_hook=crash_after_shard,
                )
            reqs_before = len(reqs_c)
            res_c = run_worker_plan(
                sidecar_path=sidecar_path,
                worker_plan_path=plan_path,
                worker_id=wid,
                catalog=self.catalog,
                profile=self.profile,
                client=client_c,
                out_path=out_c,
                s2_registry=self.s2_reg,
                s5_denylist=self.s5_denylist,
                existing_physical_states=existing_phys,
                target_battles=1,
                resume=True,
            )
            self.assertEqual(len(reqs_c), reqs_before)
            self.assertEqual(res_c["existing_battles"], 1)
            j_events_c = [json.loads(l)["event"] for l in out_c.with_name("w01.journal.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("reconciled_committed", j_events_c)

        # (d) Manifest sync failure produces no false success in sync_dir or journal.
        out_d = self.root / "fault_d" / "w01.jsonl"
        sync_d = self.root / "sync_d"
        sync_d.mkdir()

        def failing_sync(src, dst):
            if src.name.endswith(".manifest.json"):
                raise OSError(errno.EIO, "sync I/O failure")
            return _sync_file_atomic(src, dst)

        with mock_sim_server(self.catalog) as (url, _):
            client_d = SimClient(url, self.catalog, self.profile)
            with patch("labyrinth_nn.active_mining.s3_executor._sync_file_atomic", side_effect=failing_sync):
                with self.assertRaisesRegex(OSError, "sync I/O failure"):
                    run_worker_plan(
                        sidecar_path=sidecar_path,
                        worker_plan_path=plan_path,
                        worker_id=wid,
                        catalog=self.catalog,
                        profile=self.profile,
                        client=client_d,
                        out_path=out_d,
                        s2_registry=self.s2_reg,
                        s5_denylist=self.s5_denylist,
                        existing_physical_states=existing_phys,
                        target_battles=1,
                        chunk_size=1,
                        sync_dir=sync_d,
                    )
        self.assertFalse((sync_d / "w01.manifest.json").exists())
        j_events_d = [json.loads(l)["event"] for l in out_d.with_name("w01.journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertNotIn("committed", j_events_d)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_5_worker_plan_and_catalog_tamper_rejection(self):
        plan_dir = self.root / "plan_tamper"
        res = self._plan_small(plan_dir)
        sidecar_path = Path(res["sidecar_path"])
        wid = "w01"
        plan_path = plan_dir / f"s3_plan_{wid}.jsonl"
        existing_phys = {s.physical_state_key() for s in self.samples}

        with mock_sim_server(self.catalog) as (url, requests):
            client = SimClient(url, self.catalog, self.profile)

            # 1) Tamper plan row -> sha256 mismatch before any POST.
            tampered_plan = self.root / "tampered_w01.jsonl"
            rows = [json.loads(l) for l in plan_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["worker_id"] = "w02"
            write_jsonl_atomic(tampered_plan, rows)
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                run_worker_plan(
                    sidecar_path=sidecar_path,
                    worker_plan_path=tampered_plan,
                    worker_id=wid,
                    catalog=self.catalog,
                    profile=self.profile,
                    client=client,
                    out_path=self.root / "t1.jsonl",
                    s2_registry=self.s2_reg,
                    s5_denylist=self.s5_denylist,
                    existing_physical_states=existing_phys,
                )

            # 2) Wrong catalog -> fails closed before POST.
            wrong_cat = replace(self.catalog, db_version="999999999999")
            with self.assertRaisesRegex(ValueError, "db_version mismatch"):
                run_worker_plan(
                    sidecar_path=sidecar_path,
                    worker_plan_path=plan_path,
                    worker_id=wid,
                    catalog=wrong_cat,
                    profile=self.profile,
                    client=client,
                    out_path=self.root / "t3.jsonl",
                    s2_registry=self.s2_reg,
                    s5_denylist=self.s5_denylist,
                    existing_physical_states=existing_phys,
                )
            self.assertEqual(len(requests), 0)

    def test_5_s5_denylist_tamper_rejection(self):
        bad_deny = self.root / "bad_s5_denylist.json"
        raw_deny = json.loads(self.s5_denylist_path.read_text(encoding="utf-8"))
        raw_deny["seed"] += 1
        bad_deny.write_text(json.dumps(raw_deny), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "denylist_sha256 mismatch"):
            S5Denylist.load(bad_deny)

    def test_5_s5_quest_only_baseline_leaves_out_the_entire_context(self):
        first = replace(self.samples[0], outcome_counts=[2, 0, 0])
        teammate = replace(self.samples[1], outcome_counts=[1, 1, 0])
        other_context = replace(self.samples[12], outcome_counts=[0, 2, 0])
        other_quest = replace(first, quest_id=-1, outcome_counts=[2, 0, 0])
        isolated = replace(first, quest_id=-2, bucket="unseen")
        samples = [first, teammate, other_context, other_quest, isolated]
        records = [{"sample_id": s.sample_id} for s in samples]

        baseline = quest_only_baseline_records(samples, records)

        self.assertEqual([r["pred_probs"][0] for r in baseline], [0.0, 0.0, 0.75, 0.5, 0.5])

    def test_5_s4_frozen_splits(self):
        # Use fresh labeled S3 contexts beyond the gold range of the S2 and S5 fixtures.
        s3_samples = []
        for index, base in enumerate(self.samples[:3]):
            sample = replace(
                base, sample_id=f"new-s3-{index}", round_id="round2_s3", alpha_rupee=12000 + index,
                outcome_counts=[1, 0, 0], mean_damage_ratio=1.0, mean_win_rem_time=0.5,
            )
            s3_samples.append(replace(sample, context_id=semantic_context_sha256(sample)))
        frozen_parts = partition_frozen_samples(
            self.samples + s3_samples, self.s2_reg, self.s5_denylist, catalog=self.catalog
        )
        for fold in ("val_select", "val_calib", "test"):
            self.assertEqual(
                [s.sample_id for s in frozen_parts[fold]],
                [s.sample_id for s in self.s2_parts[fold]],
            )
        self.assertEqual(len(frozen_parts["train"]), len(self.s2_parts["train"]) + len(s3_samples))

        # S4 must reject any sample colliding with S2 holdout or S5 denylist.
        leaked_s5 = replace(s3_samples[0], sample_id="leak-s5", quest_id=770110101)
        with self.assertRaisesRegex(ValueError, "S5 reserved denylist"):
            with patch.object(S5Denylist, "is_denied", return_value=True):
                partition_frozen_samples(self.samples + [leaked_s5], self.s2_reg, self.s5_denylist)

        holdout_sample = self.s2_parts["val_select"][0]
        leaked_s2 = replace(holdout_sample, sample_id="s3-collides-val-select", round_id="round2_s3")
        with self.assertRaisesRegex(ValueError, "collides with frozen S2 holdout"):
            partition_frozen_samples(self.samples + [leaked_s2], self.s2_reg, self.s5_denylist)

    def test_5_s4_distillation_gate(self):
        gate_yes = evaluate_s4_distillation_gate(
            single_val_select_metrics=[{"global": {"brier_win": 0.050, "context_top1_pos_fpr": 0.045}}],
            ensemble_val_select_metrics={"global": {"brier_win": 0.045, "context_top1_pos_fpr": 0.040}},
        )
        self.assertTrue(gate_yes["distill_triggered"])
        gate_no = evaluate_s4_distillation_gate(
            single_val_select_metrics=[
                {"global": {"brier_win": 0.040, "context_top1_pos_fpr": 0.045}},
                {"global": {"brier_win": 0.040, "context_top1_pos_fpr": 0.0}},
            ],
            ensemble_val_select_metrics={"global": {"brier_win": 0.039, "context_top1_pos_fpr": 0.025}},
        )
        self.assertFalse(gate_no["distill_triggered"])
        self.assertEqual(gate_no["best_single_global"]["context_top1_pos_fpr"], 0.0)

    def _extended_s5_fixture(self):
        bad_units = {136001, 136201, 137401}
        ext_units = dict(self.catalog.units)
        for bad_uid in sorted(bad_units):
            ext_units[bad_uid] = replace(
                self.catalog.units[1001],
                unit_id=bad_uid,
                name=f"unsupported-{bad_uid}",
            )
        ext_catalog = replace(self.catalog, units=ext_units, catalog_sha256="")
        ext_catalog.finalize_hash()
        ext_samples = [
            replace(
                s,
                owned_unit_ids=sorted([*s.owned_unit_ids, 136001]),
                owned_unit_count=len(s.owned_unit_ids) + 1,
            )
            if i == 2
            else s
            for i, s in enumerate(self.samples)
        ]
        s5_ext_out = self.root / "s5_ext"
        s5_ext = build_s5_reservation(
            ext_catalog,
            self.profile,
            self.sampling,
            denied_semantic_contexts={semantic_context_sha256(s) for s in ext_samples},
            denied_physical_states={s.physical_state_key() for s in ext_samples},
            spec=S5ReservationSpec(
                stratified_bucket_counts={
                    "normal": 4,
                    "robot_ex": 4,
                    "special_dual_ex": 4,
                    "other_ex": 4,
                    "boss_3f_d5": 4,
                    "boss_5f_d5": 4,
                },
                stratified_paired_teams=4,
                stratified_repeats=5,
                paired_bucket_context_counts={"normal": 2, "robot_ex": 2},
                paired_teams_per_context=6,
                paired_repeats=11,
                seed=9912,
                num_workers=5,
            ),
            out_dir=s5_ext_out,
        )
        return bad_units, ext_catalog, ext_samples, s5_ext_out, S5Denylist.load(s5_ext["denylist_path"])

    def test_6_mutations_and_s5_reservation_filter_unsupported_team_units(self):
        bad_units, ext_catalog, _, s5_ext_out, _ = self._extended_s5_fixture()
        base_parent = self.samples[2]
        parent_with_bad = replace(
            base_parent,
            owned_unit_ids=sorted([*base_parent.owned_unit_ids, *bad_units]),
            owned_unit_count=len(base_parent.owned_unit_ids) + len(bad_units),
        )
        muts = generate_one_two_slot_mutations(ext_catalog, parent_with_bad, limit=64)
        self.assertTrue(muts)
        for team, _ in muts:
            self.assertFalse(bad_units & set(team))

        s5_rows = [
            json.loads(line)
            for w in range(5)
            for line in (s5_ext_out / f"s5_plan_{worker_id_for_index(w)}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(s5_rows), 48)
        self.assertTrue(all(not (bad_units & set(r["payload"]["team_unit_ids"])) for r in s5_rows))
        self.assertTrue(any(bad_units & set(r["payload"]["owned_unit_ids"]) for r in s5_rows))

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_6_unsupported_team_unit_filter_and_per_battle_isolation(self):
        bad_units, ext_catalog, ext_samples, _, s5_ext_deny = self._extended_s5_fixture()
        ext_ckpts = save_dummy_checkpoints(self.root / "ext_ckpts", ext_catalog, count=3)
        s2_ext_reg, _ = build_s2_holdout_registry(
            ext_samples, catalog_sha256=ext_catalog.catalog_sha256, split_seed=42, group_by="context"
        )
        # S3 plan with extended catalog filters bad units and verify_s3_plan rejects tampered excluded unit.
        s3_ext_out = self.root / "s3_ext_plan"
        s3_ext = plan_s3_mining(
            catalog=ext_catalog,
            profile=self.profile,
            sampling_config=self.sampling,
            samples=ext_samples,
            s2_registry=s2_ext_reg,
            s5_denylist=s5_ext_deny,
            checkpoint_paths=ext_ckpts,
            out_dir=s3_ext_out,
            spec=S3PlannerSpec(
                num_workers=5,
                per_worker_rows=5,
                bucket_quotas_per_worker={"normal": 2, "robot_ex": 1, "boss": 1, "broad": 1},
                repeat_tier_counts_per_worker={1: 2, 6: 2, 12: 1},
                reserve_rows_per_worker=2,
                reserve_repeats=6,
                overconfident_win_prob_min=0.45,
                disagreement_std_min=0.01,
                seed=888,
                batch_size=32,
            ),
        )
        w01_rows = [json.loads(l) for l in (s3_ext_out / "s3_plan_w01.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(not (bad_units & set(r["payload"]["team_unit_ids"])) for r in w01_rows))
        bad_row = dict(w01_rows[0], payload=dict(w01_rows[0]["payload"], team_unit_ids=[137401, *w01_rows[0]["payload"]["team_unit_ids"][1:]]))
        with self.assertRaisesRegex(ValueError, "excluded simulator unit"):
            verify_s3_plan(
                sidecar=s3_ext["sidecar"],
                worker_plans={"w01": [bad_row]},
                catalog=ext_catalog,
            )

        # 2) Per-battle isolation in s3_executor: 3-battle batch where battle #2 returns Success=False.
        plan_dir = self.root / "plan_iso"
        res = self._plan_small(plan_dir)
        sidecar_path = Path(res["sidecar_path"])
        wid = "w01"
        plan_path = plan_dir / f"s3_plan_{wid}.jsonl"
        all_rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
        three_rows = [
            dict(all_rows[0], repeat_count=1, run_seeds=[101]),
            dict(all_rows[1], repeat_count=2, run_seeds=[201, 202]),
            dict(all_rows[2], repeat_count=1, run_seeds=[301]),
        ]
        # Recompute plan_index and sidecar for exact 3-row plan with unique seeds 101,201,202,301.
        sub_plan = plan_dir / "iso_3_rows.jsonl"
        sub_sha = write_jsonl_atomic(sub_plan, three_rows)
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["workers"][wid]["plan_file"] = sub_plan.name
        sidecar["workers"][wid]["plan_sha256"] = sub_sha
        sidecar["workers"][wid]["row_count"] = 3
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        def battle2_fails_respond(query, row, batch_no, idx):
            if int(query["Settings"]["RandomSeed"]) == 202:
                return {
                    "Success": False,
                    "Message": "StartByServer 失败: The given key '137401' was not present in the dictionary.",
                }
            return row

        out_iso = self.root / "exec_iso" / "w01.jsonl"
        sync_iso = self.root / "sync_iso"
        sync_iso.mkdir()
        existing_phys = {s.physical_state_key() for s in self.samples}
        with mock_sim_server(self.catalog, respond=battle2_fails_respond) as (url, reqs_iso):
            client_iso = SimClient(url, self.catalog, self.profile)
            with self.assertRaisesRegex(RuntimeError, "plan exhausted"):
                run_worker_plan(
                    sidecar_path=sidecar_path,
                    worker_plan_path=sub_plan,
                    worker_id=wid,
                    catalog=self.catalog,
                    profile=self.profile,
                    client=client_iso,
                    out_path=out_iso,
                    s2_registry=self.s2_reg,
                    s5_denylist=self.s5_denylist,
                    existing_physical_states=existing_phys,
                    target_battles=3,
                    chunk_size=10,
                    max_subruns_per_post=64,
                    sync_dir=sync_iso,
                )
        # Single HTTP POST (no retry, no bisection); battles #1 & #3 commit, only battle #2 quarantined.
        self.assertEqual(len(reqs_iso), 1)
        manifest_iso = json.loads((sync_iso / "w01.manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest_iso["total_unique_battles"], 2)
        self.assertEqual(manifest_iso["accepted_subruns"], 2)
        self.assertEqual(manifest_iso["submitted_subruns"], 4)
        committed_sids = [s.sample_id for s in read_samples(sync_iso / "w01.jsonl")]
        self.assertEqual(committed_sids, [three_rows[0]["sample_id"], three_rows[2]["sample_id"]])
        failed_rows = [json.loads(l) for l in out_iso.with_name("w01.failed.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual([r["sample_id"] for r in failed_rows], [three_rows[1]["sample_id"]])
        self.assertEqual(failed_rows[0]["reason"], "quarantined_battle_response_error")
        j_iso = [json.loads(l) for l in out_iso.with_name("w01.journal.jsonl").read_text(encoding="utf-8").splitlines()]
        quar = [r for r in j_iso if r["event"] == "quarantined"]
        comm = [r for r in j_iso if r["event"] == "committed"]
        self.assertEqual(quar[0]["sample_ids"], [three_rows[1]["sample_id"]])
        self.assertEqual(comm[0]["sample_ids"], [three_rows[0]["sample_id"], three_rows[2]["sample_id"]])

        # 3) Crash AFTER partial-batch quarantine before shard flush reconciles the 2 good battles as indeterminate.
        out_crash = self.root / "exec_iso_crash" / "w01.jsonl"

        def crash_after_iso_parse(_prep, _samples):
            raise RuntimeError("injected crash after partial quarantine")

        with mock_sim_server(self.catalog, respond=battle2_fails_respond) as (url, reqs_crash):
            client_crash = SimClient(url, self.catalog, self.profile)
            with self.assertRaisesRegex(RuntimeError, "injected crash after partial quarantine"):
                run_worker_plan(
                    sidecar_path=sidecar_path,
                    worker_plan_path=sub_plan,
                    worker_id=wid,
                    catalog=self.catalog,
                    profile=self.profile,
                    client=client_crash,
                    out_path=out_crash,
                    s2_registry=self.s2_reg,
                    s5_denylist=self.s5_denylist,
                    existing_physical_states=existing_phys,
                    target_battles=3,
                    chunk_size=10,
                    max_subruns_per_post=64,
                    after_parse_hook=crash_after_iso_parse,
                )
            reqs_before = len(reqs_crash)
            with self.assertRaisesRegex(RuntimeError, "plan exhausted"):
                run_worker_plan(
                    sidecar_path=sidecar_path,
                    worker_plan_path=sub_plan,
                    worker_id=wid,
                    catalog=self.catalog,
                    profile=self.profile,
                    client=client_crash,
                    out_path=out_crash,
                    s2_registry=self.s2_reg,
                    s5_denylist=self.s5_denylist,
                    existing_physical_states=existing_phys,
                    target_battles=3,
                    chunk_size=10,
                    max_subruns_per_post=64,
                    resume=True,
                )
            self.assertEqual(len(reqs_crash), reqs_before)
        j_crash = [json.loads(l) for l in out_crash.with_name("w01.journal.jsonl").read_text(encoding="utf-8").splitlines()]
        rec_indet = [r for r in j_crash if r["event"] == "reconciled_indeterminate"]
        self.assertEqual(len(rec_indet), 1)
        self.assertEqual(rec_indet[0]["sample_ids"], [three_rows[0]["sample_id"], three_rows[2]["sample_id"]])

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_7_disjoint_multi_seed_cohort_planning(self):
        dir1 = self.root / "cohort1"
        dir2 = self.root / "cohort2"
        spec1 = S3PlannerSpec(
            num_workers=5,
            per_worker_rows=5,
            bucket_quotas_per_worker={"normal": 2, "robot_ex": 1, "boss": 1, "broad": 1},
            repeat_tier_counts_per_worker={1: 2, 6: 2, 12: 1},
            reserve_rows_per_worker=2,
            reserve_repeats=6,
            overconfident_win_prob_min=0.45,
            disagreement_std_min=0.01,
            seed=777,
            round_id_prefix="round2_s3",
            batch_size=32,
        )
        res1 = plan_s3_mining(
            catalog=self.catalog,
            profile=self.profile,
            sampling_config=self.sampling,
            samples=self.samples,
            s2_registry=self.s2_reg,
            s5_denylist=self.s5_denylist,
            checkpoint_paths=self.ckpts,
            out_dir=dir1,
            spec=spec1,
        )
        c1_rows = [
            json.loads(line)
            for p in list(dir1.glob("s3_plan_w0*.jsonl")) + list(dir1.glob("s3_reserve_w0*.jsonl"))
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        c1_phys = {r["physical_state_key"] for r in c1_rows}
        c1_fresh_ctx = {r["semantic_context_sha256"] for r in c1_rows if r.get("anchor_sample_id") is None}

        spec2 = replace(spec1, seed=778, round_id_prefix="round2_s3_local")
        res2 = plan_s3_mining(
            catalog=self.catalog,
            profile=self.profile,
            sampling_config=self.sampling,
            samples=self.samples,
            s2_registry=self.s2_reg,
            s5_denylist=self.s5_denylist,
            checkpoint_paths=self.ckpts,
            out_dir=dir2,
            spec=spec2,
            extra_denied_contexts=c1_fresh_ctx,
            extra_denied_physical_states=c1_phys,
        )
        c2_rows = [
            json.loads(line)
            for p in list(dir2.glob("s3_plan_w0*.jsonl")) + list(dir2.glob("s3_reserve_w0*.jsonl"))
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        c2_phys = {r["physical_state_key"] for r in c2_rows}
        c2_fresh_ctx = {r["semantic_context_sha256"] for r in c2_rows if r.get("anchor_sample_id") is None}
        self.assertEqual(c1_phys & c2_phys, set())
        self.assertEqual(c1_fresh_ctx & c2_fresh_ctx, set())
        self.assertEqual(res1["sidecar"]["s5_denylist_sha256"], res2["sidecar"]["s5_denylist_sha256"])


if __name__ == "__main__":
    unittest.main()
