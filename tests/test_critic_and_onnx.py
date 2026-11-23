from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import tempfile
import unittest
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from labyrinth_nn.catalog.schema import (
    ENEMY_STATIC_FEAT_NAMES, GLOBAL_FEAT_NAMES, TEAM_SYNERGY_FEAT_NAMES, UNIT_STATIC_FEAT_NAMES,
    EnemyEntry, FeatureCatalog, QuestEntry, RelicEntry, ScoreSettings, UnitEntry,
)
from labyrinth_nn.data.dataset import (
    BattleDataset, build_vocab_maps, collate_battle_batch, featurize_sample, mask_gold_sample, split_samples,
)
from labyrinth_nn.data.sample_schema import BattleSample, write_samples_jsonl
from labyrinth_nn.train.metrics import EVAL_BUCKETS, compute_slice_metrics, evaluate_predictions

HAS_TORCH = find_spec("torch") is not None
HAS_ONNX = find_spec("onnx") is not None
HAS_ORT = find_spec("onnxruntime") is not None
HAS_ONNX_AND_ORT = HAS_ONNX and HAS_ORT and find_spec("numpy") is not None
if HAS_TORCH:
    import torch
    from labyrinth_nn.models.critic import (
        CriticConfig, SingleTeamPriorityRanker, TowerAOnnxWrapper, TowerBOnnxWrapper,
        TwoTowerBattleCritic, TwoTowerEnsemble,
    )
    from labyrinth_nn.train.losses import compute_critic_loss, fit_temperature
    from labyrinth_nn.train import train_critic as training
    from labyrinth_nn.train.train_critic import run_experiments, split_training_samples, train_critic

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def tiny_catalog():
    widths = [90, 90, 390, 490, 780, 220, 260, 330, 540, 610, 650, 700, 800, 830, 880, 950]
    units = {}
    for i, width in enumerate(widths):
        uid = 101 + i
        units[uid] = UnitEntry(
            unit_id=uid, name=f"unit-{uid}", search_area_width=width, atk_type=1 + i % 2,
            talent_id=1 + i // 5, role_id=1 + i % 8, npc_enhance_id=0,
            has_6star=True, has_ue1=True, has_ue2=False, has_labyrinth_replace=i == 0,
            panel_hp=10000 + uid, panel_atk=1000 + uid, panel_magic_str=700 + uid,
            panel_def=600 + i, panel_magic_def=400 + i, mech_flags={"taunt_decoy": i == 0},
        )
    enemies = {i: EnemyEntry(i, i + 1000, f"enemy-{i}", 80, 30000, 30000, 500, 450,
                             250, 125, 20, 100, 0, is_boss_unit=i == 202, has_cc_resist=i == 202)
               for i in (201, 202, 203)}
    relics = {
        501: RelicEntry(501, "mark", 1, 1, 2, 100),
        502: RelicEntry(502, "guard", 2, 2, 3, 200, front_pdef_to_atk_mult=1.5, front_mdef_to_atk_mult=0.5),
        503: RelicEntry(503, "gold", 2, 3, 2, 300, gold_scaling=True),
        504: RelicEntry(504, "owned", 2, 4, 4, 400, owned_unit_scaling=True),
        505: RelicEntry(505, "element", 2, 5, 1, 500, element_bonus_amp=0.3),
    }
    quests = {
        301: QuestEntry(301, 1, 1, 1, 90, 0, [201, 0, 203], 60000, "wave-a", "normal"),
        302: QuestEntry(302, 2, 2, 3, 90, 0, [203], 30000, "wave-r", "robot_ex"),
        303: QuestEntry(303, 3, 3, 5, 90, 0, [202], 30000, "wave-b", "boss_5f_d5", True),
    }
    return FeatureCatalog(1, "mini", "", "", 5, 5, units, enemies, quests, relics, {},
                          ScoreSettings(), [], [], {})


def battle(**changes):
    sample = BattleSample(
        sample_id="s1", round_id="pilot", source="sim", db_version="mini", quest_id=301,
        canonical_wave_sig="wave-a", bucket="normal", area=3, difficulty=4, guild_id=1,
        team_unit_ids=[105, 104, 103, 102, 101], owned_unit_ids=list(range(116, 100, -1)),
        owned_unit_count=30, owned_role_counts=[2, 4, 0, 1], owned_talent_counts=[5, 5, 5, 1],
        relic_ids=[501, 502, 503, 504, 505], mark_stacks=[4, 9, 15, 3, 20], alpha_rupee=8000,
        revive_available=True, outcome_counts=[2, 1, 1], mean_damage_ratio=0.8, mean_win_rem_time=0.4,
    )
    return replace(sample, **changes)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.catalog = tiny_catalog()
        self.maps = build_vocab_maps(self.catalog)

    def features(self, sample, **options):
        return featurize_sample(self.catalog, sample, **self.maps, **options)

    def test_counterfactual_gold_is_masked_before_features(self):
        sample = battle(context_id="paired", relic_ids=[503, 999])
        known = self.features(sample)
        gold_unknown = self.features(mask_gold_sample(sample))
        synergy = TEAM_SYNERGY_FEAT_NAMES.index("gold_x_gold_relic")
        self.assertGreater(known["team_synergy"][synergy], 0)
        self.assertEqual(gold_unknown["team_synergy"][synergy], 0)
        self.assertEqual(sample.alpha_rupee, 8000)
        self.assertFalse(gold_unknown["gold_known"])
        self.assertTrue(gold_unknown["has_gold_scaling_relic"])
        self.assertEqual(gold_unknown["context_id"], "paired")
        self.assertFalse(self.features(sample, training=True, relic_id_mask_prob=1)["has_gold_scaling_relic"])

    def test_slots_signed_distance_ties_and_padding(self):
        sample = battle(team_unit_ids=[104, 102, 101, 103, 102, 0])
        row = self.features(sample)
        self.assertEqual(row["team_unit_ids_sorted"], [101, 102, 103, 104, 0])
        self.assertEqual(row["team_slot_mask"], [True] * 4 + [False])
        reversed_team = self.features(replace(sample, team_unit_ids=sample.team_unit_ids[::-1]))
        self.assertEqual(reversed_team["team_pos_matrix"], row["team_pos_matrix"])
        widths = [90, 90, 390, 490]
        self.assertEqual(len(row["team_pos_matrix"]), 5)
        for i, line in enumerate(row["team_pos_matrix"]):
            self.assertEqual(len(line), 5)
            for j, channels in enumerate(line):
                expected = [0.0] * 3
                if i < 4 and j < 4:
                    diff = widths[j] - widths[i]
                    expected = [diff / 1000, float(abs(diff) <= 300), float(abs(diff) <= 400)]
                self.assertEqual(channels, expected)
        self.assertEqual(row["enemy_ids"], [1, 0, 3, 0, 0])
        self.assertEqual(row["enemy_mask"], [True, False, True, False, False])
        self.assertEqual(len(row["enemy_feats"][0]), len(ENEMY_STATIC_FEAT_NAMES))
        self.assertAlmostEqual(row["enemy_feats"][0][1], math.log1p(30000))
        self.assertEqual(row["enemy_feats"][1], [0.0] * 11)
        self.assertEqual(row["outcome_probs_target"], [0.5, 0.25, 0.25])
        self.assertTrue(row["has_win_target"])

    def test_exact_roster_indices_survive_dropout_and_truncation(self):
        sample = battle(team_unit_ids=[116, 114, 115], owned_unit_ids=list(range(101, 117)))
        baseline = self.features(sample, roster_max_len=5)
        row = self.features(sample, roster_max_len=5, training=True, unit_id_dropout=1.0, rng=random.Random(19))
        self.assertEqual(row["roster_unit_ids"], [0] * 5)
        self.assertEqual(row["roster_raw_ids"], [114, 115, 116, 101, 102])
        self.assertEqual(row["roster_mask"], [True] * 5)
        self.assertEqual(row["roster_unit_feats"], baseline["roster_unit_feats"])
        self.assertEqual(row["team_roster_pos"][3:], [0, 0])
        for uid, pos in zip(row["team_unit_ids_sorted"][:3], row["team_roster_pos"]):
            unit = self.catalog.units[uid]
            self.assertEqual(row["roster_raw_ids"][pos], uid)
            self.assertEqual(baseline["roster_unit_ids"][pos], self.maps["unit_to_idx"][uid])
            self.assertEqual(len(row["roster_unit_feats"][pos]), 27)
            self.assertAlmostEqual(row["roster_unit_feats"][pos][0], unit.search_area_width / 1000)
            self.assertAlmostEqual(row["roster_unit_feats"][pos][10], math.log1p(unit.panel_def))
        self.assertEqual(row["roster_unit_feats"][3][UNIT_STATIC_FEAT_NAMES.index("taunt_decoy")], 1.0)
        with self.assertRaisesRegex(ValueError, "every active team"):
            self.features(sample, roster_max_len=2)

    def test_global_features_and_synergy_channels(self):
        row = self.features(battle())
        self.assertEqual(len(row["global_feats"]), 41)
        stacks = [4, 9, 15, 3, 20]
        expected = ([s / 15 for s in stacks] + [float(s >= t) for t in (4, 9, 15) for s in stacks]
                    + [1.0, 0.5] + [x / 20 for x in (2, 4, 0, 1, 0, 0, 0, 0)]
                    + [x / 20 for x in (5, 5, 5, 1, 0)] + [0.8, 1.0, 0.8, 0.6, 0.0, 1.0])
        self.assertEqual(row["global_feats"], expected)
        self.assertEqual(len(row["team_synergy"]), 17)
        expected = [1.0, 1, 0, 0, 0, 0, 1, 0.75, 1, 0.6, 0.4, 0, 0.9, 0.2, 0.8, 1.0, 0.3]
        for key, actual, want in zip(TEAM_SYNERGY_FEAT_NAMES, row["team_synergy"], expected):
            self.assertAlmostEqual(actual, want, msg=key)
        self.catalog.units[101].static_feats = [0.25] * 27
        self.catalog.units[101].static_feats[2] = 0.75
        self.catalog.enemies[201].static_feats = [0.375] * 11
        row = self.features(battle(team_unit_ids=[101], alpha_rupee=12000, owned_unit_count=10))
        self.assertEqual(row["roster_unit_feats"][0], self.catalog.units[101].static_feats)
        self.assertEqual(row["enemy_feats"][0], [0.375] * 11)
        synergy = dict(zip(TEAM_SYNERGY_FEAT_NAMES, row["team_synergy"]))
        self.assertEqual((synergy["phys_ratio"], synergy["mag_ratio"]), (1.0, 1.0))
        self.assertEqual((synergy["gold_x_gold_relic"], synergy["owned_x_owned_relic"]), (1.0, 0.4))

    def test_gold_probability_and_coupled_relic_mark_noise(self):
        sample = battle(mark_stacks=[1, 5, 2, 4, 1])
        for seed, is_known in ((1, False), (2, True)):
            row = self.features(sample, training=True, gold_mask_prob=0.55, rng=random.Random(seed))
            self.assertEqual(row["gold_known"], is_known)
            self.assertEqual(row["team_synergy"][14], 0.8 if is_known else 0.0)
        for sample_gold, training, prob in ((None, False, 0), (8000, True, 1)):
            row = self.features(replace(sample, alpha_rupee=sample_gold), training=training, gold_mask_prob=prob)
            self.assertEqual(row["global_feats"][35:37], [0.0, 0.0])
        clean = self.features(sample, training=False, relic_id_mask_prob=1, relic_stack_noise_prob=1,
                              gold_mask_prob=1, unit_id_dropout=1)
        self.assertEqual(clean, self.features(sample))
        no_id = self.features(sample, training=True, relic_id_mask_prob=1, relic_stack_noise_prob=0)
        self.assertEqual(no_id["global_feats"][:5], clean["global_feats"][:5])
        removed = self.features(sample, training=True, relic_id_mask_prob=1, relic_stack_noise_prob=1)
        self.assertFalse(any(removed["relic_mask"]))
        self.assertEqual(removed["global_feats"][:5], [0.0, 2 / 15, 0.0, 0.0, 0.0])
        self.assertEqual(removed["global_feats"][GLOBAL_FEAT_NAMES.index("has_empty_relics")], 1.0)
        self.assertEqual(removed["team_synergy"][12:], [0.0] * 5)
        no_mask = self.features(sample, training=True, relic_id_mask_prob=0, relic_stack_noise_prob=1)
        self.assertEqual(no_mask, clean)

    def test_unknown_entries_and_missing_targets(self):
        row = self.features(battle(quest_id=-7, area=-1, difficulty=-2, team_unit_ids=[999, 102],
                                   relic_ids=[], mark_stacks=[4], outcome_counts=[0, 0, 0],
                                   mean_damage_ratio=2.3, mean_win_rem_time=2))
        self.assertEqual(row["team_unit_ids_sorted"], [102, 999, 0, 0, 0])
        self.assertEqual(row["roster_unit_ids"][1], 0)
        self.assertTrue(row["roster_mask"][1])
        self.assertEqual(row["roster_unit_feats"][1], [0.0] * 27)
        self.assertEqual((row["quest_idx"], row["area_idx"], row["difficulty_idx"]), (0, 0, 0))
        self.assertFalse(any(row["enemy_mask"]))
        self.assertEqual(row["outcome_probs_target"], [0.0, 0.0, 1.0])
        self.assertFalse(row["has_outcome_target"])
        self.assertEqual((row["mean_damage_ratio"], row["mean_win_rem_time"], row["has_win_target"]), (1.0, 0.0, False))
        if not HAS_TORCH:
            for call in (lambda: BattleDataset(self.catalog, [battle()]), lambda: collate_battle_batch([row])):
                with self.assertRaisesRegex(ImportError, "PyTorch"):
                    call()

    def test_group_split_is_stable_and_leak_free(self):
        samples = [battle(sample_id=f"{i}-{j}", round_id=str(j), difficulty=j + 1,
                          quest_id=301 + j, canonical_wave_sig=f"wave-{i // 2}", relic_ids=[501 + j],
                          team_unit_ids=([101 + i % 2, 104] if j % 2 else [104, 101 + i % 2]))
                   for i in range(12) for j in range(3)]
        def keys(part):
            return {(s.canonical_wave_sig, s.canonical_team_sig()) for s in part}
        parts = split_samples(samples, 0.25, 0.25, 11)
        memberships = list(map(keys, parts))
        self.assertEqual([len(p) for p in memberships], [6, 3, 3])
        self.assertEqual(sum(map(len, parts)), 36)
        for i in range(3):
            for j in range(i):
                self.assertTrue(memberships[i].isdisjoint(memberships[j]))
        self.assertEqual(memberships, list(map(keys, split_samples(samples[::-1], 0.25, 0.25, 11))))
        self.assertNotEqual(memberships, list(map(keys, split_samples(samples, 0.25, 0.25, 12))))
        for val, test in ((0.001, 0.998), (0.998, 0.001), (0.001, 0.001)):
            self.assertTrue(all(split_samples(samples[:9], val, test)))
        self.assertEqual(split_samples(samples, 0, 0)[1:], ([], []))
        for ratios in ((-0.1, 0.2), (0.2, -0.1), (0.5, 0.5)):
            with self.assertRaises(ValueError):
                split_samples(samples, *ratios)


class MetricTests(unittest.TestCase):
    def test_context_top1_ranking_and_gold_scaling_relic_slice(self):
        def row(context, team, p, y, *, relic):
            return {"context_id": context, "team_sig": team, "canonical_wave_sig": "shared-wave",
                    "bucket": "normal", "gold_known": False, "has_gold_scaling_relic": relic,
                    "pred_probs": [p, 1-p, 0], "target_probs": [y, 1-y, 0],
                    "pred_damage_mean": 0.8, "target_damage_ratio": 0.5, "has_win_target": False}
        rows = [
            row("a", "1", 0.8, 0.25, relic=True), row("a", "2", 0.6, 1.0, relic=True),
            row("b", "1", 0.4, 0.75, relic=False), row("b", "3", 0.3, 0.0, relic=False),
            row("incomplete", "1", 0.9, 0.0, relic=True), row("incomplete", "1", 0.8, 0.2, relic=True),
        ]
        metrics = evaluate_predictions(rows)
        self.assertEqual(metrics["global"]["context_count"], 2)
        self.assertEqual(metrics["global"]["context_top1_fpr"], 0.5)
        self.assertEqual(metrics["global"]["context_top1_pos_fpr"], 1.0)
        self.assertEqual(metrics["global"]["context_top1_win_regret"], 0.375)
        relic = metrics["slices"]["gold_scaling_relic"]
        self.assertEqual((relic["count"], relic["context_count"], relic["context_top1_win_regret"]), (4, 1, 0.75))
        empty = metrics["buckets"]["other_ex"]
        self.assertEqual(empty["context_count"], 0)
        self.assertIsNone(empty["context_top1_fpr"])
        self.assertIsNone(empty["context_top1_pos_fpr"])
        self.assertIsNone(empty["context_top1_win_regret"])

    def test_soft_calibration_ranking_and_slices(self):
        def record(wave, team, p, y, damage, time, bucket, gold):
            return {"canonical_wave_sig": wave, "team_sig": team, "pred_probs": p, "target_probs": y,
                    "pred_damage_mean": damage[0], "target_damage_ratio": damage[1],
                    "has_win_target": time is not None, "pred_win_rem_time": time[0] if time else 0.0,
                    "target_win_rem_time": time[1] if time else None, "bucket": bucket, "gold_known": gold}
        rows = [
            record("a", "1", [.8, .1, .1], [0, 1, 0], [.8, .5], None, "normal", False),
            record("a", "2", [.6, .3, .1], [.75, .25, 0], [1, 1], [.3, .5], "robot_ex", True),
            record("b", "3", [.8, .2, 0], [1, 0, 0], [1, 1], [.7, .6], "boss_5f_d5", True),
            record("c", "4", [.2, .5, .3], [0, 0, 1], [.1, .5], None, "other_ex", False),
        ]
        report = evaluate_predictions(iter(rows))
        expected = {"count": 4, "auroc": 0.625, "brier_3class": 2.355 / 4, "brier_win": .7425 / 4,
                    "ece_15": .95 / 4, "accuracy_3class": .5, "loss_damage_mae": .35, "win_time_mae": .15,
                    "group_top1_fpr": 2 / 3, "group_top1_pos_fpr": .5}
        for key, value in expected.items():
            self.assertAlmostEqual(report["global"][key], value, msg=key)
        self.assertEqual(tuple(report["buckets"]), EVAL_BUCKETS)
        self.assertEqual(report["buckets"]["robot_ex"]["count"], 1)
        self.assertIsNone(report["buckets"]["robot_ex"]["loss_damage_mae"])
        self.assertIsNone(report["buckets"]["special_dual_ex"]["brier_win"])
        unknown = report["slices"]["gold_unknown"]
        self.assertEqual(unknown["count"], 2)
        self.assertIsNone(unknown["auroc"])
        self.assertIsNone(unknown["win_time_mae"])
        self.assertEqual(unknown["group_top1_fpr"], 1)
        self.assertIsNone(compute_slice_metrics([rows[-1]])["group_top1_pos_fpr"])
        ties = [rows[0], {**rows[2], "canonical_wave_sig": "a"}]
        self.assertEqual(compute_slice_metrics(ties)["group_top1_fpr"],
                         compute_slice_metrics(ties[::-1])["group_top1_fpr"])


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)
        self.catalog = tiny_catalog()
        self.maps = build_vocab_maps(self.catalog)

    def config(self, mode):
        c = CriticConfig.from_json(CONFIG_DIR / f"critic_{mode}.json", self.catalog)
        self.assertEqual(CriticConfig.from_dict(c.to_dict()), c)
        return replace(c, d_model=16, dropout=0.0, roster_max_len=16, relic_max_len=8)

    def batch(self, samples):
        return collate_battle_batch([featurize_sample(self.catalog, s, **self.maps, roster_max_len=16,
                                                     relic_max_len=8) for s in samples])

    def experiment_fixture(self, *modes, epochs=1):
        configs = [replace(self.config(mode), epochs=epochs, batch_size=16, val_ratio=.25, test_ratio=.25,
                           n_context_layers=1, n_team_layers=1) for mode in modes]
        samples = [battle(sample_id=f"s-{i}", canonical_wave_sig=f"wave-{i}", outcome_counts=[i % 3, 1, 1],
                          alpha_rupee=None if i % 2 else 1000) for i in range(16)]
        return configs, samples

    def test_eval_cache_and_epoch_seeded_training_features(self):
        evaluation = BattleDataset(self.catalog, [battle()], cache_eval=True, gold_mask_prob=1)
        with patch("labyrinth_nn.data.dataset.featurize_sample", wraps=featurize_sample) as featurize:
            self.assertTrue(evaluation[0]["gold_known"])
            evaluation.set_epoch(2)
            self.assertTrue(evaluation[0]["gold_known"])
            self.assertEqual(featurize.call_count, 1)
        training_data = BattleDataset(self.catalog, [battle()], training=True, cache_eval=True,
                                      unit_id_dropout=0.5, relic_id_mask_prob=0.5, gold_mask_prob=0.5)
        first = training_data[0]
        self.assertEqual(first, training_data[0])
        training_data.set_epoch(1)
        self.assertNotEqual(first, training_data[0])
        training_data.set_epoch(0)
        self.assertEqual(first, training_data[0])

    def test_damage_modes_mask_wins_and_detach_nll_mean(self):
        config = self.config("m2b_r1")
        model = TwoTowerBattleCritic(config)
        with torch.no_grad():
            model.tower_b.damage_log_std.weight.zero_()
            model.tower_b.damage_log_std.bias.fill_(-9)
        batch = self.batch([battle(outcome_counts=[2, 1, 1]), battle(outcome_counts=[0, 3, 1])])
        outputs = model(batch)
        torch.testing.assert_close(outputs["damage_log_std"], torch.full((2, 1), -2.5))
        outputs["damage_log_std"] = torch.full((2, 1), -2.5, requires_grad=True)
        masked, parts = compute_critic_loss(outputs, batch, config)
        huber, huber_parts = compute_critic_loss(outputs, batch, replace(config, damage_loss_mode="huber_only"))
        mean_grad, var_grad = torch.autograd.grad(
            masked, (outputs["damage_mean"], outputs["damage_log_std"]), retain_graph=True,
        )
        huber_mean_grad, huber_var_grad = torch.autograd.grad(
            huber, (outputs["damage_mean"], outputs["damage_log_std"]), retain_graph=True,
        )
        torch.testing.assert_close(mean_grad, huber_mean_grad)
        self.assertEqual(mean_grad[0].item(), 0)
        self.assertEqual(var_grad[0].item(), 0)
        self.assertNotEqual(var_grad[1].item(), 0)
        self.assertEqual(huber_var_grad.count_nonzero().item(), 0)
        self.assertEqual(huber_parts["damage_nll"].item(), 0)
        error = batch["mean_damage_ratio"][1] - outputs["damage_mean"][1].detach()
        expected = 0.5 * (error.square() / math.exp(-5) - 5)
        self.assertAlmostEqual(parts["damage_nll"].item(), expected.item(), places=5)
        only_wins = self.batch([battle(outcome_counts=[1, 0, 0])])
        for mode in ("masked_nll", "huber_only"):
            win_outputs = model(only_wins)
            loss, components = compute_critic_loss(win_outputs, only_wins, replace(config, damage_loss_mode=mode))
            self.assertEqual(components["damage_loss"].item(), 0)
            grads = torch.autograd.grad(loss, (win_outputs["damage_mean"], win_outputs["damage_log_std"]))
            self.assertTrue(all(g.count_nonzero().item() == 0 for g in grads))
        for invalid in ({"damage_log_std_min": 1}, {"damage_loss_mode": "other"}, {"lr_schedule": "other"}):
            with self.assertRaises(ValueError):
                replace(config, **invalid)

    def test_cosine_resume_gradient_clip_components_and_counterfactual_reports(self):
        config = replace(self.config("m2b_r1"), epochs=3, batch_size=4, val_ratio=0.25, test_ratio=0.25,
                         warmup_ratio=0.4)
        samples = [
            battle(sample_id=f"s-{i}", context_id=f"context-{i // 2}", canonical_wave_sig=f"wave-{i // 2}",
                   team_unit_ids=[101, 102, 103, 104, 105 + i % 2], outcome_counts=[0, 2, 0] if i % 2 else [2, 0, 0],
                   mean_win_rem_time=None if i % 2 else 0.5)
            for i in range(32)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(torch.nn.utils, "clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
                summary = run_experiments([config], self.catalog, samples, root / "full", device="cpu",
                                          group_by="context")
                self.assertGreater(clip.call_count, config.epochs)
                self.assertTrue(all(call.args[1] == 1.0 for call in clip.call_args_list))
            mode = summary["modes"]["m2b_r1"]
            run = mode["runs"][0]
            history = run["history"]
            for row in history:
                for key in (*training.LOSS_COMPONENTS, "lr"):
                    self.assertTrue(math.isfinite(row[key]), key)
            self.assertLess(history[0]["lr"], history[1]["lr"])
            self.assertLess(history[2]["lr"], history[1]["lr"])
            checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
            self.assertAlmostEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], config.lr * config.min_lr_ratio)
            test_report = json.loads(Path(run["checkpoint"]).with_name("test_metrics.json").read_text(encoding="utf-8"))
            masked = test_report["counterfactual_gold_unknown"]
            self.assertEqual(masked["global"]["count"], run["split_counts"]["test"])
            self.assertEqual(masked["slices"]["gold_scaling_relic"]["count"], masked["global"]["count"])
            self.assertGreater(masked["global"]["context_count"], 0)
            self.assertEqual(run["metrics"]["slices"]["gold_unknown"]["count"], 0)
            saved_summary = json.loads((root / "full" / "ablation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["modes"]["m2b_r1"]["counterfactual_gold_unknown"], masked)
            markdown = (root / "full" / "ablation_summary.md").read_text(encoding="utf-8")
            self.assertIn("| gold_unknown_cf |", markdown)
            self.assertIn("| gold_unknown_cf_gold_relic |", markdown)

            save = training._save_checkpoint

            def interrupt_after_epoch(path, payload):
                save(path, payload)
                raise KeyboardInterrupt

            with patch.object(training, "_save_checkpoint", side_effect=interrupt_after_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    train_critic(config, self.catalog, samples, root / "partial", device="cpu", group_by="context")
            resumed = train_critic(config, self.catalog, samples, root / "resumed", device="cpu",
                                   resume=root / "partial" / "last.pt", group_by="context")
            self.assertEqual(resumed["history"], history)
            self.assertTrue(all(not row["gold_known"] for row in resumed["counterfactual_predictions"]))
            self.assertEqual(resumed["counterfactual_gold_unknown"], masked)

    def test_cli_rejects_duplicate_ids_and_mixed_db_versions_across_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = write_samples_jsonl(root / "first.jsonl", [battle()])
            second = root / "second.jsonl"
            args = ["--config-files", str(CONFIG_DIR / "critic_m2b_r1.json"), "--catalog", str(root / "catalog.json"),
                    "--out-dir", str(root / "fit"), "--samples", str(first), str(second)]
            write_samples_jsonl(second, [battle()])
            with self.assertRaisesRegex(ValueError, "duplicate sample_id: s1"):
                training.main(args)
            write_samples_jsonl(second, [battle(sample_id="s2", db_version="other")])
            with self.assertRaisesRegex(ValueError, "mixed db_version"):
                training.main(args)

    def test_four_modes_forward_backward_and_exact_wrapper_inputs(self):
        samples = [battle(team_unit_ids=[102, 101, 105], relic_ids=[], area=1000, mean_win_rem_time=1.0),
                   battle(team_unit_ids=[], owned_unit_ids=[], owned_unit_count=0, quest_id=0,
                          relic_ids=[], outcome_counts=[0, 3, 0], mean_win_rem_time=None)]
        batch = self.batch(samples)
        self.assertFalse(batch["relic_mask"].any())
        for mode in ("m0", "m1", "m2a", "m2b"):
            with self.subTest(mode=mode):
                c = self.config(mode)
                model = TwoTowerBattleCritic(c)
                out = model(batch)
                for key, shape in {"outcome_logits": (2, 3), "outcome_probs": (2, 3),
                                   "damage_mean_logit": (2, 1), "damage_mean": (2, 1), "damage_std": (2, 1),
                                   "damage_log_std": (2, 1), "win_rem_time": (2, 1), "unit_affinity": (2, 16),
                                   "roster_ctx": (2, 16, 16), "context_summary": (2, 16)}.items():
                    self.assertEqual(tuple(out[key].shape), shape)
                    self.assertTrue(torch.isfinite(out[key]).all(), key)
                self.assertEqual(torch.count_nonzero(out["roster_ctx"][1]).item(), 0)
                self.assertTrue((out["damage_log_std"] >= -5).all() and (out["damage_log_std"] <= 1).all())
                self.assertTrue((out["win_rem_time"] >= 0).all() and (out["win_rem_time"] <= 1).all())
                torch.testing.assert_close(out["outcome_probs"].sum(-1), torch.ones(2))
                loss, parts = compute_critic_loss(out, batch, c)
                expected_ce = -(batch["outcome_probs_target"] * out["outcome_logits"].log_softmax(-1)).sum(-1).mean()
                torch.testing.assert_close(parts["outcome_loss"], expected_ce)
                time_grad, = torch.autograd.grad(loss, out["win_rem_time"], retain_graph=True)
                self.assertNotEqual(time_grad[0].item(), 0)
                self.assertEqual(time_grad[1].item(), 0)
                loss.backward()
                for name, param in model.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertTrue(all(torch.isfinite(v) for v in parts.values()))
                a, b = TowerAOnnxWrapper(model), TowerBOnnxWrapper(model)
                for wrapper in (a, b):
                    self.assertEqual(tuple(inspect.signature(wrapper.forward).parameters), wrapper.input_names)
                self.assertEqual("enemy_feats" in a.input_names, c.use_static_feats)
                self.assertEqual("roster_unit_feats" in a.input_names, c.use_static_feats)
                self.assertEqual("global_feats" in a.input_names, c.use_global_feats)
                self.assertEqual("team_pos_matrix" in b.input_names, c.use_pos_bias)
                self.assertEqual("team_synergy" in b.input_names, c.use_global_feats)
                encoded = dict(zip(a.output_names, a(*(batch[k] for k in a.input_names))))
                inputs = {**batch, **encoded}
                for name, value in zip(b.output_names, b(*(inputs[k] for k in b.input_names))):
                    torch.testing.assert_close(value, out[name])
        failed = self.batch([samples[1]])
        model = TwoTowerBattleCritic(self.config("m0"))
        out = model(failed)
        loss, components = compute_critic_loss(out, failed, model.config)
        self.assertEqual(components["time_loss"].item(), 0.0)
        grad, = torch.autograd.grad(loss, out["win_rem_time"])
        self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))
        if torch.cuda.is_available():
            model = TwoTowerBattleCritic(self.config("m2b")).cuda().half().eval()
            unknown = self.batch([battle(team_unit_ids=[101, 999], relic_ids=[])])
            cuda_batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in unknown.items()}
            with torch.no_grad():
                self.assertTrue(all(torch.isfinite(v).all() for v in model(cuda_batch).values()))

    def test_independent_temperature_fit(self):
        model = TwoTowerBattleCritic(self.config("m1"))
        with torch.no_grad():
            model.tower_b.outcome.weight.zero_()
            model.tower_b.outcome.bias.copy_(torch.tensor([5., 0., -1.]))
        batch = self.batch([battle(outcome_counts=[1, 1, 1]), battle(outcome_counts=[1, 1, 1])])
        report = fit_temperature(model, [batch], "cpu")
        self.assertGreater(model.log_temperature.item(), 0)
        self.assertLess(report["nll_after"], report["nll_before"])
        self.assertAlmostEqual(model.log_temperature.exp().item(), report["temperature"])
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        nll = -(batch["outcome_probs_target"] * model(batch)["outcome_probs"].log()).sum(-1).mean()
        self.assertAlmostEqual(nll.item(), report["nll_after"], places=5)

    def test_cached_ensemble_priority_filter_and_exact_packing(self):
        class ScriptedCritic(TwoTowerBattleCritic):
            def __init__(self, config, offset):
                super().__init__(config)
                self.calls = 0
                self.offset = offset

            def encode_context(self, batch):
                self.calls += 1
                ids = batch["roster_unit_ids"].float()
                roster = ids.unsqueeze(-1).expand(-1, -1, self.config.d_model)
                return roster, -ids, roster[:, 0]

            def score_team(self, roster, summary, batch):
                ids = roster[:, :, 0].gather(1, batch["team_roster_pos"])
                groups = ((ids - 1) / 5).floor()
                winner = (groups == groups[:, :1]).all(1)
                win = torch.where(winner, 0.92, 0.03) + self.offset
                probs = torch.stack((win, (1 - win) * 0.9, (1 - win) * 0.1), 1)
                damage = torch.where(winner, 0.1, 0.99).unsqueeze(1)
                time = (0.8 - 0.1 * groups[:, 0]).unsqueeze(1)
                return probs.log(), damage.logit(), torch.zeros_like(damage), time

        c = self.config("m0")
        models = [ScriptedCritic(c, -0.01), ScriptedCritic(c, 0.01)]
        ensemble = TwoTowerEnsemble(models, lcb_z=2)
        output = ensemble(self.batch([battle(), battle(team_unit_ids=[101, 102, 103, 104, 106])]))
        torch.testing.assert_close(output["p_win_std"], torch.tensor([0.01, 0.01]))
        torch.testing.assert_close(output["lcb_win"], torch.tensor([0.9, 0.01]))
        ranker = SingleTeamPriorityRanker(ensemble, self.catalog)
        result = ranker.recommend(battle(), top_k=120, max_candidates=120, batch_size=9)
        self.assertEqual([m.calls for m in models], [2, 2])
        ranked = result["ranked_teams"]
        self.assertEqual(result["evaluated_count"], len(ranked))
        self.assertGreater(len(ranked), 15)
        self.assertEqual([t["team_sig"] for t in ranked[:3]], ["101,102,103,104,105", "106,107,108,109,110",
                                                              "111,112,113,114,115"])
        self.assertGreater(ranked[-1]["damage_mean"], ranked[0]["damage_mean"])
        self.assertEqual(len(result["packing"]), 3)
        self.assertEqual(len({uid for t in result["packing"] for uid in t["team_unit_ids"]}), 15)
        self.assertAlmostEqual(result["joint_win_probability"], 1 - 0.1**3, places=5)
        failed = ranked[0]["team_sig"]
        filtered = ranker.recommend(battle(), failed_team_signatures={failed}, top_k=120, max_candidates=120)
        self.assertTrue(all(t["team_sig"] != failed for t in filtered["ranked_teams"] + filtered["packing"]))
        self.assertEqual([m.calls for m in models], [3, 3])
        almost = {"lcb_win": 1e-5, "win_rem_time": 0.0, "damage_mean": 0.0}
        timeout = {"lcb_win": 0.0, "win_rem_time": 1.0, "damage_mean": 1.0}
        self.assertGreater(ranker.priority(almost), ranker.priority(timeout))
        self.assertGreater(ranker.priority(timeout), ranker.priority({**timeout, "damage_mean": 0.5}))
        trap = [{"team_sig": str(i), "team_unit_ids": team, "lcb_win": p, "damage_mean": 0.1}
                for i, (team, p) in enumerate([([1, 2, 3], .9), ([1, 4], .65), ([2, 5], .65), ([3, 6], .65)])]
        self.assertEqual({t["team_sig"] for t in ranker.pack_teams(trap)}, {"1", "2", "3"})

    def test_training_resume_and_ensemble_report(self):
        c = replace(self.config("m0"), epochs=1, batch_size=4, val_ratio=.3, test_ratio=.2,
                    n_context_layers=1, n_team_layers=1)
        samples = [battle(sample_id=str(i), canonical_wave_sig=f"wave-{i}", outcome_counts=[i % 3, 1, 1],
                          alpha_rupee=None if i % 2 else 1000) for i in range(16)]
        parts = split_training_samples(samples, c, 42)
        memberships = [{(s.canonical_wave_sig, s.canonical_team_sig()) for s in rows} for rows in parts.values()]
        self.assertEqual(len(set.union(*memberships)), sum(map(len, memberships)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_experiments([c], self.catalog, samples, root, seeds=[42, 43], device="cpu")
            report = summary["modes"]["m0"]
            self.assertEqual(report["ensemble"]["global"]["count"], len(parts["test"]))
            self.assertEqual(len(report["runs"]), 2)
            for run in report["runs"]:
                self.assertLessEqual(run["calibration"]["nll_after"], run["calibration"]["nll_before"])
                self.assertEqual(run["split_counts"]["val_calib"], len(parts["val_calib"]))
            self.assertTrue((root / "ablation_summary.json").is_file())
            self.assertIn("gold_unknown", (root / "ablation_summary.md").read_text(encoding="utf-8"))
            resumed = train_critic(replace(c, epochs=2), self.catalog, samples, root / "resume", device="cpu",
                                   resume=report["runs"][0]["checkpoint"])
            self.assertEqual([r["epoch"] for r in resumed["history"]], [1, 2])
            self.assertEqual(resumed["metrics"]["global"]["count"], len(parts["test"]))
            with self.assertRaisesRegex(ValueError, "sample split"):
                train_critic(c, self.catalog, samples[:-1], root / "changed", device="cpu",
                             resume=report["runs"][1]["checkpoint"])

    def test_auto_resume_skips_completed_resumes_partial_and_preserves_summary(self):
        configs, samples = self.experiment_fixture("m0", "m1", epochs=2)
        first, second = configs
        seeds = (42, 43)
        options = dict(split_seed=42, device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean, target = root / "clean", root / "target"
            run_experiments(configs, self.catalog, samples, clean, seeds=seeds, **options)

            completed_dir = target / first.ablation_mode / "seed_42"
            train_critic(first, self.catalog, samples, completed_dir, seed=42, **options)
            best_path = completed_dir / "best.pt"
            state = torch.load(best_path, map_location="cpu", weights_only=True)
            state.pop("group_by")
            state.pop("calibration")
            torch.save(state, best_path)
            metrics_path = completed_dir / "test_metrics.json"
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            for field in ("calibration", "counterfactual_gold_unknown"):
                report.pop(field)
            metrics_path.write_text(json.dumps(report), encoding="utf-8")

            interrupted_dir = target / first.ablation_mode / "seed_43"
            save_checkpoint = training._save_checkpoint

            def interrupt_after_epoch_one(path, payload):
                save_checkpoint(path, payload)
                if path.name == "last.pt" and payload["epoch"] == 1:
                    raise KeyboardInterrupt

            with patch.object(training, "_save_checkpoint", side_effect=interrupt_after_epoch_one):
                with self.assertRaises(KeyboardInterrupt):
                    train_critic(first, self.catalog, samples, interrupted_dir, seed=43, **options)
            last_path = interrupted_dir / "last.pt"
            self.assertTrue(last_path.is_file())
            self.assertFalse((interrupted_dir / "best.pt").exists())

            with patch("labyrinth_nn.train.train_critic.train_critic", wraps=training.train_critic) as fit:
                run_experiments(configs, self.catalog, samples, target, seeds=seeds, auto_resume=True, **options)
            calls = {(call.args[0].ablation_mode, call.kwargs["seed"]): call.kwargs["resume"]
                     for call in fit.call_args_list}
            self.assertEqual(fit.call_count, 3)
            self.assertEqual(calls, {(first.ablation_mode, 43): last_path, (second.ablation_mode, 42): None,
                                     (second.ablation_mode, 43): None})

            def read_summary(directory):
                summary = json.loads((directory / "ablation_summary.json").read_text(encoding="utf-8"))
                for mode in summary["modes"].values():
                    for run in mode["runs"]:
                        run["checkpoint"] = Path(run["checkpoint"]).relative_to(directory).as_posix()
                return summary

            self.assertEqual(read_summary(clean), read_summary(target))
            self.assertEqual((clean / "ablation_summary.md").read_bytes(), (target / "ablation_summary.md").read_bytes())

            longer = [replace(config, epochs=3) for config in configs]
            with patch("labyrinth_nn.train.train_critic.train_critic", wraps=training.train_critic) as fit:
                summary = run_experiments(longer, self.catalog, samples, target, seeds=seeds, auto_resume=True, **options)
            self.assertEqual(fit.call_count, 4)
            self.assertEqual({(call.args[0].ablation_mode, call.kwargs["seed"]): call.kwargs["resume"]
                              for call in fit.call_args_list},
                             {(config.ablation_mode, seed): target / config.ablation_mode / f"seed_{seed}" / "last.pt"
                              for config in longer for seed in seeds})
            for mode in summary["modes"].values():
                for run in mode["runs"]:
                    self.assertEqual([row["epoch"] for row in run["history"]], [1, 2, 3])

            early_stop_dir = root / "early_stop"
            with patch.object(training, "_selection_nll", return_value=1.0):
                stopped = run_experiments([replace(first, epochs=3)], self.catalog, samples, early_stop_dir,
                                          seeds=[42], patience=1, **options)
            run = stopped["modes"][first.ablation_mode]["runs"][0]
            self.assertEqual([row["epoch"] for row in run["history"]], [1, 2])
            with patch("labyrinth_nn.train.train_critic.train_critic", wraps=training.train_critic) as fit:
                reused = run_experiments([replace(first, epochs=4)], self.catalog, samples, early_stop_dir,
                                         seeds=[42], patience=1, auto_resume=True, **options)
            fit.assert_not_called()
            self.assertEqual(reused, stopped)

    def test_single_resume_does_not_clobber_multi_mode_ablation_summary(self):
        configs, samples = self.experiment_fixture("m0", "m1")
        config = configs[0]
        with tempfile.TemporaryDirectory() as tmp:
            for name, run_configs, seeds in (("multi_mode", configs, (42,)), ("multi_seed", [config], (42, 43))):
                with self.subTest(name=name):
                    root = Path(tmp) / name
                    run_experiments(run_configs, self.catalog, samples, root, seeds=seeds, device="cpu")
                    json_path = root / "ablation_summary.json"
                    markdown_path = root / "ablation_summary.md"
                    json_before, markdown_before = json_path.read_bytes(), markdown_path.read_bytes()
                    run_dir = root / config.ablation_mode / "seed_42"
                    resume = run_dir / "last.pt"
                    longer = replace(config, epochs=2)
                    summary = run_experiments([longer], self.catalog, samples, root, device="cpu", resume=resume)
                    self.assertEqual(json_path.read_bytes(), json_before)
                    self.assertEqual(markdown_path.read_bytes(), markdown_before)
                    self.assertEqual(json.loads((run_dir / "resume_summary.json").read_text(encoding="utf-8")), summary)
                    run = summary["modes"][config.ablation_mode]["runs"][0]
                    self.assertEqual([row["epoch"] for row in run["history"]], [1, 2])
                    with self.assertRaisesRegex(ValueError, "use either --resume or --auto-resume, not both"):
                        run_experiments([longer], self.catalog, samples, root, resume=resume, auto_resume=True)

    def test_cli_epochs_override_and_auto_resume(self):
        configs, samples = self.experiment_fixture("m0", epochs=3)
        config = configs[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            catalog_path = self.catalog.save(root / "catalog.json")
            samples_path = write_samples_jsonl(root / "samples.jsonl", samples)
            out_dir = root / "fit"
            args = ["--config", str(config_path), "--catalog", str(catalog_path), "--samples", str(samples_path),
                    "--out-dir", str(out_dir), "--device", "cpu"]
            summary_path = out_dir / "ablation_summary.json"
            run_dir = out_dir / config.ablation_mode / "seed_42"
            auto_args = [*args, "--epochs", "1", "--auto-resume"]
            with patch("labyrinth_nn.train.train_critic.train_critic", wraps=training.train_critic) as fit:
                training.main(auto_args)
                fit.assert_called_once()
                self.assertEqual(fit.call_args.args[0].epochs, 1)
                self.assertIsNone(fit.call_args.kwargs["resume"])
                fit.reset_mock()

                expected = json.loads(summary_path.read_text(encoding="utf-8"))
                run = expected["modes"][config.ablation_mode]["runs"][0]
                self.assertEqual([row["epoch"] for row in run["history"]], [1])
                metrics_path = run_dir / "test_metrics.json"
                metrics_path.write_text(json.dumps({**run, "seed": 999, "metrics": {}}), encoding="utf-8")
                training.main(auto_args)
                fit.assert_not_called()
                self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), expected)

                with self.assertRaisesRegex(ValueError, "--epochs must be at least 1"):
                    training.main([*args, "--epochs", "0"])
                with self.assertRaisesRegex(ValueError, "use either --resume or --auto-resume, not both"):
                    training.main([*auto_args, "--resume", str(run_dir / "last.pt")])
                fit.assert_not_called()

    @unittest.skipUnless(HAS_TORCH and HAS_ONNX_AND_ORT, "requires PyTorch, NumPy, ONNX and ONNX Runtime")
    def test_export_manifest_replay_inputs_and_runtime_parity(self):
        import onnxruntime as ort

        from labyrinth_nn.export.export_onnx import CASES, export_onnx
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_file = self.catalog.save(root / "catalog.json")
            for mode in ("m0", "m1", "m2a", "m2b"):
                with self.subTest(mode=mode):
                    model = TwoTowerBattleCritic(self.config(mode))
                    model.log_temperature.fill_(0.25)
                    directory = root / mode
                    source = model
                    if mode == "m2b":
                        source = root / "critic.pt"
                        torch.save({"model": model.state_dict(), "config": model.config.to_dict(),
                                    "catalog_sha256": self.catalog.catalog_sha256}, source)
                    result = export_onnx(source, catalog_file, directory)
                    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(result, manifest)
                    self.assertEqual(manifest["format"], "limingjie_twotower_onnx_v1")
                    self.assertEqual(manifest["catalog_sha256"], self.catalog.catalog_sha256)
                    self.assertEqual(manifest["ablation_mode"], mode)
                    self.assertEqual(manifest["validation"]["batch_sizes"], [1, 6])
                    self.assertLess(manifest["validation"]["max_abs_diff"], 1e-4)
                    self.assertAlmostEqual(manifest["tower_b"]["temperature"], math.exp(0.25), places=6)
                    self.assertEqual(set(manifest["files"]), {"tower_a.onnx", "tower_b.onnx", "golden_io.json"})
                    for name, digest in manifest["files"].items():
                        with (directory / name).open("rb") as stream:
                            self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), digest)
                    for tower in ("tower_a", "tower_b"):
                        signature = manifest[tower]
                        session = ort.InferenceSession(str(directory / signature["file"]), sess_options=options,
                                                       providers=["CPUExecutionProvider"])
                        for key, nodes in (("inputs", session.get_inputs()), ("outputs", session.get_outputs())):
                            self.assertEqual(signature[key], [{"name": n.name, "dtype": n.type, "shape": n.shape}
                                                              for n in nodes])
                    golden = json.loads((directory / "golden_io.json").read_text(encoding="utf-8"))
                    cases = {case["name"]: case for case in golden["cases"]}
                    self.assertEqual(tuple(cases), CASES)
                    for case in cases.values():
                        for tower in ("tower_a", "tower_b"):
                            self.assertEqual(list(case[tower]["inputs"]), manifest[tower]["input_names"])
                            self.assertEqual(list(case[tower]["outputs"]), manifest[tower]["output_names"])
                    self.assertFalse(any(cases["empty_relics"]["tower_a"]["inputs"]["relic_mask"][0]))
                    self.assertEqual(sum(cases["underfull_team"]["tower_b"]["inputs"]["team_slot_mask"][0]), 3)
                    widths = cases["tied_search_area_width"]["search_area_widths"]
                    self.assertLess(len(set(widths)), len(widths))
                    self.assertIsNone(cases["gold_unknown"]["sample"]["alpha_rupee"])
                    self.assertEqual(cases["robot_ex"]["sample"]["bucket"], "robot_ex")
                    boss = cases["boss_5f_d5"]["sample"]
                    self.assertEqual((boss["bucket"], boss["area"], boss["difficulty"]), ("boss_5f_d5", 5, 5))


if __name__ == "__main__":
    unittest.main()
