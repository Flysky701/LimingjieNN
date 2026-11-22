"""Export both towers and publish replayable fixtures after CPU Runtime parity passes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from labyrinth_nn.catalog.schema import FeatureCatalog, QuestEntry
from labyrinth_nn.data.dataset import build_vocab_maps, collate_battle_batch, featurize_sample
from labyrinth_nn.data.sample_schema import NUM_OUTCOME_CLASSES, OUTCOME_NAMES, BattleSample
from labyrinth_nn.models.critic import CriticConfig, TowerAOnnxWrapper, TowerBOnnxWrapper, TwoTowerBattleCritic

CASES = ("empty_relics", "underfull_team", "tied_search_area_width", "gold_unknown", "robot_ex", "boss_5f_d5")
TOLERANCE = 1e-4


def _golden_cases(catalog, config):
    ids = catalog.sorted_unit_ids()
    if min(len(ids), config.roster_max_len) < 2:
        raise ValueError("the tied-position golden case needs at least two roster slots and two catalog units")
    maps = build_vocab_maps(catalog)
    by_width = defaultdict(list)
    for uid in ids:
        by_width[catalog.units[uid].search_area_width].append(uid)
    tied = next((group[:2] for group in by_width.values() if len(group) >= 2), None)
    quests = sorted(catalog.quests.values(), key=lambda q: q.quest_id)
    results = []
    for name in CASES:
        fixture_catalog = catalog
        overrides = {}
        team = ids[:min(5, config.roster_max_len)]
        if name == "underfull_team":
            team = team[:3]
        if name == "tied_search_area_width":
            pair = tied or ids[:2]
            team = (pair[::-1] + [uid for uid in ids if uid not in pair])[:len(team)]
            if tied is None:
                # Preserve vocabulary IDs. Record the isolated feature override for exact replay.
                first, second = (catalog.units[uid] for uid in pair)
                static = list(second.static_feats)
                if static:
                    static[0] = first.search_area_width / 1000.0
                changed = replace(second, search_area_width=first.search_area_width, static_feats=static)
                fixture_catalog = replace(catalog, units={**catalog.units, second.unit_id: changed})
                overrides["unit_search_area_width"] = {str(second.unit_id): first.search_area_width}
        bucket = name if name in ("robot_ex", "boss_5f_d5") else "normal"
        quest = next((q for q in quests if q.bucket == bucket), None)
        if quest is None:
            # An unseen quest uses vocabulary index zero just as it would in production.
            qid = min([0, *catalog.quests]) - 1
            enemy_ids = [eid for eid in catalog.sorted_enemy_ids()
                         if bucket != "boss_5f_d5" or catalog.enemies[eid].is_boss_unit][:5]
            quest = QuestEntry(qid, 0, 0, 5 if bucket == "boss_5f_d5" else 1, 90, 0, enemy_ids, 0,
                               f"golden-{bucket}", bucket)
            fixture_catalog = replace(fixture_catalog, quests={**fixture_catalog.quests, qid: quest})
            overrides["synthetic_quest"] = {"quest_id": qid, "enemy_ids": enemy_ids, "bucket": bucket}
        relics = [] if name == "empty_relics" else catalog.sorted_relic_ids()[:config.relic_max_len]
        owned = list(dict.fromkeys(team + ids))[:config.roster_max_len]
        roles = [sum(catalog.units[u].role_id == role for u in owned) for role in range(1, 9)]
        talents = [sum(catalog.units[u].talent_id == talent for u in owned) for talent in range(1, 6)]
        sample = BattleSample(
            sample_id=name, round_id="onnx_golden", source="sim", db_version=catalog.db_version,
            quest_id=quest.quest_id, canonical_wave_sig=quest.canonical_wave_sig, bucket=bucket,
            area=5 if name == "boss_5f_d5" else quest.area, difficulty=5 if name == "boss_5f_d5" else 1,
            guild_id=0, team_unit_ids=team, owned_unit_ids=owned, owned_unit_count=len(owned),
            owned_role_counts=roles, owned_talent_counts=talents, relic_ids=relics,
            mark_stacks=catalog.mark_stacks_for_relics(relics), alpha_rupee=None if name == "gold_unknown" else 6400,
        )
        row = featurize_sample(fixture_catalog, sample, **maps, roster_max_len=config.roster_max_len,
                               relic_max_len=config.relic_max_len)
        results.append((row, {"name": name, "sample": sample.to_dict(), "overrides": overrides,
                              "team_unit_ids_sorted": row["team_unit_ids_sorted"],
                              "search_area_widths": [fixture_catalog.units[u].search_area_width
                                                     for u in row["team_unit_ids_sorted"] if u]}))
    return results


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _export_graph(wrapper, tensors, path):
    names, outputs = list(wrapper.input_names), list(wrapper.output_names)
    options = {}
    parameters = inspect.signature(torch.onnx.export).parameters
    if "dynamo" in parameters:
        options["dynamo"] = False
    if "external_data" in parameters:
        options["external_data"] = False
    torch.onnx.export(wrapper, tuple(tensors[n] for n in names), str(path), input_names=names,
                      output_names=outputs, opset_version=17, dynamic_axes={n: {0: "batch"} for n in names + outputs},
                      **options)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for actual, expected in ((session.get_inputs(), names), (session.get_outputs(), outputs)):
        if [node.name for node in actual] != expected or any(node.shape[0] != "batch" for node in actual):
            raise RuntimeError(f"{path.name} has an unexpected input/output signature or a static batch dimension")
    return session


def _compare(label, expected, actual):
    differences = {}
    for name, value in expected.items():
        reference = value.detach().cpu().numpy()
        result = actual[name]
        if result.shape != reference.shape or not (np.isfinite(reference).all() and np.isfinite(result).all()):
            raise ValueError(f"{label}/{name}: inconsistent shapes or non-finite output")
        diff = float(np.max(np.abs(result - reference)))
        if diff >= TOLERANCE:
            raise ValueError(f"{label}/{name}: max_abs_diff={diff:.9g} (limit {TOLERANCE})")
        differences[name] = diff
    return differences


@torch.no_grad()
def export_onnx(model: TwoTowerBattleCritic | str | Path, catalog: FeatureCatalog | str | Path,
                out_dir: str | Path) -> dict:
    if not isinstance(catalog, FeatureCatalog):
        catalog = FeatureCatalog.load(catalog)
    catalog_hash = catalog.catalog_sha256 or catalog.compute_sha256()
    if isinstance(model, (str, Path)):
        checkpoint = torch.load(model, map_location="cpu", weights_only=True)
        if checkpoint.get("catalog_sha256", catalog_hash) != catalog_hash:
            raise ValueError("checkpoint and export catalog hashes differ")
        model = TwoTowerBattleCritic(CriticConfig.from_dict(checkpoint["config"], catalog))
        model.load_state_dict(checkpoint["model"])
        del checkpoint
    else:
        CriticConfig.from_dict(model.config.to_dict(), catalog)
        model = copy.deepcopy(model)
    model = model.cpu().float().eval()
    a, b = TowerAOnnxWrapper(model).eval(), TowerBOnnxWrapper(model).eval()
    cases = _golden_cases(catalog, model.config)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as staging:
        stage = Path(staging)
        example = collate_battle_batch([cases[0][0]])
        a_values = dict(zip(a.output_names, a(*(example[n] for n in a.input_names))))
        a_session = _export_graph(a, example, stage / "tower_a.onnx")
        b_session = _export_graph(b, {**example, **a_values}, stage / "tower_b.onnx")

        def verify(rows, label, record=None):
            batch = collate_battle_batch(rows)
            a_inputs = {n: batch[n] for n in a.input_names}
            pt_a = dict(zip(a.output_names, a(*a_inputs.values())))
            a_feed = {n: x.numpy() for n, x in a_inputs.items()}
            ort_a = dict(zip(a.output_names, a_session.run(list(a.output_names), a_feed)))
            b_values = {**batch, **pt_a}
            b_inputs = {n: b_values[n] for n in b.input_names}
            pt_b = dict(zip(b.output_names, b(*b_inputs.values())))
            # Chain actual ORT activations into B so parity covers the deployment path as well.
            b_feed = {n: ort_a[n] if n in ort_a else x.numpy() for n, x in b_inputs.items()}
            ort_b = dict(zip(b.output_names, b_session.run(list(b.output_names), b_feed)))
            if record is not None:
                for key, inputs, outputs in (("tower_a", a_inputs, pt_a), ("tower_b", b_inputs, pt_b)):
                    record[key] = {"inputs": {n: x.tolist() for n, x in inputs.items()},
                                   "outputs": {n: x.tolist() for n, x in outputs.items()}}
            return {**_compare(f"{label}/tower_a", pt_a, ort_a), **_compare(f"{label}/tower_b", pt_b, ort_b)}

        checks = {case["name"]: verify([row], case["name"], case) for row, case in cases}
        checks["batch_6"] = verify([row for row, _ in cases], "batch_6")
        validation = {"batch_sizes": [1, len(cases)], "max_abs_diff": max(max(d.values()) for d in checks.values()),
                      "tolerance": TOLERANCE, "max_abs_diff_by_output": checks}
        golden = {"cases": [case for _, case in cases], "validation": validation}
        (stage / "golden_io.json").write_text(json.dumps(golden, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        files = {n: _sha256(stage / n) for n in ("tower_a.onnx", "tower_b.onnx", "golden_io.json")}
        config = model.config.to_dict()
        dims = ("num_units", "num_enemies", "num_quests", "num_relics", "unit_feat_dim", "enemy_feat_dim",
                "global_feat_dim", "team_synergy_dim", "team_pos_channels", "d_model", "n_heads",
                "relic_max_len", "roster_max_len", "max_areas", "max_difficulties")
        dimensions = {k: config[k] for k in dims}
        dimensions.update(team_slots=5, enemy_slots=5, num_outcome_classes=NUM_OUTCOME_CLASSES)
        manifest = {"format": "limingjie_twotower_onnx_v1", "ablation_mode": config["ablation_mode"],
                    "catalog_sha256": catalog_hash, "opset_version": 17, "files": files,
                    "dimensions": dimensions, "outcome_names": list(OUTCOME_NAMES),
                    "config": config, "validation": validation}
        for key, wrapper, session in (("tower_a", a, a_session), ("tower_b", b, b_session)):
            filename = f"{key}.onnx"
            manifest[key] = {"file": filename, "sha256": files[filename],
                             "input_names": list(wrapper.input_names), "output_names": list(wrapper.output_names),
                             "inputs": [{"name": node.name, "dtype": node.type, "shape": node.shape}
                                        for node in session.get_inputs()],
                             "outputs": [{"name": node.name, "dtype": node.type, "shape": node.shape}
                                         for node in session.get_outputs()]}
        manifest["tower_b"]["temperature"] = model.log_temperature.clamp(-4.0, 4.0).exp().item()
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
                                             encoding="utf-8")
        for name in (*files, "manifest.json"):
            (stage / name).replace(root / name)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = export_onnx(args.checkpoint, args.catalog, args.out_dir)
    print(f"{args.out_dir / 'manifest.json'}  max_abs_diff={manifest['validation']['max_abs_diff']:.9g}")


if __name__ == "__main__":
    main()
