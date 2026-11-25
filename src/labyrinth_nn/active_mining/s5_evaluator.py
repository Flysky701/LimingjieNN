"""S5 sealed panel evaluator, context-clustered CIs, 21-seed confirmation, and desktop ONNX latency bench."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from labyrinth_nn.active_mining.identity import UNSUPPORTED_SIM_TEAM_UNITS, semantic_context_sha256, write_json_atomic
from labyrinth_nn.active_mining.s3_planner import load_calibrated_ensemble
from labyrinth_nn.catalog.schema import FeatureCatalog
from labyrinth_nn.data.dataset import BattleDataset, build_vocab_maps, collate_battle_batch, featurize_sample, mask_gold_sample, team_features
from labyrinth_nn.data.sample_schema import BattleSample, OUTCOME_WIN, read_samples
from labyrinth_nn.train.metrics import EVAL_BUCKETS, evaluate_predictions

DEFAULT_TRAFFIC_WEIGHTS: dict[str, float] = {
    "normal": 0.30,
    "robot_ex": 0.25,
    "special_dual_ex": 0.11,
    "other_ex": 0.12,
    "boss_3f_d5": 0.10,
    "boss_5f_d5": 0.12,
}

BUCKET_AUC_GATES: dict[str, float] = {
    "normal": 0.800,
    "robot_ex": 0.820,
    "special_dual_ex": 0.935,
    "other_ex": 0.915,
    "boss_3f_d5": 0.910,
    "boss_5f_d5": 0.945,
}


def quest_only_baseline_records(samples: Sequence[BattleSample], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Leave-one-context-out / empirical quest-bucket win-rate baseline matching panel records."""
    contexts = [semantic_context_sha256(s) for s in samples]
    wins: Counter[tuple[int | str, str | None]] = Counter()
    counts: Counter[tuple[int | str, str | None]] = Counter()
    for s, ctx in zip(samples, contexts, strict=True):
        total = sum(s.outcome_counts)
        if total <= 0:
            continue
        win_rate = s.outcome_counts[OUTCOME_WIN] / total
        for group in (s.quest_id, s.bucket):
            for key in ((group, None), (group, ctx)):
                wins[key] += win_rate
                counts[key] += 1
    out = []
    for s, rec, ctx in zip(samples, records, contexts, strict=True):
        p = 0.5
        for group in (s.quest_id, s.bucket):
            count = counts[(group, None)] - counts[(group, ctx)]
            if count:
                p = (wins[(group, None)] - wins[(group, ctx)]) / count
                break
        rem = 1.0 - p
        out.append({**rec, "pred_probs": [p, rem * 0.5, rem * 0.5]})
    return out


def reweighted_probability_metrics(
    records: Sequence[Mapping[str, Any]],
    traffic_weights: Mapping[str, float] = DEFAULT_TRAFFIC_WEIGHTS,
) -> dict[str, float]:
    by_bucket: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        by_bucket[str(r["bucket"])].append(r)
    active = {b: w for b, w in traffic_weights.items() if by_bucket.get(b)}
    norm = sum(active.values()) or 1.0
    brier_3 = brier_w = ece = acc = 0.0
    for b, w in active.items():
        bw = w / norm
        sub = evaluate_predictions(by_bucket[b])["global"]
        brier_3 += bw * float(sub["brier_3class"])
        brier_w += bw * float(sub["brier_win"])
        ece += bw * float(sub["ece_15"])
        acc += bw * float(sub["accuracy_3class"])
    return {
        "reweighted_brier_3class": brier_3,
        "reweighted_brier_win": brier_w,
        "reweighted_ece_15": ece,
        "reweighted_accuracy_3class": acc,
    }


def one_sided_wilson_upper_95(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    z = 1.6448536269514722  # 95% one-sided normal quantile
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    rad = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return min(1.0, (center + rad) / denom)


def context_clustered_bootstrap_ci(
    context_values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 20261155,
) -> dict[str, float]:
    n = len(context_values)
    if n == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "upper95_one_sided": 0.0}
    mean_val = statistics.fmean(context_values)
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(context_values, k=n)) for _ in range(n_boot))
    return {
        "mean": mean_val,
        "ci95_low": boots[int(0.025 * (n_boot - 1))],
        "ci95_high": boots[int(0.975 * (n_boot - 1))],
        "upper95_one_sided": boots[int(0.95 * (n_boot - 1))],
    }


def evaluate_paired_recommendations(
    paired_samples: Sequence[BattleSample],
    paired_records: Sequence[Mapping[str, Any]],
    *,
    win_threshold: float = 0.5,
    confirmed_21_outcomes: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    by_ctx: dict[str, list[tuple[BattleSample, Mapping[str, Any]]]] = defaultdict(list)
    for s, r in zip(paired_samples, paired_records, strict=True):
        ctx = semantic_context_sha256(s)
        by_ctx[ctx].append((s, r))

    all_top1_fails: list[float] = []
    feasible_top1_fails: list[float] = []
    no_feasible_count = 0
    covered_count = 0
    for ctx, items in sorted(by_ctx.items()):
        # Sort by predicted win probability desc, deterministic team_sig tiebreak.
        chosen_sample, chosen_rec = min(
            items,
            key=lambda pair: (-float(pair[1]["pred_probs"][0]), pair[0].canonical_team_sig()),
        )
        # Observed win indicator from 21-seed confirmation if provided, else panel outcome.
        if confirmed_21_outcomes and chosen_sample.sample_id in confirmed_21_outcomes:
            obs_win = float(confirmed_21_outcomes[chosen_sample.sample_id])
        else:
            tot = sum(chosen_sample.outcome_counts)
            obs_win = chosen_sample.outcome_counts[OUTCOME_WIN] / tot if tot else 0.0
        fail = 1.0 if obs_win < 0.5 else 0.0
        all_top1_fails.append(fail)
        has_feasible = any(float(r["pred_probs"][0]) >= win_threshold for _, r in items)
        if has_feasible:
            covered_count += 1
            feasible_top1_fails.append(fail)
        else:
            no_feasible_count += 1

    total_ctx = len(by_ctx)
    feas_n = len(feasible_top1_fails)
    feas_k = int(sum(feasible_top1_fails))
    all_k = int(sum(all_top1_fails))
    return {
        "total_contexts": total_ctx,
        "covered_contexts": covered_count,
        "no_feasible_contexts": no_feasible_count,
        "coverage": covered_count / total_ctx if total_ctx else 0.0,
        "unfiltered_top1_failures": all_k,
        "unfiltered_top1_failure_rate": all_k / total_ctx if total_ctx else 0.0,
        "unfiltered_ci": context_clustered_bootstrap_ci(all_top1_fails),
        "feasible_top1_failures": feas_k,
        "feasible_contexts": feas_n,
        "feasible_top1_failure_rate": feas_k / feas_n if feas_n else 0.0,
        "feasible_wilson_upper95": one_sided_wilson_upper_95(feas_k, feas_n),
        "feasible_ci": context_clustered_bootstrap_ci(feasible_top1_fails),
    }


def benchmark_onnx_desktop_latency(
    bundle_dir: str | Path,
    catalog: FeatureCatalog,
    samples: Sequence[BattleSample],
    *,
    num_contexts: int = 200,
    num_threads: int = 1,
) -> dict[str, Any]:
    """Measure cold start and warm CPU p50/p95 for Tower A, 64-team Tower B batch, and 512-candidate rec."""
    import numpy as np
    import onnxruntime as ort

    root = Path(bundle_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False  # Matches LimingjieDesktop deployment guidance.

    t_cold0 = time.perf_counter()
    sess_a = ort.InferenceSession(str(root / "tower_a.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(root / "tower_b.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    cold_load_ms = (time.perf_counter() - t_cold0) * 1000.0

    a_names = list(manifest["tower_a"]["input_names"])
    a_outs = list(manifest["tower_a"]["output_names"])
    b_names = list(manifest["tower_b"]["input_names"])
    b_outs = list(manifest["tower_b"]["output_names"])
    roster_max_len = int(manifest["dimensions"]["roster_max_len"])
    relic_max_len = int(manifest["dimensions"]["relic_max_len"])
    maps = build_vocab_maps(catalog)

    # Deduplicate contexts from samples up to num_contexts.
    by_ctx: dict[str, BattleSample] = {}
    for s in samples:
        c = semantic_context_sha256(s)
        if c not in by_ctx:
            by_ctx[c] = s
        if len(by_ctx) >= num_contexts:
            break
    ctx_samples = list(by_ctx.values())
    if not ctx_samples:
        raise ValueError("latency benchmark requires at least 1 context sample")

    # Warmup
    warm_feat = featurize_sample(catalog, ctx_samples[0], **maps, roster_max_len=roster_max_len, relic_max_len=relic_max_len)
    warm_batch = collate_battle_batch([warm_feat])
    warm_a = dict(zip(a_outs, sess_a.run(a_outs, {n: warm_batch[n].numpy() for n in a_names})))
    b_in_warm = {**{n: warm_batch[n].numpy() for n in b_names if n in warm_batch}, **{n: warm_a[n] for n in b_names if n in warm_a}}
    sess_b.run(b_outs, b_in_warm)

    tower_a_ms: list[float] = []
    tower_b_64_ms: list[float] = []
    full_512_ms: list[float] = []

    for s in ctx_samples:
        # 1) Full context featurize + Tower A
        t0 = time.perf_counter()
        feat = featurize_sample(catalog, s, **maps, roster_max_len=roster_max_len, relic_max_len=relic_max_len)
        batch_a = collate_battle_batch([feat])
        ta0 = time.perf_counter()
        out_a = dict(zip(a_outs, sess_a.run(a_outs, {n: batch_a[n].numpy() for n in a_names})))
        ta1 = time.perf_counter()
        tower_a_ms.append((ta1 - ta0) * 1000.0)

        # Draw candidates from playable units retained in the encoded roster.
        index = {uid: i for i, uid in enumerate(feat["roster_raw_ids"]) if uid}
        owned = [uid for uid in index if uid in catalog.units and uid not in UNSUPPORTED_SIM_TEAM_UNITS]
        gold_norm = feat["global_feats"][catalog.global_feat_names.index("alpha_rupee_norm")]
        owned_norm = feat["global_feats"][catalog.global_feat_names.index("owned_unit_count_25_norm")]
        rng = random.Random(s.sample_id)
        teams_512 = [
            sorted(rng.sample(owned, min(5, len(owned))), key=lambda u: (catalog.units[u].search_area_width, u))
            for _ in range(512)
        ]

        # Score 512 teams in 64-team Tower B chunks, measuring both single 64-batch and end-to-end 512 rec.
        for chunk_idx in range(0, 512, 64):
            chunk_teams = teams_512[chunk_idx : chunk_idx + 64]
            rows = []
            for t in chunk_teams:
                r = team_features(catalog, t, s.relic_ids, gold_norm, owned_norm)
                r["team_roster_pos"] = [index.get(u, 0) for u in r["team_unit_ids_sorted"]]
                rows.append(r)
            tb = collate_battle_batch(rows)
            n_items = len(rows)
            roster_ctx = np.broadcast_to(out_a["roster_ctx"], (n_items, *out_a["roster_ctx"].shape[1:])).copy()
            ctx_sum = np.broadcast_to(out_a["context_summary"], (n_items, *out_a["context_summary"].shape[1:])).copy()
            feed_b = {
                "roster_ctx": roster_ctx,
                "context_summary": ctx_sum,
                "team_roster_pos": tb["team_roster_pos"].numpy(),
                "team_slot_mask": tb["team_slot_mask"].numpy(),
            }
            if "team_pos_matrix" in b_names:
                feed_b["team_pos_matrix"] = tb["team_pos_matrix"].numpy()
            if "team_synergy" in b_names:
                feed_b["team_synergy"] = tb["team_synergy"].numpy()
            tb0 = time.perf_counter()
            sess_b.run(b_outs, feed_b)
            tb1 = time.perf_counter()
            if chunk_idx == 0:
                tower_b_64_ms.append((tb1 - tb0) * 1000.0)
        t1 = time.perf_counter()
        full_512_ms.append((t1 - t0) * 1000.0)

    def pct(values: list[float], q: float) -> float:
        svals = sorted(values)
        return svals[min(len(svals) - 1, int(q * (len(svals) - 1)))]

    return {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "intra_op_threads": num_threads,
            "onnxruntime_version": ort.__version__,
        },
        "cold_load_ms": cold_load_ms,
        "contexts_measured": len(ctx_samples),
        "tower_a_p50_ms": pct(tower_a_ms, 0.50),
        "tower_a_p95_ms": pct(tower_a_ms, 0.95),
        "tower_b_64_p50_ms": pct(tower_b_64_ms, 0.50),
        "tower_b_64_p95_ms": pct(tower_b_64_ms, 0.95),
        "full_512_p50_ms": pct(full_512_ms, 0.50),
        "full_512_p95_ms": pct(full_512_ms, 0.95),
        "gates_passed": (
            pct(tower_a_ms, 0.95) <= 25.0
            and pct(tower_b_64_ms, 0.95) <= 30.0
            and pct(full_512_ms, 0.95) <= 500.0
        ),
    }


def evaluate_s5_sealed_release(
    *,
    checkpoint_paths: Sequence[str | Path],
    catalog: FeatureCatalog,
    stratified_samples: Sequence[BattleSample],
    paired_samples: Sequence[BattleSample],
    export_dir: str | Path | None = None,
    latency_samples: Sequence[BattleSample] | None = None,
    confirmed_21_outcomes: Mapping[str, float] | None = None,
    device: str = "cpu",
    batch_size: int = 512,
) -> dict[str, Any]:
    """Run complete S5 sealed evaluation against TASK.md release gates."""
    from torch.utils.data import DataLoader

    from labyrinth_nn.train.train_critic import _ensemble_records, predict_records

    ensemble, ckpt_meta = load_calibrated_ensemble(checkpoint_paths, catalog, device=device)

    def score_models(rows: Sequence[BattleSample]) -> list[dict[str, Any]]:
        per_model = []
        for model in ensemble.models:
            ds = BattleDataset(
                catalog,
                rows,
                training=False,
                cache_eval=False,
                roster_max_len=model.config.roster_max_len,
                relic_max_len=model.config.relic_max_len,
            )
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_battle_batch)
            per_model.append(predict_records(model, loader, device))
        return per_model[0] if len(per_model) == 1 else _ensemble_records(per_model)

    strat_recs = score_models(stratified_samples)
    strat_masked_recs = score_models([mask_gold_sample(s) for s in stratified_samples])
    quest_recs = quest_only_baseline_records(stratified_samples, strat_recs)

    strat_metrics = evaluate_predictions(strat_recs)
    strat_masked_metrics = evaluate_predictions(strat_masked_recs)
    quest_metrics = evaluate_predictions(quest_recs)
    reweighted = reweighted_probability_metrics(strat_recs)

    # Check bucket counts & AUCs
    bucket_pos_neg = {
        b: {
            "pos": sum(1 for s in stratified_samples if s.bucket == b and sum(s.outcome_counts) and s.outcome_counts[0] / sum(s.outcome_counts) >= 0.5),
            "neg": sum(1 for s in stratified_samples if s.bucket == b and sum(s.outcome_counts) and s.outcome_counts[0] / sum(s.outcome_counts) < 0.5),
        }
        for b in EVAL_BUCKETS
    }

    paired_recs = score_models(paired_samples)
    rec_risk = evaluate_paired_recommendations(
        paired_samples, paired_recs, confirmed_21_outcomes=confirmed_21_outcomes
    )

    export_report = None
    latency_report = None
    if export_dir is not None:
        from labyrinth_nn.export.export_onnx import export_onnx

        if len(checkpoint_paths) != 1:
            raise ValueError("ONNX export accepts a single deployable checkpoint path")
        out_root = Path(export_dir)
        manifest = export_onnx(checkpoint_paths[0], catalog, out_root)
        total_bytes = sum((out_root / name).stat().st_size for name in ("tower_a.onnx", "tower_b.onnx"))
        export_report = {
            "total_onnx_bytes": total_bytes,
            "size_gate_passed": total_bytes < 10_000_000,
            "max_abs_diff": manifest["validation"]["max_abs_diff"],
            "parity_gate_passed": manifest["validation"]["max_abs_diff"] < 1e-4,
        }
        bench_rows = list(latency_samples) if latency_samples is not None else list(stratified_samples)
        latency_report = benchmark_onnx_desktop_latency(out_root, catalog, bench_rows, num_contexts=min(200, len(bench_rows)))

    g = strat_metrics["global"]
    mg = strat_masked_metrics["global"]
    gates = {
        "global_auroc_ge_0950": (g["auroc"] or 0.0) >= 0.950,
        "bucket_auroc_gates": all(
            (strat_metrics["buckets"][b]["auroc"] or 0.0) >= thresh for b, thresh in BUCKET_AUC_GATES.items()
        ),
        "normal_beats_quest_002": (strat_metrics["buckets"]["normal"]["auroc"] or 0.0)
        - (quest_metrics["buckets"]["normal"]["auroc"] or 0.0)
        >= 0.02,
        "robot_beats_quest_002": (strat_metrics["buckets"]["robot_ex"]["auroc"] or 0.0)
        - (quest_metrics["buckets"]["robot_ex"]["auroc"] or 0.0)
        >= 0.02,
        "reweighted_brier3_le_0210": reweighted["reweighted_brier_3class"] <= 0.210,
        "brier_win_le_0074": (g["brier_win"] or 1.0) <= 0.074,
        "global_ece_le_0030": (g["ece_15"] or 1.0) <= 0.030,
        "bucket_ece_le_0055": all((strat_metrics["buckets"][b]["ece_15"] or 0.0) <= 0.055 for b in EVAL_BUCKETS),
        "accuracy_3class_ge_0845": (g["accuracy_3class"] or 0.0) >= 0.845,
        "masked_gold_ece_le_0045": (mg["ece_15"] or 1.0) <= 0.045,
        "masked_gold_brier_delta_le_0010": ((mg["brier_win"] or 0.0) - (g["brier_win"] or 0.0)) <= 0.010,
        "damage_mae_le_0185": (g["loss_damage_mae"] or 1.0) <= 0.185,
        "time_mae_le_0125": (g["win_time_mae"] or 1.0) <= 0.125,
        "rec_feasible_top1_le_003": rec_risk["feasible_top1_failure_rate"] <= 0.03,
        "rec_feasible_wilson_upper95_le_005": rec_risk["feasible_wilson_upper95"] <= 0.05,
        "rec_coverage_ge_080": rec_risk["coverage"] >= 0.80,
    }
    if export_report is not None:
        gates["onnx_export_passed"] = export_report["size_gate_passed"] and export_report["parity_gate_passed"]
    if latency_report is not None:
        gates["desktop_latency_passed"] = latency_report["gates_passed"]

    return {
        "checkpoints": ckpt_meta,
        "damage_label_policy": "truncated_aggregate_damage_proxy",
        "stratified_metrics": strat_metrics,
        "stratified_masked_gold": strat_masked_metrics,
        "quest_only_baseline": quest_metrics,
        "reweighted_probability_metrics": reweighted,
        "bucket_pos_neg_counts": bucket_pos_neg,
        "recommendation_risk": rec_risk,
        "export_report": export_report,
        "latency_report": latency_report,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--stratified-samples", nargs="+", type=Path, required=True)
    parser.add_argument("--paired-samples", nargs="+", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    catalog = FeatureCatalog.load(args.catalog)
    strat = [s for p in args.stratified_samples for s in read_samples(p)]
    paired = [s for p in args.paired_samples for s in read_samples(p)]
    report = evaluate_s5_sealed_release(
        checkpoint_paths=args.checkpoints,
        catalog=catalog,
        stratified_samples=strat,
        paired_samples=paired,
        export_dir=args.export_dir,
        device=args.device,
    )
    write_json_atomic(args.out, report, indent=2)
    print(json.dumps({"all_gates_passed": report["all_gates_passed"], "gates": report["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
