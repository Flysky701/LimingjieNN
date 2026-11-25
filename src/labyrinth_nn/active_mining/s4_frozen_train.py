"""S4 frozen-holdout trainer, S2 holdout protection, and val_select distillation gate."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from pickle import UnpicklingError
from typing import TYPE_CHECKING, Any

from labyrinth_nn.active_mining.identity import (
    ALL_FOLDS,
    HOLDOUT_FOLDS,
    S2HoldoutRegistry,
    semantic_context_sha256,
    split_digest_for_parts,
)
from labyrinth_nn.active_mining.s5_reservation import S5Denylist
from labyrinth_nn.catalog.schema import FeatureCatalog
from labyrinth_nn.data.dataset import BattleDataset, collate_battle_batch, mask_gold_sample
from labyrinth_nn.data.sample_schema import BattleSample, OUTCOME_WIN, read_samples
from labyrinth_nn.train.metrics import evaluate_predictions

if TYPE_CHECKING:
    import torch

    from labyrinth_nn.models.critic import CriticConfig


def partition_frozen_samples(
    samples: Sequence[BattleSample],
    s2_registry: S2HoldoutRegistry,
    s5_denylist: S5Denylist,
    *,
    catalog: FeatureCatalog | None = None,
) -> dict[str, list[BattleSample]]:
    """Freeze S2 val_select/val_calib/test exactly as in S2; admit S3+ samples into train only."""
    seen_ids: set[str] = set()
    versions: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        versions.add(sample.db_version)
    if len(versions) > 1:
        raise ValueError(f"mixed db_version values: {', '.join(map(repr, sorted(versions)))}")
    if s2_registry.db_version != s5_denylist.db_version:
        raise ValueError("S2 registry and S5 denylist db_version mismatch")
    if s2_registry.catalog_sha256 != s5_denylist.catalog_sha256:
        raise ValueError("S2 registry and S5 denylist catalog_sha256 mismatch")
    if catalog is not None:
        cat_sha = catalog.catalog_sha256 or catalog.compute_sha256()
        if cat_sha != s2_registry.catalog_sha256 or catalog.db_version != s2_registry.db_version:
            raise ValueError("catalog differs from frozen S2 registry")

    by_id: dict[str, BattleSample] = {}
    s3_train: list[BattleSample] = []
    for sample in samples:
        if sample.db_version != s2_registry.db_version:
            raise ValueError(f"sample {sample.sample_id} db_version mismatch")
        if s5_denylist.is_denied(sample):
            raise ValueError(f"sample {sample.sample_id} hits S5 reserved denylist")
        if sample.sample_id in s2_registry.sample_folds:
            by_id[sample.sample_id] = sample
        else:
            if s2_registry.is_holdout_sample(sample):
                raise ValueError(f"new sample {sample.sample_id} collides with frozen S2 holdout context")
            if sum(sample.outcome_counts) > 0:
                s3_train.append(sample)

    parts: dict[str, list[BattleSample]] = {fold: [] for fold in ALL_FOLDS}
    missing_s2 = [sid for sid in s2_registry.sample_folds if sid not in by_id]
    if missing_s2:
        raise ValueError(f"missing {len(missing_s2)} frozen S2 samples (first: {missing_s2[0]})")

    if s2_registry.fold_orders:
        for fold in ALL_FOLDS:
            parts[fold] = [by_id[sid] for sid in s2_registry.fold_orders[fold]]
        if split_digest_for_parts(parts) != s2_registry.split_sha256:
            raise ValueError("reconstructed S2 folds do not match frozen split_sha256")
    else:
        for sid, fold in sorted(s2_registry.sample_folds.items()):
            parts[fold].append(by_id[sid])

    # Verify holdout contexts never appear inside S2 train or S3 train.
    holdout_ctx_set = set(s2_registry.holdout_semantic_contexts)
    for fold in HOLDOUT_FOLDS:
        for s in parts[fold]:
            holdout_ctx_set.add(semantic_context_sha256(s))
    for s in s3_train:
        if semantic_context_sha256(s) in holdout_ctx_set:
            raise ValueError(f"new training sample {s.sample_id} shares semantic context with S2 holdout")

    s3_train.sort(key=lambda s: (s.sample_id, s.physical_state_key()))
    parts["train"].extend(s3_train)
    return parts


def _frozen_split_digest(parts: dict[str, list[BattleSample]]) -> str:
    return split_digest_for_parts(parts)


def train_frozen_single(
    config: CriticConfig,
    catalog: FeatureCatalog,
    parts: dict[str, list[BattleSample]],
    s2_registry: S2HoldoutRegistry,
    s5_denylist: S5Denylist,
    out_dir: Path,
    *,
    seed: int = 42,
    device: str | torch.device | None = None,
    amp: bool = False,
    resume: Path | None = None,
    patience: int = 5,
    num_workers: int = 0,
    pin_memory: bool = False,
    teacher_soft_probs: dict[str, list[float]] | None = None,
    distill_alpha: float = 0.0,
) -> dict[str, Any]:
    """Train one critic on frozen S2+S3 folds without re-splitting S2 holdouts."""
    import torch

    from labyrinth_nn.models.critic import TwoTowerBattleCritic
    from labyrinth_nn.train.losses import batch_to_device, compute_critic_loss, fit_temperature
    from labyrinth_nn.train.train_critic import (
        LOSS_COMPONENTS,
        _counterfactual_metrics,
        _cosine_scheduler,
        _make_loader,
        _same_training_config,
        _save_checkpoint,
        _selection_nll,
        _write_json,
        predict_records,
    )

    if not 0.0 <= distill_alpha <= 1.0:
        raise ValueError("distill_alpha must be in [0, 1]")
    if teacher_soft_probs is not None and distill_alpha > 0:
        train_rows = []
        for s in parts["train"]:
            probs = teacher_soft_probs.get(s.sample_id)
            if probs is None:
                train_rows.append(s)
                continue
            total = sum(s.outcome_counts)
            hard = [c / total for c in s.outcome_counts]
            blended = [(1.0 - distill_alpha) * h + distill_alpha * p for h, p in zip(hard, probs, strict=True)]
            # Preserve total outcome count scale as integer-free surrogate via 1000-scaled soft counts.
            scaled_counts = [max(0, round(v * 1000)) for v in blended]
            if not sum(scaled_counts):
                scaled_counts = [0, 0, 1000]
            train_rows.append(replace(s, outcome_counts=scaled_counts))
        parts = {**parts, "train": train_rows}

    split_digest = _frozen_split_digest(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_hash = catalog.catalog_sha256 or catalog.compute_sha256()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume is not None else {}

    torch.manual_seed(seed)
    shuffle = torch.Generator()
    loader_opts = dict(seed=seed, num_workers=num_workers, pin_memory=pin_memory)
    loaders = {
        name: _make_loader(
            config,
            catalog,
            rows,
            shuffle=shuffle if name == "train" else None,
            cache_eval=name != "train",
            **loader_opts,
        )
        for name, rows in parts.items()
    }
    model = TwoTowerBattleCritic(config).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    schedule_steps = state.get("schedule_steps", len(loaders["train"]) * config.epochs)
    scheduler = _cosine_scheduler(optimizer, config, schedule_steps) if config.lr_schedule == "cosine" else None
    scale_grads = amp and dev.type == "cuda"
    scaler_cls = getattr(torch.amp, "GradScaler", None)
    scaler = (
        scaler_cls("cuda", enabled=scale_grads)
        if scaler_cls is not None
        else torch.cuda.amp.GradScaler(enabled=scale_grads)
    )

    best_state = None
    best_score = math.inf
    best_epoch = start_epoch = stale = 0
    history: list[dict[str, Any]] = []
    if resume is not None:
        if (
            state.get("catalog_sha256"),
            state.get("seed"),
            state.get("split_sha256"),
            state.get("s2_split_sha256"),
            state.get("s5_denylist_sha256"),
        ) != (catalog_hash, seed, split_digest, s2_registry.split_sha256, s5_denylist.denylist_sha256):
            raise ValueError("resume checkpoint metadata differs from frozen S4 configuration")
        if not _same_training_config(state["config"], config, catalog):
            raise ValueError("resume checkpoint training config differs")
        model.load_state_dict(state.get("resume_model", state["model"]))
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scale_grads and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["cpu_rng"])
        if dev.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], dev)
        start_epoch = int(state["epoch"])
        best_epoch = int(state["best_epoch"])
        best_score = float(state["best_score"])
        best_state = state["best_model"]
        stale = int(state["stale"])
        history = list(state["history"])

    last_path = out_dir / "last.pt"
    loss_dtype = torch.float32 if dev.type == "mps" else torch.float64
    from contextlib import nullcontext

    for epoch in range(start_epoch + 1, config.epochs + 1):
        if stale >= patience:
            break
        model.train()
        shuffle.manual_seed(seed + epoch)
        loaders["train"].dataset.set_epoch(epoch)
        running = torch.zeros(1 + len(LOSS_COMPONENTS), device=dev, dtype=loss_dtype)
        running_lr = 0.0
        for raw in loaders["train"]:
            batch = batch_to_device(raw, dev)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(dev.type, dtype=torch.float16 if dev.type == "cuda" else torch.bfloat16)
                if amp
                else nullcontext()
            )
            with autocast:
                outputs = model(batch)
                loss, components = compute_critic_loss(outputs, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            if config.grad_clip_norm > 0:
                if scale_grads:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            batch_count = len(raw["sample_id"])
            running_lr += optimizer.param_groups[0]["lr"] * batch_count
            scale_before = scaler.get_scale() if scheduler is not None and scale_grads else 1.0
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and (not scale_grads or scaler.get_scale() >= scale_before):
                scheduler.step()
            batch_losses = torch.stack([loss.detach(), *(components[key] for key in LOSS_COMPONENTS)])
            running += batch_losses.to(loss_dtype) * batch_count
        score = _selection_nll(model, loaders["val_select"], dev)
        if not math.isfinite(score):
            raise FloatingPointError(f"non-finite selection NLL at epoch {epoch}")
        n = len(parts["train"])
        averages = dict(zip(("train_loss", *LOSS_COMPONENTS), (running / n).tolist()))
        history.append({"epoch": epoch, **averages, "lr": running_lr / n, "val_select_nll": score})
        if score < best_score:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        snap = {
            "config": config.to_dict(),
            "catalog_sha256": catalog_hash,
            "seed": seed,
            "split_seed": s2_registry.split_seed,
            "split_sha256": split_digest,
            "s2_split_sha256": s2_registry.split_sha256,
            "s5_denylist_sha256": s5_denylist.denylist_sha256,
            "frozen_holdout_verified": True,
            "group_by": s2_registry.group_by,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "schedule_steps": schedule_steps,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_model": best_state,
            "stale": stale,
            "history": history,
        }
        _save_checkpoint(last_path, snap)

    if history and history[-1]["epoch"] > start_epoch:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
    else:
        state["model"] = state.pop("resume_model", state["model"])
        _save_checkpoint(last_path, state)

    model.load_state_dict(state["best_model"])
    calibration = fit_temperature(model, loaders["val_calib"], dev)
    val_select_records = predict_records(model, loaders["val_select"], dev)
    val_select_metrics = evaluate_predictions(val_select_records)
    test_records = predict_records(model, loaders["test"], dev)
    test_metrics = evaluate_predictions(test_records)
    masked_test_loader = _make_loader(
        config, catalog, [mask_gold_sample(s) for s in parts["test"]], **loader_opts
    )
    masked_test_records = predict_records(model, masked_test_loader, dev)

    best_path = out_dir / "best.pt"
    state["resume_model"] = state["model"]
    state["model"] = model.state_dict()
    state["calibration"] = calibration
    _save_checkpoint(best_path, state)

    report = {
        "ablation_mode": config.ablation_mode,
        "seed": seed,
        "split_seed": s2_registry.split_seed,
        "group_by": s2_registry.group_by,
        "s2_split_sha256": s2_registry.split_sha256,
        "s5_denylist_sha256": s5_denylist.denylist_sha256,
        "frozen_holdout_verified": True,
        "split_counts": {k: len(rows) for k, rows in parts.items()},
        "best_epoch": best_epoch,
        "history": history,
        "calibration": calibration,
        "checkpoint": str(best_path),
        "val_select_metrics": val_select_metrics,
        "metrics": test_metrics,
        "counterfactual_gold_unknown": _counterfactual_metrics(masked_test_records),
    }
    _write_json(out_dir / "test_metrics.json", report)
    return {
        **report,
        "val_select_predictions": val_select_records,
        "test_predictions": test_records,
        "counterfactual_predictions": masked_test_records,
    }


def _load_completed_frozen_run(
    config: CriticConfig,
    catalog: FeatureCatalog,
    parts: dict[str, list[BattleSample]],
    s2_registry: S2HoldoutRegistry,
    s5_denylist: S5Denylist,
    run_dir: Path,
    *,
    seed: int,
    device: str | torch.device | None,
    patience: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, Any] | None:
    import torch

    from labyrinth_nn.models.critic import TwoTowerBattleCritic
    from labyrinth_nn.train.train_critic import _make_loader, _same_training_config, predict_records

    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "test_metrics.json"
    if not best_path.is_file() or not metrics_path.is_file():
        return None
    catalog_hash = catalog.catalog_sha256 or catalog.compute_sha256()
    split_digest = _frozen_split_digest(parts)
    try:
        state = torch.load(best_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            return None
        if (
            state.get("catalog_sha256"),
            state.get("seed"),
            state.get("split_sha256"),
            state.get("s2_split_sha256"),
            state.get("s5_denylist_sha256"),
        ) != (catalog_hash, seed, split_digest, s2_registry.split_sha256, s5_denylist.denylist_sha256):
            return None
        if not (state["epoch"] >= config.epochs or state["stale"] >= patience):
            return None
        if not _same_training_config(state["config"], config, catalog):
            return None
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = TwoTowerBattleCritic(config).to(dev)
        model.load_state_dict(state["model"])
    except (OSError, EOFError, RuntimeError, UnpicklingError, KeyError, TypeError, ValueError):
        return None

    loader_opts = dict(seed=seed, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = _make_loader(config, catalog, parts["val_select"], cache_eval=True, **loader_opts)
    test_loader = _make_loader(config, catalog, parts["test"], cache_eval=True, **loader_opts)
    masked_loader = _make_loader(
        config, catalog, [mask_gold_sample(s) for s in parts["test"]], cache_eval=True, **loader_opts
    )
    return {
        **report,
        "checkpoint": str(best_path),
        "val_select_predictions": predict_records(model, val_loader, dev),
        "test_predictions": predict_records(model, test_loader, dev),
        "counterfactual_predictions": predict_records(model, masked_loader, dev),
    }


def evaluate_s4_distillation_gate(
    *,
    single_val_select_metrics: Sequence[Mapping[str, Any]],
    ensemble_val_select_metrics: Mapping[str, Any],
    min_win_brier_gain: float = 0.003,
    min_top1_fpr_gain: float = 0.01,
) -> dict[str, Any]:
    """Compare calibrated 3-seed teacher vs best single member strictly on frozen val_select."""
    def single_rank(metrics: Mapping[str, Any]) -> tuple[float, float]:
        summary = metrics["global"]
        fpr = summary.get("context_top1_pos_fpr")
        return summary["brier_win"], fpr if fpr is not None else 1.0

    best_single = min(
        single_val_select_metrics,
        key=single_rank,
    )
    ens_g = ensemble_val_select_metrics["global"]
    sng_g = best_single["global"]
    brier_gain = float(sng_g["brier_win"]) - float(ens_g["brier_win"])
    sng_top1 = sng_g.get("context_top1_pos_fpr")
    ens_top1 = ens_g.get("context_top1_pos_fpr")
    top1_gain = (float(sng_top1) - float(ens_top1)) if (sng_top1 is not None and ens_top1 is not None) else 0.0
    triggered = brier_gain >= min_win_brier_gain or top1_gain >= min_top1_fpr_gain
    return {
        "evaluated_on": "frozen_val_select",
        "distill_triggered": triggered,
        "win_brier_gain": brier_gain,
        "min_win_brier_gain": min_win_brier_gain,
        "context_top1_pos_fpr_gain": top1_gain,
        "min_top1_fpr_gain": min_top1_fpr_gain,
        "best_single_global": sng_g,
        "ensemble_global": ens_g,
    }


def run_frozen_experiments(
    configs: Sequence[CriticConfig],
    catalog: FeatureCatalog,
    samples: Sequence[BattleSample],
    s2_registry: S2HoldoutRegistry,
    s5_denylist: S5Denylist,
    out_dir: str | Path,
    *,
    seeds: Sequence[int] = (42, 43, 44),
    device: str | None = None,
    amp: bool = False,
    patience: int = 5,
    num_workers: int = 0,
    pin_memory: bool = False,
    auto_resume: bool = True,
) -> dict[str, Any]:
    from labyrinth_nn.train.train_critic import _counterfactual_metrics, _ensemble_records, _write_json

    parts = partition_frozen_samples(samples, s2_registry, s5_denylist, catalog=catalog)
    root = Path(out_dir)
    modes: dict[str, Any] = {}
    for config in configs:
        runs = []
        val_preds, test_preds, masked_preds = [], [], []
        for seed in seeds:
            run_dir = root / config.ablation_mode / f"seed_{seed}"
            result = (
                _load_completed_frozen_run(
                    config,
                    catalog,
                    parts,
                    s2_registry,
                    s5_denylist,
                    run_dir,
                    seed=seed,
                    device=device,
                    patience=patience,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                if auto_resume
                else None
            )
            if result is None:
                last_ckpt = run_dir / "last.pt" if auto_resume and (run_dir / "last.pt").is_file() else None
                result = train_frozen_single(
                    config,
                    catalog,
                    parts,
                    s2_registry,
                    s5_denylist,
                    run_dir,
                    seed=seed,
                    device=device,
                    amp=amp,
                    resume=last_ckpt,
                    patience=patience,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            val_preds.append(result.pop("val_select_predictions"))
            test_preds.append(result.pop("test_predictions"))
            masked_preds.append(result.pop("counterfactual_predictions"))
            runs.append(result)
        val_ensemble = evaluate_predictions(_ensemble_records(val_preds))
        distill_gate = evaluate_s4_distillation_gate(
            single_val_select_metrics=[r["val_select_metrics"] for r in runs],
            ensemble_val_select_metrics=val_ensemble,
        )
        modes[config.ablation_mode] = {
            "runs": runs,
            "val_select_ensemble": val_ensemble,
            "distillation_gate": distill_gate,
            "ensemble": evaluate_predictions(_ensemble_records(test_preds)),
            "counterfactual_gold_unknown": _counterfactual_metrics(_ensemble_records(masked_preds)),
        }

    summary = {
        "s2_split_sha256": s2_registry.split_sha256,
        "s5_denylist_sha256": s5_denylist.denylist_sha256,
        "frozen_holdout_verified": True,
        "seeds": list(seeds),
        "split_counts": {k: len(v) for k, v in parts.items()},
        "modes": modes,
    }
    _write_json(root / "ablation_summary.json", summary)
    return summary


def predict_train_teacher_probs(
    teacher_checkpoints: Sequence[str | Path],
    catalog: FeatureCatalog,
    train_samples: Sequence[BattleSample],
    *,
    batch_size: int = 512,
    num_workers: int = 0,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Compute calibrated teacher outcome probabilities strictly on the train pool."""
    import torch
    from torch.utils.data import DataLoader

    from labyrinth_nn.active_mining.s3_planner import load_calibrated_ensemble

    ensemble, _ = load_calibrated_ensemble(teacher_checkpoints, catalog, device=device)
    dataset = BattleDataset(
        catalog,
        train_samples,
        training=False,
        cache_eval=False,
        roster_max_len=ensemble.config.roster_max_len,
        relic_max_len=ensemble.config.relic_max_len,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_battle_batch
    )
    probs_map: dict[str, list[float]] = {}
    with torch.no_grad():
        for raw in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
            out = ensemble(batch)
            probs = out["outcome_probs"].detach().cpu().tolist()
            for sid, p in zip(raw["sample_id"], probs, strict=True):
                probs_map[str(sid)] = [float(x) for x in p]
    return probs_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="train models on frozen S2 holdouts + S3 train")
    train_p.add_argument("--config-files", nargs="+", type=Path, required=True)
    train_p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    train_p.add_argument("--epochs", type=int)
    train_p.add_argument("--patience", type=int, default=5)
    train_p.add_argument("--amp", action="store_true")
    train_p.add_argument("--auto-resume", action="store_true")
    train_p.add_argument("--device")
    train_p.add_argument("--num-workers", type=int, default=0)
    train_p.add_argument("--pin-memory", action="store_true")
    train_p.add_argument("--catalog", type=Path, required=True)
    train_p.add_argument("--s2-registry", type=Path, required=True)
    train_p.add_argument("--s5-denylist", type=Path, required=True)
    train_p.add_argument("--samples", type=Path, nargs="+", required=True)
    train_p.add_argument("--out-dir", type=Path, required=True)

    dist_p = sub.add_parser("distill", help="distill a 3-seed teacher onto train pool into a single student")
    dist_p.add_argument("--student-config", type=Path, required=True)
    dist_p.add_argument("--teacher-checkpoints", nargs="+", type=Path, required=True)
    dist_p.add_argument("--seed", type=int, default=42)
    dist_p.add_argument("--alpha", type=float, default=0.7)
    dist_p.add_argument("--epochs", type=int)
    dist_p.add_argument("--patience", type=int, default=5)
    dist_p.add_argument("--amp", action="store_true")
    dist_p.add_argument("--device")
    dist_p.add_argument("--num-workers", type=int, default=0)
    dist_p.add_argument("--pin-memory", action="store_true")
    dist_p.add_argument("--catalog", type=Path, required=True)
    dist_p.add_argument("--s2-registry", type=Path, required=True)
    dist_p.add_argument("--s5-denylist", type=Path, required=True)
    dist_p.add_argument("--samples", type=Path, nargs="+", required=True)
    dist_p.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    import torch

    from labyrinth_nn.models.critic import CriticConfig

    catalog = FeatureCatalog.load(args.catalog)
    s2_reg = S2HoldoutRegistry.load(args.s2_registry)
    s5_deny = S5Denylist.load(args.s5_denylist)
    samples = [s for p in args.samples for s in read_samples(p)]

    if args.cmd == "train":
        configs = [CriticConfig.from_json(p, catalog) for p in args.config_files]
        if args.epochs is not None:
            configs = [replace(c, epochs=args.epochs) for c in configs]
        summary = run_frozen_experiments(
            configs,
            catalog,
            samples,
            s2_reg,
            s5_deny,
            args.out_dir,
            seeds=args.seeds,
            device=args.device,
            amp=args.amp,
            patience=args.patience,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            auto_resume=args.auto_resume,
        )
        print(json.dumps({k: v["distillation_gate"] for k, v in summary["modes"].items()}, indent=2))
        return 0

    parts = partition_frozen_samples(samples, s2_reg, s5_deny, catalog=catalog)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    teacher_probs = predict_train_teacher_probs(
        args.teacher_checkpoints, catalog, parts["train"], num_workers=args.num_workers, device=dev
    )
    student_cfg = CriticConfig.from_json(args.student_config, catalog)
    if args.epochs is not None:
        student_cfg = replace(student_cfg, epochs=args.epochs)
    report = train_frozen_single(
        student_cfg,
        catalog,
        parts,
        s2_reg,
        s5_deny,
        args.out_dir,
        seed=args.seed,
        device=dev,
        amp=args.amp,
        patience=args.patience,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        teacher_soft_probs=teacher_probs,
        distill_alpha=args.alpha,
    )
    print(json.dumps(report["val_select_metrics"]["global"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
