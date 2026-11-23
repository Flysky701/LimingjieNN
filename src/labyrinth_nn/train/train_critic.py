"""Train, independently calibrate and compare late-interaction battle critics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from labyrinth_nn.catalog.schema import FeatureCatalog
from labyrinth_nn.data.dataset import BattleDataset, collate_battle_batch, mask_gold_sample, split_samples
from labyrinth_nn.data.sample_schema import BattleSample, read_samples
from labyrinth_nn.models.critic import CriticConfig, TwoTowerBattleCritic
from labyrinth_nn.train.losses import batch_to_device, compute_critic_loss, fit_temperature
from labyrinth_nn.train.metrics import evaluate_predictions

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
LOSS_COMPONENTS = ("outcome_loss", "damage_loss", "damage_huber", "damage_nll", "time_loss", "affinity_loss")


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _save_checkpoint(path: Path, payload) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def split_training_samples(samples, config, split_seed, group_by="wave_team"):
    train, val, test = split_samples(samples, config.val_ratio, config.test_ratio, split_seed, group_by=group_by)
    select, calib, _ = split_samples(val, val_ratio=0.5, test_ratio=0.0, seed=split_seed + 1, group_by=group_by)
    parts = {name: [s for s in rows if sum(s.outcome_counts) > 0]
             for name, rows in (("train", train), ("val_select", select), ("val_calib", calib), ("test", test))}
    if not parts["train"]:
        raise ValueError("training needs labeled samples")
    if config.val_ratio and (not parts["val_select"] or not parts["val_calib"]):
        raise ValueError(f"independent selection and calibration need at least two labeled {group_by} groups "
                         "in validation; increase val_ratio or supply more groups")
    if config.test_ratio and not parts["test"]:
        raise ValueError(f"the requested test split has no labeled {group_by} group")
    return parts


def _validate_samples(samples: Sequence[BattleSample]) -> None:
    seen = set()
    versions = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample.sample_id}")
        seen.add(sample.sample_id)
        versions.add(sample.db_version)
    if len(versions) > 1:
        raise ValueError(f"mixed db_version values: {', '.join(map(repr, sorted(versions)))}")


def _cosine_scheduler(optimizer, config: CriticConfig, steps: int):
    warmup = min(steps - 1, math.ceil(steps * config.warmup_ratio))

    def ratio(step: int) -> float:
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = min(1.0, (step - warmup) / max(1, steps - warmup - 1))
        return config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)


@torch.no_grad()
def _selection_nll(model, loader, device):
    model.eval()
    total = count = 0
    for raw in loader:
        batch = batch_to_device(raw, device)
        logits = model(batch)["outcome_logits"].float()
        total += -(batch["outcome_probs_target"] * F.log_softmax(logits, dim=-1)).sum().item()
        count += logits.shape[0]
    return total / count


@torch.no_grad()
def predict_records(model, loader, device):
    model.eval()
    records = []
    for raw in loader:
        outputs = model(batch_to_device(raw, device))
        probs = outputs["outcome_probs"].tolist()
        damage = outputs["damage_mean"].reshape(-1).tolist()
        time = outputs["win_rem_time"].reshape(-1).tolist()
        contexts = raw.get("context_id", [""] * len(probs))
        gold_relics = raw.get("has_gold_scaling_relic", [False] * len(probs))
        for i, p in enumerate(probs):
            records.append({
                "sample_id": raw["sample_id"][i], "canonical_wave_sig": raw["canonical_wave_sig"][i],
                "team_sig": raw["team_sig"][i], "bucket": raw["bucket"][i], "gold_known": raw["gold_known"][i],
                "context_id": contexts[i], "has_gold_scaling_relic": gold_relics[i],
                "pred_probs": p, "target_probs": raw["outcome_probs_target"][i].tolist(),
                "pred_damage_mean": damage[i], "target_damage_ratio": raw["mean_damage_ratio"][i].item(),
                "pred_win_rem_time": time[i], "target_win_rem_time": raw["mean_win_rem_time"][i].item(),
                "has_win_target": raw["has_win_target"][i].item(),
            })
    return records


def _counterfactual_metrics(records):
    metrics = evaluate_predictions(records)
    return {"global": metrics["global"], "slices": {"gold_scaling_relic": metrics["slices"]["gold_scaling_relic"]}}


def train_critic(config: CriticConfig | str | Path, catalog: FeatureCatalog | str | Path,
                 samples: Sequence[BattleSample] | str | Path, out_dir: str | Path, *, seed: int | None = None,
                 split_seed: int | None = None, device: str | torch.device | None = None, amp: bool = False,
                 resume: str | Path | None = None, patience: int = 5, num_workers: int = 0,
                 pin_memory: bool = False, group_by: str = "wave_team") -> dict[str, Any]:
    """A single run. Both checkpoints can resume at the last completed training epoch."""
    if not isinstance(catalog, FeatureCatalog):
        catalog = FeatureCatalog.load(catalog)
    if not isinstance(config, CriticConfig):
        config = CriticConfig.from_json(config, catalog)
    else:
        CriticConfig.from_dict(config.to_dict(), catalog)
    if isinstance(samples, (str, Path)):
        samples = read_samples(samples)
    _validate_samples(samples)
    if min(config.epochs, config.batch_size, patience) < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume is not None else {}
    seed = state.get("seed", 42) if seed is None else seed
    split_seed = state.get("split_seed", seed) if split_seed is None else split_seed
    parts = split_training_samples(samples, config, split_seed, group_by=group_by)
    # Stable hash ordering alone cannot protect a resumed holdout if groups were added or removed.
    partition = hashlib.sha256()
    for name, rows in parts.items():
        for sample in rows:
            identity = (name, sample.sample_id, sample.canonical_wave_sig, sample.canonical_team_sig())
            partition.update(json.dumps(identity, ensure_ascii=False).encode("utf-8") + b"\n")
    split_digest = partition.hexdigest()
    data_dir = Path(out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    catalog_hash = catalog.catalog_sha256 or catalog.compute_sha256()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if amp and device.type not in ("cuda", "cpu"):
        raise ValueError("AMP training supports CUDA float16 and CPU bfloat16")
    torch.manual_seed(seed)
    shuffle = torch.Generator()
    def make_loader(rows, *, training=False, cache_eval=False):
        dataset = BattleDataset(catalog, rows, seed=seed, training=training, cache_eval=cache_eval,
                                relic_max_len=config.relic_max_len, roster_max_len=config.roster_max_len,
                                unit_id_dropout=config.unit_id_dropout, relic_id_mask_prob=config.relic_id_mask_prob,
                                relic_stack_noise_prob=config.relic_stack_noise_prob, gold_mask_prob=config.gold_mask_prob)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=training,
                          collate_fn=collate_battle_batch, generator=shuffle if training else None,
                          num_workers=num_workers, pin_memory=pin_memory)

    loaders = {name: make_loader(rows, training=name == "train", cache_eval=name != "train")
               for name, rows in parts.items()}
    model = TwoTowerBattleCritic(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # Use the original schedule horizon on resume, even if more epochs were requested.
    schedule_steps = state.get("schedule_steps", len(loaders["train"]) * config.epochs)
    scheduler = (_cosine_scheduler(optimizer, config, schedule_steps) if config.lr_schedule == "cosine" else None)
    scale_grads = amp and device.type == "cuda"
    scaler_cls = getattr(torch.amp, "GradScaler", None)
    scaler = (scaler_cls("cuda", enabled=scale_grads) if scaler_cls is not None
              else torch.cuda.amp.GradScaler(enabled=scale_grads))
    best_state = None
    best_score = math.inf
    best_epoch = start_epoch = stale = 0
    history = []
    if resume is not None:
        saved_config = CriticConfig.from_dict(state["config"], catalog).to_dict()
        same_config = all(saved_config[k] == v for k, v in config.to_dict().items() if k != "epochs")
        if ((state["catalog_sha256"], state["seed"], state["split_seed"]) != (catalog_hash, seed, split_seed)
                or not same_config):
            raise ValueError("checkpoint catalog, seeds or training configuration differ")
        if state["split_sha256"] != split_digest or state.get("group_by", "wave_team") != group_by:
            raise ValueError("checkpoint sample split differs from the supplied samples")
        model.load_state_dict(state.get("resume_model", state["model"]))
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scale_grads and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        start_epoch, best_epoch, best_score = state["epoch"], state["best_epoch"], state["best_score"]
        best_state, stale, history = state["best_model"], state["stale"], state["history"]
    last_path = data_dir / "last.pt"
    loss_dtype = torch.float32 if device.type == "mps" else torch.float64
    for epoch in range(start_epoch + 1, config.epochs + 1):
        if stale >= patience:
            break
        model.train()
        shuffle.manual_seed(seed + epoch)
        loaders["train"].dataset.set_epoch(epoch)
        running = torch.zeros(1 + len(LOSS_COMPONENTS), device=device, dtype=loss_dtype)
        running_lr = 0.0
        for raw in loaders["train"]:
            batch = batch_to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (torch.autocast(device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16)
                        if amp else nullcontext())
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
        score = _selection_nll(model, loaders["val_select"], device) if parts["val_select"] else None
        if score is not None and not math.isfinite(score):
            raise FloatingPointError(f"non-finite selection NLL at epoch {epoch}")
        n = len(parts["train"])
        averages = dict(zip(("train_loss", *LOSS_COMPONENTS), (running / n).tolist()))
        history.append({"epoch": epoch, **averages, "lr": running_lr / n, "val_select_nll": score})
        if score is None or score < best_score:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch, best_score, stale = epoch, score if score is not None else math.inf, 0
        else:
            stale += 1
        state = {
            "config": config.to_dict(), "catalog_sha256": catalog_hash, "seed": seed, "split_seed": split_seed,
            "split_sha256": split_digest, "group_by": group_by,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None, "schedule_steps": schedule_steps,
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "epoch": epoch, "best_epoch": best_epoch, "best_score": best_score, "best_model": best_state,
            "stale": stale, "history": history,
        }
        _save_checkpoint(last_path, state)
    # Reload the independent training snapshot. Loading best weights below must not mutate it through aliases.
    if history and history[-1]["epoch"] > start_epoch:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
    else:
        state["model"] = state.pop("resume_model", state["model"])
        _save_checkpoint(last_path, state)
    model.load_state_dict(state["best_model"])
    calibration = fit_temperature(model, loaders["val_calib"], device) if parts["val_calib"] else None
    records = predict_records(model, loaders["test"], device)
    metrics = evaluate_predictions(records)
    masked_loader = make_loader([mask_gold_sample(sample) for sample in parts["test"]])
    masked_records = predict_records(model, masked_loader, device)
    best_path = data_dir / "best.pt"
    state["resume_model"] = state["model"]
    state["model"] = model.state_dict()
    _save_checkpoint(best_path, state)
    report = {
        "ablation_mode": config.ablation_mode, "seed": seed, "split_seed": split_seed, "group_by": group_by,
        "split_counts": {k: len(rows) for k, rows in parts.items()},
        "best_epoch": best_epoch, "history": history, "calibration": calibration,
        "checkpoint": str(best_path), "metrics": metrics,
        "counterfactual_gold_unknown": _counterfactual_metrics(masked_records),
    }
    _write_json(data_dir / "test_metrics.json", report)
    return {**report, "test_predictions": records, "counterfactual_predictions": masked_records}


def _ensemble_records(members):
    if any(len(rows) != len(members[0]) for rows in members):
        raise ValueError("ensemble members must share the same held-out samples")
    averaged = []
    for group in zip(*members):
        first = group[0]
        keys = ("sample_id", "canonical_wave_sig", "team_sig", "target_probs")
        if any(any(r[k] != first[k] for k in keys) for r in group[1:]):
            raise ValueError("ensemble prediction rows do not align")
        result = dict(first)
        result["pred_probs"] = [statistics.fmean(p) for p in zip(*(r["pred_probs"] for r in group))]
        for name in ("pred_damage_mean", "pred_win_rem_time"):
            result[name] = statistics.fmean(r[name] for r in group)
        averaged.append(result)
    return averaged


def run_experiments(configs: Sequence[CriticConfig], catalog: FeatureCatalog, samples: Sequence[BattleSample],
                    out_dir: str | Path, *, seeds: Sequence[int] | None = None, split_seed: int | None = None,
                    device=None, amp=False, resume=None, patience=5, num_workers=0, pin_memory=False,
                    group_by="wave_team"):
    if resume and (seeds is None or split_seed is None):
        metadata = torch.load(resume, map_location="meta", weights_only=True)
        seeds = [metadata["seed"]] if seeds is None else seeds
        split_seed = metadata["split_seed"] if split_seed is None else split_seed
        del metadata
    if seeds is None:
        seeds = (42,)
    if not configs or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("supply at least one configuration and distinct training seeds")
    if len({c.ablation_mode for c in configs}) != len(configs):
        raise ValueError("each configuration needs a distinct ablation_mode")
    if resume and len(configs) * len(seeds) != 1:
        raise ValueError("a resume checkpoint identifies one configuration and training seed")
    split_seed = seeds[0] if split_seed is None else split_seed
    root = Path(out_dir)
    modes = {}
    for config in configs:
        runs, predictions, masked_predictions = [], [], []
        for seed in seeds:
            result = train_critic(config, catalog, samples, root / config.ablation_mode / f"seed_{seed}",
                                  seed=seed, split_seed=split_seed, device=device, amp=amp, resume=resume,
                                  patience=patience, num_workers=num_workers, pin_memory=pin_memory, group_by=group_by)
            predictions.append(result.pop("test_predictions"))
            masked_predictions.append(result.pop("counterfactual_predictions"))
            runs.append(result)
        aggregate = {}
        for metric in runs[0]["metrics"]["global"]:
            values = [run["metrics"]["global"][metric] for run in runs if run["metrics"]["global"][metric] is not None]
            aggregate[metric] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values)} if values else None
        modes[config.ablation_mode] = {
            "runs": runs, "seed_summary": aggregate, "ensemble": evaluate_predictions(_ensemble_records(predictions)),
            "counterfactual_gold_unknown": _counterfactual_metrics(_ensemble_records(masked_predictions)),
        }
    summary = {"split_seed": split_seed, "seeds": list(seeds), "group_by": group_by, "modes": modes}
    _write_json(root / "ablation_summary.json", summary)
    columns = ("auroc", "brier_3class", "brier_win", "ece_15", "accuracy_3class", "loss_damage_mae",
               "win_time_mae", "group_top1_fpr", "group_top1_pos_fpr")
    lines = ["# 消融对比", "", f"切分种子 {split_seed}，训练种子 {', '.join(map(str, seeds))}。表中为集成预测。", "",
             "| 配置 | 测试切片 | 样本数 | " + " | ".join(columns) + " |",
             "|" + " --- |" * (3 + len(columns))]
    for mode, entry in modes.items():
        metrics = entry["ensemble"]
        masked = entry["counterfactual_gold_unknown"]
        rows = {"global": metrics["global"], **metrics["buckets"], **metrics["slices"],
                "gold_unknown_cf": masked["global"], "gold_unknown_cf_gold_relic": masked["slices"]["gold_scaling_relic"]}
        for name, row in rows.items():
            values = [f"{row[k]:.5f}" if row[k] is not None else "—" for k in columns]
            lines.append(f"| {mode} | {name} | {row['count']} | " + " | ".join(values) + " |")
    (root / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", type=Path)
    group.add_argument("--configs", nargs="+", choices=("m0", "m1", "m2a", "m2b"))
    parser.add_argument("--config-files", nargs="+", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--split-seed", type=int, help="common holdout seed (defaults to the first training seed)")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--group-by", choices=("wave_team", "context", "wave"), default="wave_team")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [args.config] if args.config else [CONFIG_DIR / f"critic_{m}.json" for m in (args.configs or ())]
    paths.extend(args.config_files or ())
    if not paths:
        parser.error("supply --config, --configs, or --config-files")
    for path in args.samples:
        if not path.is_file():
            raise FileNotFoundError(f"sample shard does not exist: {path}")
    samples = [s for path in args.samples for s in read_samples(path)]
    _validate_samples(samples)
    catalog = FeatureCatalog.load(args.catalog)
    configs = [CriticConfig.from_json(path, catalog) for path in paths]
    run_experiments(configs, catalog, samples, args.out_dir, seeds=args.seeds, split_seed=args.split_seed,
                    device=args.device, amp=args.amp, resume=args.resume, patience=args.patience,
                    num_workers=args.num_workers, pin_memory=args.pin_memory, group_by=args.group_by)
    print(args.out_dir / "ablation_summary.md")


if __name__ == "__main__":
    main()
