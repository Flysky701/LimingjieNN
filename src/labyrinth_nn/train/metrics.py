"""Calibration and ranking metrics for pooled simulator outcomes, using the standard library."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import groupby
from typing import Any

EVAL_BUCKETS = ("normal", "robot_ex", "special_dual_ex", "other_ex", "boss_3f_d5", "boss_5f_d5")


def _value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def _scalar(value: Any) -> float:
    value = _value(value)
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return float(value)


@dataclass(slots=True)
class _Record:
    pred: list[float]
    target: list[float]
    damage_error: float | None
    time_error: float | None
    wave: str
    team: str
    bucket: str
    gold: bool

    @classmethod
    def read(cls, row: Mapping[str, Any]) -> _Record:
        pred = list(map(_scalar, _value(row["pred_probs"])))
        target = list(map(_scalar, _value(row["target_probs"])))
        if len(pred) != 3 or len(target) != 3:
            raise ValueError("expected three outcome probabilities in win, timeout, wipe order")
        damage = None
        if target[0] < 0.5:
            damage = abs(_scalar(row["pred_damage_mean"]) - _scalar(row["target_damage_ratio"]))
        time = None
        if _scalar(row["has_win_target"]):
            time = abs(_scalar(row["pred_win_rem_time"]) - _scalar(row["target_win_rem_time"]))
        return cls(pred, target, damage, time, str(row["canonical_wave_sig"]),
                   str(row.get("team_sig", "")), str(row["bucket"]), bool(_scalar(row["gold_known"])))


def _auc(rows: list[_Record]) -> float | None:
    """Win-majority vs non-win ROC AUC, with half credit for tied scores."""
    positives = sum(r.target[0] >= 0.5 for r in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(rows, key=lambda r: r.pred[0])
    wins = lower_negatives = 0
    for _, tied in groupby(ordered, key=lambda r: r.pred[0]):
        labels = [r.target[0] >= 0.5 for r in tied]
        pos = sum(labels)
        neg = len(labels) - pos
        wins += pos * (lower_negatives + 0.5 * neg)
        lower_negatives += neg
    return wins / (positives * negatives)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metrics(rows: list[_Record]) -> dict[str, int | float | None]:
    n = len(rows)
    win_brier = full_brier = correct = 0.0
    bin_residual = [0.0] * 15
    top: dict[str, _Record] = {}
    for row in rows:
        pred, target = row.pred, row.target
        full_brier += sum((p - y) ** 2 for p, y in zip(pred, target))
        win_brier += (pred[0] - target[0]) ** 2
        correct += max(range(3), key=pred.__getitem__) == max(range(3), key=target.__getitem__)
        bin_residual[min(14, max(0, int(pred[0] * 15)))] += pred[0] - target[0]
        previous = top.get(row.wave)
        if previous is None or (-pred[0], row.team) < (-previous.pred[0], previous.team):
            top[row.wave] = row
    return {
        "count": n, "auroc": _auc(rows),
        "brier_3class": full_brier / n if n else None, "brier_win": win_brier / n if n else None,
        "ece_15": sum(map(abs, bin_residual)) / n if n else None,
        "accuracy_3class": correct / n if n else None,
        "loss_damage_mae": _mean([r.damage_error for r in rows if r.damage_error is not None]),
        "win_time_mae": _mean([r.time_error for r in rows if r.time_error is not None]),
        "group_top1_fpr": _mean([float(r.target[0] < 0.5) for r in top.values()]),
        "group_top1_pos_fpr": _mean([float(r.target[0] < 0.5) for r in top.values() if r.pred[0] >= 0.5]),
    }


def compute_slice_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, int | float | None]:
    """Brier and 15-bin win ECE use soft targets. Top-1 FPR counts failed recommendations
    among all waves, or among waves with a predicted positive top team for pos_fpr.
    """
    return _metrics([_Record.read(record) for record in records])


def evaluate_predictions(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [_Record.read(record) for record in records]
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.bucket].append(row)
    return {
        "global": _metrics(rows), "buckets": {name: _metrics(buckets[name]) for name in EVAL_BUCKETS},
        "slices": {"gold_unknown": _metrics([row for row in rows if not row.gold])},
    }
