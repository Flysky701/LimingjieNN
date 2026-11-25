"""Canonical context hashing, worker sharding, and frozen S2 holdout registries."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labyrinth_nn.data.dataset import split_samples
from labyrinth_nn.data.sample_schema import BattleSample

DEFAULT_NUM_WORKERS = 5
HOLDOUT_FOLDS = ("val_select", "val_calib", "test")
ALL_FOLDS = ("train", *HOLDOUT_FOLDS)
UNSUPPORTED_SIM_TEAM_UNITS: frozenset[int] = frozenset({136001, 136201, 137401})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def write_text_atomic(path: str | Path, text: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return sha256_file(target)


def write_json_atomic(path: str | Path, payload: Any, *, indent: int | None = 2) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False) + "\n"
    return write_text_atomic(path, text)


def write_jsonl_atomic(path: str | Path, records: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        for record in records
    ]
    text = ("\n".join(lines) + "\n") if lines else ""
    return write_text_atomic(path, text)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def semantic_context_tuple(sample: BattleSample | Mapping[str, Any]) -> tuple[Any, ...]:
    if isinstance(sample, BattleSample):
        db_version = sample.db_version
        quest_id = sample.quest_id
        difficulty = sample.difficulty
        guild_id = sample.guild_id
        owned = sample.owned_unit_ids
        relics = sample.relic_ids
        gold = sample.alpha_rupee
    else:
        db_version = sample.get("db_version", "")
        quest_id = sample["quest_id"]
        difficulty = sample["difficulty"]
        guild_id = sample["guild_id"]
        owned = sample["owned_unit_ids"]
        relics = sample["relic_ids"]
        gold = sample.get("alpha_rupee")
    return (
        str(db_version),
        int(quest_id),
        int(difficulty),
        int(guild_id),
        sorted(int(u) for u in owned),
        sorted(int(r) for r in relics),
        None if gold is None else int(gold),
    )


def semantic_context_sha256(sample: BattleSample | Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(semantic_context_tuple(sample), separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def physical_state_key_from_mapping(sample: BattleSample | Mapping[str, Any]) -> str:
    if isinstance(sample, BattleSample):
        return sample.physical_state_key()
    identity = (
        str(sample.get("db_version", "")),
        int(sample["quest_id"]),
        int(sample["difficulty"]),
        int(sample["guild_id"]),
        sorted(int(u) for u in sample["team_unit_ids"]),
        sorted(int(u) for u in sample["owned_unit_ids"]),
        sorted(int(r) for r in sample["relic_ids"]),
        None if sample.get("alpha_rupee") is None else int(sample["alpha_rupee"]),
    )
    return hashlib.sha256(json.dumps(identity, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def worker_for_context(ctx_sha256: str, num_workers: int = DEFAULT_NUM_WORKERS) -> int:
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    normalized = ctx_sha256.lower()
    if not _HEX64.fullmatch(normalized):
        raise ValueError(f"semantic context digest must be 64 lowercase hex chars, got: {ctx_sha256!r}")
    return int(normalized, 16) % num_workers


def worker_id_for_index(index: int) -> str:
    if index < 0:
        raise ValueError("worker index must be nonnegative")
    return f"w{index + 1:02d}"


def worker_index_from_id(worker_id: str, num_workers: int = DEFAULT_NUM_WORKERS) -> int:
    match = re.fullmatch(r"w(\d{2,})", worker_id)
    if not match:
        raise ValueError(f"invalid worker_id: {worker_id!r}")
    index = int(match.group(1)) - 1
    if not 0 <= index < num_workers:
        raise ValueError(f"worker_id {worker_id!r} outside [0, {num_workers})")
    return index


def split_digest_for_parts(parts: Mapping[str, Sequence[BattleSample]]) -> str:
    partition = hashlib.sha256()
    for name, rows in parts.items():
        for sample in rows:
            identity = (name, sample.sample_id, sample.canonical_wave_sig, sample.canonical_team_sig())
            partition.update(json.dumps(identity, ensure_ascii=False).encode("utf-8") + b"\n")
    return partition.hexdigest()


def split_s2_samples(
    samples: Sequence[BattleSample],
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: int = 42,
    group_by: str = "context",
) -> dict[str, list[BattleSample]]:
    """Pure-stdlib equivalent of train_critic.split_training_samples."""
    train, val, test = split_samples(samples, val_ratio, test_ratio, split_seed, group_by=group_by)
    select, calib, _ = split_samples(val, val_ratio=0.5, test_ratio=0.0, seed=split_seed + 1, group_by=group_by)
    parts = {
        name: [s for s in rows if sum(s.outcome_counts) > 0]
        for name, rows in (("train", train), ("val_select", select), ("val_calib", calib), ("test", test))
    }
    if not parts["train"]:
        raise ValueError("S2 split requires labeled train samples")
    if val_ratio and (not parts["val_select"] or not parts["val_calib"]):
        raise ValueError("S2 split requires labeled val_select and val_calib groups")
    if test_ratio and not parts["test"]:
        raise ValueError("S2 split requires labeled test samples")
    return parts


@dataclass(frozen=True)
class S2HoldoutRegistry:
    schema_version: int
    db_version: str
    catalog_sha256: str
    split_seed: int
    group_by: str
    split_sha256: str
    fold_counts: dict[str, int]
    fold_orders: dict[str, tuple[str, ...]]
    sample_folds: dict[str, str]
    holdout_semantic_contexts: frozenset[str]
    holdout_raw_context_ids: frozenset[str]
    holdout_physical_states: frozenset[str]
    holdout_sample_ids: frozenset[str]
    denylist_sha256: str
    seal_sha256: str = ""

    def is_holdout_sample(self, sample: BattleSample | Mapping[str, Any]) -> bool:
        ctx_sha = semantic_context_sha256(sample)
        phys_sha = physical_state_key_from_mapping(sample)
        sample_id = sample.sample_id if isinstance(sample, BattleSample) else str(sample.get("sample_id", ""))
        raw_ctx = sample.context_id if isinstance(sample, BattleSample) else str(sample.get("context_id") or "")
        return (
            ctx_sha in self.holdout_semantic_contexts
            or phys_sha in self.holdout_physical_states
            or (bool(sample_id) and sample_id in self.holdout_sample_ids)
            or (bool(raw_ctx) and raw_ctx in self.holdout_raw_context_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "db_version": self.db_version,
            "catalog_sha256": self.catalog_sha256,
            "seal_sha256": self.seal_sha256,
            "split_seed": self.split_seed,
            "group_by": self.group_by,
            "split_sha256": self.split_sha256,
            "fold_counts": dict(self.fold_counts),
            "fold_orders": {k: list(v) for k, v in self.fold_orders.items()},
            "sample_folds": dict(sorted(self.sample_folds.items())),
            "holdout_semantic_contexts": sorted(self.holdout_semantic_contexts),
            "holdout_raw_context_ids": sorted(self.holdout_raw_context_ids),
            "holdout_physical_states": sorted(self.holdout_physical_states),
            "holdout_sample_ids": sorted(self.holdout_sample_ids),
            "denylist_sha256": self.denylist_sha256,
        }

    def save(self, path: str | Path) -> str:
        return write_json_atomic(path, self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> S2HoldoutRegistry:
        denylist_core = {
            "db_version": str(raw["db_version"]),
            "catalog_sha256": str(raw["catalog_sha256"]),
            "split_sha256": str(raw["split_sha256"]),
            "holdout_semantic_contexts": sorted(raw["holdout_semantic_contexts"]),
            "holdout_raw_context_ids": sorted(raw["holdout_raw_context_ids"]),
            "holdout_physical_states": sorted(raw["holdout_physical_states"]),
            "holdout_sample_ids": sorted(raw["holdout_sample_ids"]),
        }
        computed_denylist = sha256_json(denylist_core)
        if raw.get("denylist_sha256") != computed_denylist:
            raise ValueError("S2 holdout denylist_sha256 mismatch")
        fold_orders_raw = raw.get("fold_orders") or {}
        fold_orders = {str(k): tuple(str(x) for x in v) for k, v in fold_orders_raw.items()}
        return cls(
            schema_version=int(raw["schema_version"]),
            db_version=str(raw["db_version"]),
            catalog_sha256=str(raw["catalog_sha256"]),
            seal_sha256=str(raw.get("seal_sha256", "")),
            split_seed=int(raw["split_seed"]),
            group_by=str(raw["group_by"]),
            split_sha256=str(raw["split_sha256"]),
            fold_counts={str(k): int(v) for k, v in raw["fold_counts"].items()},
            fold_orders=fold_orders,
            sample_folds={str(k): str(v) for k, v in raw["sample_folds"].items()},
            holdout_semantic_contexts=frozenset(raw["holdout_semantic_contexts"]),
            holdout_raw_context_ids=frozenset(raw["holdout_raw_context_ids"]),
            holdout_physical_states=frozenset(raw["holdout_physical_states"]),
            holdout_sample_ids=frozenset(raw["holdout_sample_ids"]),
            denylist_sha256=computed_denylist,
        )

    @classmethod
    def load(cls, path: str | Path) -> S2HoldoutRegistry:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_s2_holdout_registry(
    samples: Sequence[BattleSample],
    *,
    catalog_sha256: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: int = 42,
    group_by: str = "context",
    expected_split_sha256: str | None = None,
    seal_sha256: str = "",
    sort_inputs: bool = False,
) -> tuple[S2HoldoutRegistry, dict[str, list[BattleSample]]]:
    """Validate S2 split digest (preserving canonical input order by default) and emit frozen holdout registry."""
    ordered = (
        sorted(samples, key=lambda s: (s.sample_id, s.physical_state_key()))
        if sort_inputs
        else list(samples)
    )
    seen_ids: set[str] = set()
    db_versions: set[str] = set()
    for s in ordered:
        if s.sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id in S2 corpus: {s.sample_id}")
        seen_ids.add(s.sample_id)
        db_versions.add(s.db_version)
    if len(db_versions) != 1:
        raise ValueError(f"expected single db_version in S2 corpus, got: {sorted(db_versions)}")
    db_version = next(iter(db_versions))

    parts = split_s2_samples(
        ordered, val_ratio=val_ratio, test_ratio=test_ratio, split_seed=split_seed, group_by=group_by
    )
    split_sha = split_digest_for_parts(parts)
    if expected_split_sha256 is not None and split_sha != expected_split_sha256:
        raise ValueError(f"S2 split_sha256 mismatch: expected={expected_split_sha256} actual={split_sha}")

    sample_folds: dict[str, str] = {}
    holdout_ctx: set[str] = set()
    holdout_raw_ctx: set[str] = set()
    holdout_phys: set[str] = set()
    holdout_ids: set[str] = set()
    for fold, rows in parts.items():
        for s in rows:
            sample_folds[s.sample_id] = fold
            if fold in HOLDOUT_FOLDS:
                holdout_ctx.add(semantic_context_sha256(s))
                holdout_phys.add(s.physical_state_key())
                holdout_ids.add(s.sample_id)
                if s.context_id:
                    holdout_raw_ctx.add(s.context_id)

    denylist_core = {
        "db_version": db_version,
        "catalog_sha256": catalog_sha256,
        "split_sha256": split_sha,
        "holdout_semantic_contexts": sorted(holdout_ctx),
        "holdout_raw_context_ids": sorted(holdout_raw_ctx),
        "holdout_physical_states": sorted(holdout_phys),
        "holdout_sample_ids": sorted(holdout_ids),
    }
    registry = S2HoldoutRegistry(
        schema_version=1,
        db_version=db_version,
        catalog_sha256=catalog_sha256,
        seal_sha256=seal_sha256,
        split_seed=split_seed,
        group_by=group_by,
        split_sha256=split_sha,
        fold_counts={fold: len(rows) for fold, rows in parts.items()},
        fold_orders={fold: tuple(s.sample_id for s in rows) for fold, rows in parts.items()},
        sample_folds=sample_folds,
        holdout_semantic_contexts=frozenset(holdout_ctx),
        holdout_raw_context_ids=frozenset(holdout_raw_ctx),
        holdout_physical_states=frozenset(holdout_phys),
        holdout_sample_ids=frozenset(holdout_ids),
        denylist_sha256=sha256_json(denylist_core),
    )
    return registry, parts
