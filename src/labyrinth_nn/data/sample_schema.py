"""Unified battle sample schema supporting JSONL (stdlib) and optional Parquet."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

OUTCOME_WIN = 0
OUTCOME_TIMEOUT = 1
OUTCOME_WIPE = 2
OUTCOME_NAMES: tuple[str, str, str] = ("win", "timeout", "wipe")
NUM_OUTCOME_CLASSES = 3
FRAMES_PER_SECOND = 60.0


@dataclass
class SingleRunResult:
    seed: int
    result: int  # PCRCalculator eBattleResult: 0=Victory, 1=Lose, 2=TimeOver
    outcome_class: int  # 0=win, 1=timeout, 2=wipe
    battle_frames: int
    limit_time: int
    elapsed_sec: float
    rem_time_ratio: float
    total_damage: int
    damage_ratio: float
    enemy_final_hps: list[int] = field(default_factory=list)


@dataclass
class BattleSample:
    sample_id: str
    round_id: str
    source: str  # "sim" | "live"
    db_version: str
    quest_id: int
    canonical_wave_sig: str
    bucket: str
    area: int
    difficulty: int
    guild_id: int
    team_unit_ids: list[int]
    owned_unit_ids: list[int]
    owned_unit_count: int
    owned_role_counts: list[int]
    owned_talent_counts: list[int]
    relic_ids: list[int]
    mark_stacks: list[int]
    alpha_rupee: int | None
    revive_available: bool = False
    runs: list[SingleRunResult] = field(default_factory=list)
    outcome_counts: list[int] = field(default_factory=lambda: [0, 0, 0])
    mean_damage_ratio: float = 0.0
    mean_win_rem_time: float | None = None
    context_id: str = ""

    def canonical_team_sig(self) -> str:
        return ",".join(str(uid) for uid in sorted(self.team_unit_ids))

    def physical_state_key(self) -> str:
        identity = (
            self.db_version, self.quest_id, self.difficulty, self.guild_id,
            sorted(self.team_unit_ids), sorted(self.owned_unit_ids), sorted(self.relic_ids), self.alpha_rupee,
        )
        return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "round_id": self.round_id,
            "source": self.source,
            "db_version": self.db_version,
            "quest_id": self.quest_id,
            "canonical_wave_sig": self.canonical_wave_sig,
            "bucket": self.bucket,
            "area": self.area,
            "difficulty": self.difficulty,
            "guild_id": self.guild_id,
            "team_unit_ids": list(self.team_unit_ids),
            "owned_unit_ids": list(self.owned_unit_ids),
            "owned_unit_count": self.owned_unit_count,
            "owned_role_counts": list(self.owned_role_counts),
            "owned_talent_counts": list(self.owned_talent_counts),
            "relic_ids": list(self.relic_ids),
            "mark_stacks": list(self.mark_stacks),
            "alpha_rupee": self.alpha_rupee,
            "revive_available": self.revive_available,
            "runs": [asdict(r) for r in self.runs],
            "outcome_counts": list(self.outcome_counts),
            "mean_damage_ratio": float(self.mean_damage_ratio),
            "mean_win_rem_time": self.mean_win_rem_time,
            "context_id": self.context_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BattleSample":
        runs_raw = raw.get("runs") or []
        if isinstance(runs_raw, str):
            runs_raw = json.loads(runs_raw)
        runs = [SingleRunResult(**r) for r in runs_raw]
        return cls(
            sample_id=str(raw["sample_id"]),
            round_id=str(raw["round_id"]),
            source=str(raw["source"]),
            db_version=str(raw.get("db_version", "")),
            quest_id=int(raw["quest_id"]),
            canonical_wave_sig=str(raw["canonical_wave_sig"]),
            bucket=str(raw["bucket"]),
            area=int(raw["area"]),
            difficulty=int(raw["difficulty"]),
            guild_id=int(raw["guild_id"]),
            team_unit_ids=[int(x) for x in raw["team_unit_ids"]],
            owned_unit_ids=[int(x) for x in raw["owned_unit_ids"]],
            owned_unit_count=int(raw["owned_unit_count"]),
            owned_role_counts=[int(x) for x in raw["owned_role_counts"]],
            owned_talent_counts=[int(x) for x in raw["owned_talent_counts"]],
            relic_ids=[int(x) for x in raw["relic_ids"]],
            mark_stacks=[int(x) for x in raw["mark_stacks"]],
            alpha_rupee=None if raw.get("alpha_rupee") is None else int(raw["alpha_rupee"]),
            revive_available=bool(raw.get("revive_available", False)),
            runs=runs,
            outcome_counts=[int(x) for x in raw["outcome_counts"]],
            mean_damage_ratio=float(raw["mean_damage_ratio"]),
            mean_win_rem_time=None if raw.get("mean_win_rem_time") is None else float(raw["mean_win_rem_time"]),
            context_id=str(raw.get("context_id") or ""),
        )


def outcome_class_from_sim(result: int, battle_frames: int, limit_time: int) -> int:
    """Maps simulator eBattleResult (0=Victory, 1=Lose, 2=TimeOver) cleanly to (win, timeout, wipe)."""
    if result == 0:
        return OUTCOME_WIN
    limit_frames = int(round(max(1, limit_time) * FRAMES_PER_SECOND))
    if result == 2 or battle_frames >= limit_frames - 2:
        return OUTCOME_TIMEOUT
    return OUTCOME_WIPE


def aggregate_runs(runs: Iterable[SingleRunResult]) -> tuple[list[int], float, float | None]:
    """Aggregates per-run outcomes into outcome_counts[3], mean_damage_ratio, mean_win_rem_time."""
    run_list = list(runs)
    counts = [0, 0, 0]
    if not run_list:
        return counts, 0.0, None
    dmg_sum = 0.0
    win_rem_sum = 0.0
    win_n = 0
    for r in run_list:
        counts[r.outcome_class] += 1
        dmg_sum += max(0.0, min(1.0, float(r.damage_ratio)))
        if r.outcome_class == OUTCOME_WIN:
            win_rem_sum += max(0.0, min(1.0, float(r.rem_time_ratio)))
            win_n += 1
    mean_dmg = dmg_sum / len(run_list)
    mean_win_rem = (win_rem_sum / win_n) if win_n > 0 else None
    return counts, mean_dmg, mean_win_rem


def write_samples_jsonl(path: str | Path, samples: Iterable[BattleSample], append: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out.open(mode, encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    return out


def read_samples_jsonl(path: str | Path) -> list[BattleSample]:
    p = Path(path)
    if not p.exists():
        return []
    samples: list[BattleSample] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                samples.append(BattleSample.from_dict(json.loads(s)))
    return samples


def write_samples(path: str | Path, samples: Iterable[BattleSample]) -> Path:
    p = Path(path)
    sample_list = list(samples)
    if p.suffix.lower() == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Writing .parquet requires pyarrow; use .jsonl or install pyarrow.") from exc
        rows = []
        for s in sample_list:
            d = s.to_dict()
            d["runs"] = json.dumps(d["runs"], ensure_ascii=False, separators=(",", ":"))
            rows.append(d)
        p.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, p, compression="zstd")
        return p
    return write_samples_jsonl(p, sample_list, append=False)


def read_samples(path: str | Path) -> list[BattleSample]:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Reading .parquet requires pyarrow; use .jsonl or install pyarrow.") from exc
        table = pq.read_table(p)
        return [BattleSample.from_dict(row) for row in table.to_pylist()]
    return read_samples_jsonl(p)
