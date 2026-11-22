"""Official settlement arithmetic. All public point values are difficulty-scaled."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import gcd

from labyrinth_nn.catalog.schema import ScoreSettings


DEFAULT_SETTINGS = ScoreSettings(
    area_clear_pts=dict.fromkeys(range(1, 6), 500),
    block_pts=dict.fromkeys(range(1, 6), 100),
    score_rates=dict(enumerate((100, 120, 140, 170, 200, 220, 240, 260, 280, 300), 1)),
)


@dataclass(frozen=True)
class ScoreCounts:
    areas_cleared: int | tuple[int, ...] | list[int] | set[int] = 0
    blocks_cleared: int | dict[int, int] | tuple[int, ...] | list[int] = 0
    units: int = 0
    relics: int = 0
    normal_wins: int = 0
    ex_wins: int = 0
    boss_wins: int = 0

    def area_ids(self, settings: ScoreSettings) -> set[int]:
        if isinstance(self.areas_cleared, int):
            if not 0 <= self.areas_cleared <= len(settings.area_clear_pts):
                raise ValueError("area count outside configured floors")
            return set(sorted(settings.area_clear_pts)[:self.areas_cleared])
        return set(self.areas_cleared)


def base_score(settings: ScoreSettings, counts: ScoreCounts) -> int:
    blocks = counts.blocks_cleared
    if isinstance(blocks, int):
        prices = set(settings.block_pts.values())
        if blocks and len(prices) != 1:
            raise ValueError("supply block area ids or counts per area for nonuniform block points")
        block_score = blocks * next(iter(prices), 0)
    elif isinstance(blocks, dict):
        block_score = sum(settings.block_pts[a] * n for a, n in blocks.items())
    else:
        block_score = sum(settings.block_pts[a] for a in blocks)
    return (sum(settings.area_clear_pts[a] for a in counts.area_ids(settings))
            + block_score + counts.units * settings.unit_pts + counts.relics * settings.relic_pts
            + counts.normal_wins * settings.normal_pts + counts.ex_wins * settings.ex_pts
            + counts.boss_wins * settings.boss_pts)


def scale_pts(settings: ScoreSettings, difficulty: int, base_delta: int) -> int:
    return base_delta * settings.score_rates[difficulty] // 100


def compute_final_score(settings: ScoreSettings, difficulty: int, counts: ScoreCounts) -> int:
    return scale_pts(settings, difficulty, base_score(settings, counts))


def score_gcd(settings: ScoreSettings, difficulty: int) -> int:
    values = (settings.unit_pts, settings.relic_pts, settings.normal_pts, settings.ex_pts,
              settings.boss_pts, *settings.area_clear_pts.values(), *settings.block_pts.values())
    product = gcd(*(v for v in values if v > 0)) * settings.score_rates[difficulty]
    return product // 100 if product % 100 == 0 else 1


def unit_delta_pts(settings: ScoreSettings, difficulty: int) -> int:
    return scale_pts(settings, difficulty, settings.unit_pts)


def relic_delta_pts(settings: ScoreSettings, difficulty: int) -> int:
    return scale_pts(settings, difficulty, settings.relic_pts)


def normal_win_delta_pts(settings: ScoreSettings, difficulty: int) -> int:
    return scale_pts(settings, difficulty, settings.normal_pts)


def ex_win_delta_pts(settings: ScoreSettings, difficulty: int) -> int:
    return scale_pts(settings, difficulty, settings.ex_pts)


def boss_win_delta_pts(settings: ScoreSettings, difficulty: int) -> int:
    return scale_pts(settings, difficulty, settings.boss_pts)


def block_delta_pts(settings: ScoreSettings, difficulty: int, area: int) -> int:
    return scale_pts(settings, difficulty, settings.block_pts[area])


def area_clear_delta_pts(settings: ScoreSettings, difficulty: int, area: int) -> int:
    return scale_pts(settings, difficulty, settings.area_clear_pts[area])


def delta_ex_vs_normal_pts(settings: ScoreSettings, difficulty: int, *,
                          extra_relics: int = 0, extra_units: int = 0) -> int:
    """Extra counts are signed EX-minus-normal differences in collection rewards."""
    return scale_pts(settings, difficulty, settings.ex_pts - settings.normal_pts
                     + extra_relics * settings.relic_pts + extra_units * settings.unit_pts)


@dataclass(frozen=True)
class ScoreGateResult:
    current_score: int
    projected_clear_score: int
    minimum_score: int
    passed: bool
    shortfall: int
    should_retire: bool


def score_check_gate(settings: ScoreSettings, difficulty: int, counts: ScoreCounts,
                     minimum_score: int, final_area: int | None = None) -> ScoreGateResult:
    if final_area is None:
        final_area = max(settings.area_clear_pts)
    cleared = counts.area_ids(settings)
    current = base_score(settings, counts)
    pending = settings.boss_pts + settings.block_pts[final_area]
    if final_area not in cleared:
        pending += settings.area_clear_pts[final_area]
    projected = scale_pts(settings, difficulty, current + pending)
    shortfall = max(0, minimum_score - projected)
    return ScoreGateResult(scale_pts(settings, difficulty, current), projected, minimum_score,
                           shortfall == 0, shortfall, shortfall > 0)


REVIVE_RELIC_ID = 50002


@dataclass(frozen=True)
class ReviveState:
    has_relic: bool = False
    remaining_charges: int = 0
    used_count: int = 0

    @classmethod
    def from_relics(cls, relic_ids: Iterable[int], used_count: int = 0) -> ReviveState:
        if used_count < 0:
            raise ValueError("used_count must be nonnegative")
        present = REVIVE_RELIC_ID in relic_ids
        return cls(present, int(present and used_count == 0), used_count)

    def can_prevent_defeat(self, is_boss: bool) -> bool:
        return self.has_relic and self.remaining_charges > 0 and not is_boss

    def apply_battle_loss(self, is_boss: bool) -> ReviveState:
        if self.can_prevent_defeat(is_boss):
            return replace(self, remaining_charges=0, used_count=self.used_count + 1)
        return self
