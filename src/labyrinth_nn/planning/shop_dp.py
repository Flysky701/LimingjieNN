"""Observed-shelf purchase decisions and the distribution of later shop settlements."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from math import fsum, isclose, isfinite

from labyrinth_nn.catalog.schema import ScoreSettings
from .scoring import DEFAULT_SETTINGS, relic_delta_pts, unit_delta_pts


@dataclass(frozen=True)
class ShopDiscountModel:
    has_enhance_node_21: bool = True
    has_relic_40003: bool = False
    extra_discount_pct: int = 0

    @property
    def discount_pct(self) -> int:
        return 25 * self.has_enhance_node_21 + 10 * self.has_relic_40003 + self.extra_discount_pct

    def price(self, base_price: int) -> int:
        return base_price * (100 - min(90, self.discount_pct)) // 100


class ShopItemKind(StrEnum):
    RANDOM_IMPRINT = "RANDOM_IMPRINT"
    CHOICE_IMPRINT = "CHOICE_IMPRINT"
    RELIC_1STAR = "RELIC_1STAR"
    RELIC_2STAR = "RELIC_2STAR"
    RELIC_3STAR = "RELIC_3STAR"
    RELIC_4STAR = "RELIC_4STAR"


BASE_PRICES = dict(zip(ShopItemKind, (1200, 2800, 1800, 2800, 3800, 9999), strict=True))


@dataclass(frozen=True)
class ShopItem:
    kind: ShopItemKind | str
    relic_id: int | None = None
    extra_battle_pts: float = 0.0
    unit_id: int | None = None
    # For a recorded imprint offering an already-owned unit.
    collection_count: int = 1

    def __post_init__(self):
        object.__setattr__(self, "kind", ShopItemKind(self.kind))
        if self.collection_count < 0 or not isfinite(self.extra_battle_pts):
            raise ValueError("invalid shop reward")

    @property
    def base_price(self) -> int:
        return BASE_PRICES[self.kind]

    @property
    def is_relic(self) -> bool:
        return self.kind not in (ShopItemKind.RANDOM_IMPRINT, ShopItemKind.CHOICE_IMPRINT)


Shelf = tuple[ShopItem, ...]
RetentionValue = Callable[[int], float]


@dataclass(frozen=True)
class ShopShelfModel:
    """Empirical distribution over WHOLE shelves; lineup correlations are preserved.

    Supply counted replay observations for calibration. The two default shelves are
    example priors. labyrinth_coin_shop_lineup describes a different shop.
    """
    shelves: tuple[tuple[float, Shelf], ...] = (
        (0.5, (ShopItem("RANDOM_IMPRINT"), ShopItem("RELIC_1STAR"))),
        (0.5, (ShopItem("CHOICE_IMPRINT"), ShopItem("RELIC_2STAR"))),
    )

    def __post_init__(self):
        shelves = tuple((float(p), tuple(items)) for p, items in self.shelves)
        if not isclose(fsum(p for p, _ in shelves), 1.0, abs_tol=1e-12) or any(
                not 0 <= p <= 1 for p, _ in shelves):
            raise ValueError("shelf probabilities must sum to one")
        object.__setattr__(self, "shelves", tuple((p, s) for p, s in shelves if p))

    @classmethod
    def from_observations(cls, observations: Iterable[Iterable[ShopItem]]) -> ShopShelfModel:
        frequencies = Counter(tuple(s) for s in observations)
        total = frequencies.total()
        if not total:
            raise ValueError("no shelf observations")
        return cls(tuple((n / total, shelf) for shelf, n in frequencies.items()))


DEFAULT_DISCOUNT = ShopDiscountModel()
DEFAULT_SHELVES = ShopShelfModel()


@dataclass(frozen=True)
class ShopPlanResult:
    purchases: Shelf
    purchased_indices: tuple[int, ...]
    gold_remaining: int  # after visible purchases, before paying the chosen refresh fee
    should_refresh: bool
    refresh_cost: int
    expected_pts: float  # expected official collection points across all planned visits
    expected_value: float  # additionally includes tactical premiums and terminal retention


def _efficiency(item, unit_pts, relic_pts, discount):
    effective = item.collection_count * (relic_pts if item.is_relic else unit_pts) + item.extra_battle_pts
    return discount.price(item.base_price) / effective if effective > 0 else float("inf")


def clearance_ladder(items: Iterable[ShopItem], gold: int, settings: ScoreSettings = DEFAULT_SETTINGS,
                     difficulty: int = 5, *, discount_model: ShopDiscountModel = DEFAULT_DISCOUNT,
                     coin_retention_value_fn: RetentionValue | None = None) -> Shelf:
    """Greedy clearance in ascending coins per effective point, with terminal coin protection."""
    u, r = unit_delta_pts(settings, difficulty), relic_delta_pts(settings, difficulty)
    bought = []
    for item in sorted(items, key=lambda it: _efficiency(it, u, r, discount_model)):
        price = discount_model.price(item.base_price)
        gain = item.collection_count * (r if item.is_relic else u) + item.extra_battle_pts
        if price <= gold:
            opportunity = coin_retention_value_fn(gold) - coin_retention_value_fn(gold - price) if coin_retention_value_fn else 0
            if gain > opportunity:
                bought.append(item)
                gold -= price
    return tuple(bought)


class _ShopDP:
    """One immutable economic configuration. Caches are scoped to its lifetime."""

    def __init__(self, unit_ticks, relic_ticks, fees, discount, model, limit, retention, refresh_flags):
        self.fees = fees
        self.discount = discount
        self.model = model
        self.unit_pts = unit_ticks / 100
        self.relic_pts = relic_ticks / 100

        def may_refresh(shops_left):
            return refresh_flags[-shops_left] if shops_left <= len(refresh_flags) else True

        def acquire(item, owned, remainder):
            count = item.collection_count
            if not item.is_relic and item.unit_id is not None and count:
                count -= int(item.unit_id in owned)
                owned = owned | {item.unit_id}
            gain, remainder = divmod(remainder + count * (relic_ticks if item.is_relic else unit_ticks), 100)
            return count, gain, owned, remainder

        @lru_cache(maxsize=65536)
        def future(gold, shops, owned, remainder):
            if shops == 0:
                return retention(gold) if retention else 0.0
            return fsum(p * visible(shelf, gold, shops, 0, 0, may_refresh(shops), owned, remainder)[0]
                        for p, shelf in model.shelves)

        @lru_cache(maxsize=65536)
        def visible(shelf, gold, shops, refresh, mask, allow, owned, remainder):
            # Exit can carry all remaining money into another shop, or the remaining fights.
            best, action = future(gold, shops - 1, owned, remainder), -1
            for i, it in enumerate(shelf):
                if mask & (1 << i):
                    continue
                price = discount.price(it.base_price)
                if price > gold:
                    continue
                _, gain, after, rem = acquire(it, owned, remainder)
                value = gain + it.extra_battle_pts + visible(shelf, gold - price, shops, refresh,
                                                             mask | (1 << i), allow, after, rem)[0]
                if value > best:
                    best, action = value, i
            if allow and refresh < min(limit, len(fees)) and fees[refresh] <= gold:
                value = fsum(p * visible(s, gold - fees[refresh], shops, refresh + 1, 0, True, owned, remainder)[0]
                             for p, s in model.shelves)
                if value > best:
                    best, action = value, -2
            return best, action

        @lru_cache(maxsize=16384)
        def future_pmf(gold, shops, owned, remainder):
            if not shops:
                return ((0, 1.0),)
            return mix((p, visible_pmf(s, gold, shops, 0, 0, may_refresh(shops), owned, remainder))
                       for p, s in model.shelves)

        def mix(components):
            mass = defaultdict(float)
            for weight, pmf in components:
                for score, prob in pmf:
                    mass[score] += weight * prob
            return tuple(mass.items())

        @lru_cache(maxsize=32768)
        def checkout(shelf, gold, shops, refresh, mask, allow, owned, remainder):
            _, action = visible(shelf, gold, shops, refresh, mask, allow, owned, remainder)
            if action == -1:
                return (((0, 0, gold, owned, remainder), 1.0),)
            if action == -2:
                wallet = gold - fees[refresh]
                return mix((p, checkout(s, wallet, shops, refresh + 1, 0, True, owned, remainder))
                           for p, s in model.shelves)
            item = shelf[action]
            count, _, after, rem = acquire(item, owned, remainder)
            pmf = checkout(shelf, gold - discount.price(item.base_price), shops, refresh,
                           mask | (1 << action), allow, after, rem)
            units, relics = (0, count) if item.is_relic else (count, 0)
            return tuple(((u + units, r + relics, coins, ids, frac), p) for (u, r, coins, ids, frac), p in pmf)

        @lru_cache(maxsize=16384)
        def visible_pmf(shelf, gold, shops, refresh, mask, allow, owned, remainder):
            mass = defaultdict(float)
            for (u, r, wallet, ids, rem), p in checkout(shelf, gold, shops, refresh, mask, allow, owned, remainder):
                gained = (remainder + u * unit_ticks + r * relic_ticks) // 100
                for later, q in future_pmf(wallet, shops - 1, ids, rem):
                    mass[gained + later] += p * q
            return tuple(mass.items())

        self.acquire = acquire
        self.checkout = checkout
        self.visible = visible
        self.visible_pmf = visible_pmf
        self.future_pmf = future_pmf

        @lru_cache(maxsize=16384)
        def curve(gold, shops, owned, remainder):
            return fsum(s * p for s, p in future_pmf(gold, shops, owned, remainder))
        self.curve = curve

_cached_solver = lru_cache(maxsize=32)(_ShopDP)


def _solver(settings, difficulty, discount, model, limit, retention=None, refresh_flags=()):
    # Snapshot the mutable ScoreSettings by value so configuration edits invalidate the cache.
    fees = tuple(settings.shop_refresh_costs)
    if limit < 0 or any(f < 0 for f in fees):
        raise ValueError("negative refresh limit or cost")
    factory = _cached_solver if retention is None else _ShopDP
    rate = settings.score_rates[difficulty]
    return factory(settings.unit_pts * rate, settings.relic_pts * rate,
                   fees, discount, model, limit, retention, tuple(refresh_flags))


def plan_shop_visit(items: Iterable[ShopItem], gold: int, settings: ScoreSettings = DEFAULT_SETTINGS,
                    difficulty: int = 5, *, discount_model: ShopDiscountModel = DEFAULT_DISCOUNT,
                    shelf_model: ShopShelfModel = DEFAULT_SHELVES, refresh_count: int = 0,
                    max_refreshes: int | None = None, refresh_available: bool = True,
                    is_final_area: bool = False, remaining_shops: int = 1, bought_mask: int = 0,
                    coin_retention_value_fn: RetentionValue | None = None,
                    owned_unit_ids: Iterable[int] = (), score_remainder: int = 0,
                    future_refresh_available: tuple[bool, ...] = ()) -> ShopPlanResult:
    """remaining_shops includes this one. All remaining visits in the final area have dark buttons."""
    shelf = tuple(items)
    if min(gold, refresh_count, bought_mask) < 0 or remaining_shops < 1 or bought_mask >= 1 << len(shelf):
        raise ValueError("invalid visible shop state")
    limit = len(settings.shop_refresh_costs) if max_refreshes is None else max_refreshes
    flags = (False,) * (remaining_shops - 1) if is_final_area else future_refresh_available
    dp = _solver(settings, difficulty, discount_model, shelf_model, limit, coin_retention_value_fn, flags)
    allow = refresh_available and not is_final_area
    owned = frozenset(owned_unit_ids) | {it.unit_id for i, it in enumerate(shelf) if bought_mask & (1 << i)
                                         and it.unit_id is not None and not it.is_relic}
    args = (shelf, gold, remaining_shops, refresh_count, bought_mask, allow, owned, score_remainder)
    value, action = dp.visible(*args)
    expected = fsum(s * p for s, p in dp.visible_pmf(*args))
    indices = []
    while action >= 0:
        indices.append(action)
        gold -= discount_model.price(shelf[action].base_price)
        bought_mask |= 1 << action
        _, _, owned, score_remainder = dp.acquire(shelf[action], owned, score_remainder)
        _, action = dp.visible(shelf, gold, remaining_shops, refresh_count, bought_mask, allow, owned, score_remainder)
    indices.sort(key=lambda i: _efficiency(shelf[i], dp.unit_pts, dp.relic_pts, discount_model))
    return ShopPlanResult(tuple(shelf[i] for i in indices), tuple(indices), gold, action == -2,
                          dp.fees[refresh_count] if action == -2 else 0, expected, value)


def gold_to_pts_curve(gold: int, remaining_shops: int, settings: ScoreSettings = DEFAULT_SETTINGS,
                      difficulty: int = 5, *, discount_model: ShopDiscountModel = DEFAULT_DISCOUNT,
                      shelf_model: ShopShelfModel = DEFAULT_SHELVES, max_refreshes: int = 0,
                      owned_unit_ids: Iterable[int] = (), score_remainder: int = 0) -> float:
    if min(gold, remaining_shops) < 0:
        raise ValueError("negative gold or remaining shop count")
    if not remaining_shops:
        return 0.0
    return _solver(settings, difficulty, discount_model, shelf_model, max_refreshes).curve(
        gold, remaining_shops, frozenset(owned_unit_ids), score_remainder)


def shop_score_distribution(gold: int, remaining_shops: int, settings: ScoreSettings = DEFAULT_SETTINGS,
                            difficulty: int = 5, *, discount_model: ShopDiscountModel = DEFAULT_DISCOUNT,
                            shelf_model: ShopShelfModel = DEFAULT_SHELVES, max_refreshes: int = 0,
                            owned_unit_ids: Iterable[int] = (), score_remainder: int = 0) -> dict[int, float]:
    """Settlement PMF under the same purchase policy used by f_shop, for threshold decisions."""
    if min(gold, remaining_shops) < 0:
        raise ValueError("negative gold or remaining shop count")
    return dict(_solver(settings, difficulty, discount_model, shelf_model, max_refreshes).future_pmf(
        gold, remaining_shops, frozenset(owned_unit_ids), score_remainder))


def shop_visit_distribution(gold: int, remaining_shops: int, settings: ScoreSettings, difficulty: int, *,
                            discount_model: ShopDiscountModel, shelf_model: ShopShelfModel, refresh_available: bool,
                            max_refreshes: int | None = None, owned_unit_ids: Iterable[int] = (), score_remainder: int = 0,
                            coin_retention_value_fn: RetentionValue | None = None,
                            future_refresh_available: tuple[bool, ...] = ()) -> dict[tuple[int, int, int], float]:
    """Joint (new units, acquired relics, wallet on exit) for exactly the current unknown shelf."""
    limit = len(settings.shop_refresh_costs) if max_refreshes is None else max_refreshes
    dp = _solver(settings, difficulty, discount_model, shelf_model, limit, coin_retention_value_fn, future_refresh_available)
    result = defaultdict(float)
    for prob, shelf in shelf_model.shelves:
        pmf = dp.checkout(shelf, gold, remaining_shops, 0, 0, refresh_available, frozenset(owned_unit_ids), score_remainder)
        for (units, relics, wallet, _, _), chance in pmf:
            result[units, relics, wallet] += prob * chance
    return dict(result)


def marginal_gold_pts(gold_before: int, delta_gold: int, remaining_shops: int,
                      settings: ScoreSettings = DEFAULT_SETTINGS, difficulty: int = 5, **kwargs) -> float:
    return (gold_to_pts_curve(gold_before + delta_gold, remaining_shops, settings, difficulty, **kwargs)
            - gold_to_pts_curve(gold_before, remaining_shops, settings, difficulty, **kwargs))
