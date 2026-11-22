"""Inventory and event choices in official settlement points, or threshold hits per attempt."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import exp, fsum, isclose, isfinite, sqrt
from typing import Protocol

from labyrinth_nn.catalog.schema import EventChoiceEntry, FeatureCatalog, MARK_THRESHOLDS, RelicEntry, ScoreSettings, UnitEntry
from .scoring import DEFAULT_SETTINGS, ReviveState
from .shop_dp import DEFAULT_SHELVES, ShopDiscountModel, ShopShelfModel, gold_to_pts_curve, shop_score_distribution


@dataclass(frozen=True)
class BattleExam:
    quest_id: int
    weight_pts: float
    required_power: float = 3.0
    is_boss: bool = False
    expected_time_sec: float = 0.0


@dataclass(frozen=True)
class PlanningState:
    difficulty: int = 5
    gold: int = 0
    owned_unit_ids: frozenset[int] = frozenset()
    relic_ids: tuple[int, ...] = ()
    mark_stacks: tuple[int, ...] = (0, 0, 0, 0, 0)
    remaining_battles: tuple[BattleExam, ...] = ()
    remaining_shops: int = 0
    confirmed_pts: int = 0
    score_remainder: int = 0  # hundredths, carried until final settlement
    revive_used_count: int = 0
    has_enhance_node_21: bool = True
    battle_power: float = 0.0
    element_bonus: float = 0.75  # five matching elements; configurable for a recorded team
    unidentified_units: int = 0
    unidentified_relics: int = 0
    future_refresh_available: tuple[bool, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "owned_unit_ids", frozenset(self.owned_unit_ids))
        for name in ("relic_ids", "mark_stacks", "remaining_battles", "future_refresh_available"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (len(self.mark_stacks) != 5 or min(self.gold, self.remaining_shops, self.unidentified_units,
                self.unidentified_relics, self.revive_used_count) < 0 or not 0 <= self.score_remainder < 100):
            raise ValueError("invalid mark stacks, wallet or shop count")

    @property
    def unit_count(self):
        return len(self.owned_unit_ids) + self.unidentified_units

    @property
    def relic_count(self):
        return len(self.relic_ids) + self.unidentified_relics

    @property
    def revive(self):
        return ReviveState.from_relics(self.relic_ids, self.revive_used_count)

    @property
    def discount(self):
        return ShopDiscountModel(self.has_enhance_node_21, 40003 in self.relic_ids)


@dataclass(frozen=True)
class PassEstimate:
    mean_prob: float
    std_prob: float
    lcb_prob: float


class BattlePassEstimator(Protocol):
    """Estimate one battle attempt. The planner accounts for the shared revive charge separately."""
    def estimate(self, state: PlanningState, exam: BattleExam) -> PassEstimate: ...


@dataclass(frozen=True)
class HeuristicBattlePassEstimator:
    """Saturating rule prior for sensitivity tests. Its uncertainty is a configurable heuristic."""
    catalog: FeatureCatalog | None = None
    lcb_z: float = 1.0

    def estimate(self, state: PlanningState, exam: BattleExam) -> PassEstimate:
        power = state.battle_power + min(state.unit_count, 60) * 0.10
        power += sum(sum(bonus for threshold, bonus in zip(MARK_THRESHOLDS, (0.65, 1.0, 1.35))
                         if n >= threshold) for n in state.mark_stacks)
        amp = 0.0
        for rid in state.relic_ids:
            rel = self.catalog.relics.get(rid) if self.catalog else None
            amp += max(float(rid == 30001), rel.element_bonus_amp if rel else 0.0)
            if rid in (50003, 50009) or (rel and rel.gold_scaling):
                power += 2.0 * min(state.gold, 10000) / 10000
            if rid == 50001 or (rel and rel.owned_unit_scaling):
                power += 1.5 * min(state.unit_count, 25) / 25
        power += 2.0 * state.element_bonus * (1.0 + amp)
        mean = 1 / (1 + exp(max(-60, min(60, exam.required_power - power))))
        if mean >= 0.995:
            return PassEstimate(1.0, 0.0, 1.0)
        std = 0.08 * sqrt(mean * (1 - mean))
        return PassEstimate(mean, std, max(0.0, mean - self.lcb_z * std))


def pass_profile(state: PlanningState, estimator: BattlePassEstimator) -> tuple[tuple[float, ...], float, float]:
    """Conditional pass LCBs, joint survival and expected fight time with one shared retry."""
    with_charge = float(state.revive.remaining_charges > 0)
    without = 1.0 - with_charge
    conditional = []
    elapsed = 0.0
    for exam in state.remaining_battles:
        p = estimator.estimate(state, exam).lcb_prob
        if not 0 <= p <= 1:
            raise ValueError("LCB probability outside [0, 1]")
        alive = with_charge + without
        retry_mass = 0.0 if exam.is_boss else with_charge * (1 - p)
        elapsed += (alive + retry_mass) * exam.expected_time_sec
        with_charge, without = with_charge * p, (without + retry_mass) * p
        conditional.append((with_charge + without) / alive if alive else 0.0)
    return tuple(conditional), with_charge + without, elapsed


def battle_value(state: PlanningState, estimator: BattlePassEstimator) -> float:
    probabilities, _, _ = pass_profile(state, estimator)
    return fsum(e.weight_pts * p for e, p in zip(state.remaining_battles, probabilities))


def award_collections(state: PlanningState, settings: ScoreSettings, *, unit_ids: Iterable[int] = (),
                      relic_ids: Iterable[int] = (), new_units: int = 0, new_relics: int = 0,
                      catalog: FeatureCatalog | None = None, entry: RelicEntry | None = None) -> PlanningState:
    owned = state.owned_unit_ids.union(unit_ids)
    relics = tuple(rid for rid in dict.fromkeys(relic_ids) if rid not in state.relic_ids)
    marks = list(state.mark_stacks)
    for rid in relics:
        rel = entry if entry and entry.relic_id == rid else (catalog.relics.get(rid) if catalog else None)
        if rel and 1 <= rel.relic_mark_id <= 5:
            marks[rel.relic_mark_id - 1] += rel.relic_mark_count
    base = ((len(owned) - len(state.owned_unit_ids) + new_units) * settings.unit_pts
            + (len(relics) + new_relics) * settings.relic_pts)
    delta, remainder = divmod(state.score_remainder + base * settings.score_rates[state.difficulty], 100)
    return replace(state, owned_unit_ids=owned, relic_ids=state.relic_ids + relics,
                   unidentified_units=state.unidentified_units + new_units,
                   unidentified_relics=state.unidentified_relics + new_relics, mark_stacks=tuple(marks),
                   confirmed_pts=state.confirmed_pts + delta, score_remainder=remainder)


@dataclass(frozen=True)
class EventBranch:
    probability: float = 1.0
    gold_delta: int = 0
    next_event_id: int | None = None
    unit_ids: tuple[int, ...] = ()
    relic_ids: tuple[int, ...] = ()
    new_units: int = 0
    new_relics: int = 0
    elapsed_sec: float = 0.0


@dataclass(frozen=True)
class EventOption:
    choice_id: int
    branches: tuple[EventBranch, ...] = (EventBranch(),)
    condition_value: int = 0
    time_sec: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "branches", tuple(self.branches))
        if (not isclose(fsum(b.probability for b in self.branches), 1.0, abs_tol=1e-12)
                or self.condition_value < 0 or not isfinite(self.time_sec) or self.time_sec < 0 or
                any(not 0 <= b.probability <= 1 or not isfinite(b.elapsed_sec) or min(b.elapsed_sec, b.new_units, b.new_relics) < 0
                    for b in self.branches)):
            raise ValueError(f"invalid event choice {self.choice_id}")


def default_event_tree(normal_win_prob: float = 1.0, ex_win_prob: float = 1.0) -> dict[int, tuple[EventOption, ...]]:
    """Protocol choice ids. Anonymous collections can be replaced with recorded identities."""
    tree = {}
    for family in (14, 15, 16):
        for stage in (1, 2, 3):
            event = stage * 1000 + 300 + family
            next_id = event + 1000 if stage < 3 else None
            success = (0.5, 0.25, 0.10)[stage - 1] if family == 14 else 0.5
            prize = (2000, 4000, 8000)[stage - 1] if family == 16 else 0
            cost = (0, 2000, 4000)[stage - 1] if family == 16 else 0
            win = EventBranch(success, prize, next_id, new_relics=int(family == 14),
                              new_units=2 if family == 15 and stage == 3 else 0)
            play = EventOption(event * 10 + 1, (win, EventBranch(1 - success, 500)), cost, 1.0)
            options = [play]
            if family in (15, 16) and stage > 1:
                options.append(EventOption(event * 10 + 2, (EventBranch(new_units=2 if family == 15 else 0),)))
            tree[event] = tuple(options)
    tree[1107] = (EventOption(11071, (EventBranch(new_relics=1),), condition_value=2000),
                  EventOption(11072, (EventBranch(new_units=1),), condition_value=2000),
                  EventOption(11073, (EventBranch(.5, 1500), EventBranch(.5, 2000))))
    for event, normal, ex in ((1105, 200, 300), (1109, 300, 450), (1113, 400, 600), (1117, 500, 750)):
        tree[event] = tuple(EventOption(event * 10 + i,
                            (EventBranch(p, coins), EventBranch(1 - p, 500)), time_sec=1.0)
                            for i, (p, coins) in enumerate(((normal_win_prob, normal), (ex_win_prob, ex)), 1))
    return tree


@dataclass(frozen=True)
class EventLeaf:
    probability: float
    state: PlanningState
    elapsed_sec: float
    choices: tuple[int, ...]
    branch_indices: tuple[int, ...]


@dataclass(frozen=True)
class EvaluatedCandidate:
    candidate_id: int
    value: float
    direct_pts: float
    shop_pts: float
    battle_pts: float
    next_state: PlanningState | None
    prob_reach_target: float = 0.0
    expected_time_sec: float = 0.0
    outcomes: tuple[EventLeaf, ...] = ()


ShopCurve = Callable[[PlanningState], float]
PointDistribution = Mapping[int, float]


def _wallet_value(state, settings, shelf_model, shop_curve):
    if shop_curve is not None:
        return shop_curve(state)
    return gold_to_pts_curve(state.gold, state.remaining_shops, settings, state.difficulty,
                             discount_model=state.discount, shelf_model=shelf_model,
                             owned_unit_ids=state.owned_unit_ids, score_remainder=state.score_remainder)


def _rank(candidates):
    return sorted(candidates, key=lambda c: (-c.value, c.expected_time_sec, c.candidate_id))


def _collection_choices(state, candidates, estimator, settings, shelf_model, shop_curve, catalog, relic):
    if catalog and isinstance(estimator, HeuristicBattlePassEstimator) and estimator.catalog is None:
        estimator = replace(estimator, catalog=catalog)
    before_battle = battle_value(state, estimator)
    before_shop = _wallet_value(state, settings, shelf_model, shop_curve)
    result = []
    for item in candidates:
        if relic:
            entry = item if isinstance(item, RelicEntry) else None
            ident = entry.relic_id if entry else int(item)
            after = award_collections(state, settings, relic_ids=(ident,), catalog=catalog, entry=entry)
        else:
            ident = item.unit_id if isinstance(item, UnitEntry) else int(item)
            after = award_collections(state, settings, unit_ids=(ident,))
        direct = after.confirmed_pts - state.confirmed_pts
        shop = _wallet_value(after, settings, shelf_model, shop_curve) - before_shop
        battle = battle_value(after, estimator) - before_battle
        result.append(EvaluatedCandidate(ident, direct + shop + battle, direct, shop, battle, after))
    return _rank(result)


def evaluate_role_choices(state: PlanningState, candidates: Iterable[int | UnitEntry],
                          estimator: BattlePassEstimator, *, settings: ScoreSettings = DEFAULT_SETTINGS,
                          shelf_model: ShopShelfModel = DEFAULT_SHELVES, shop_curve: ShopCurve | None = None):
    return _collection_choices(state, candidates, estimator, settings, shelf_model, shop_curve, None, False)


def evaluate_relic_choices(state: PlanningState, candidates: Iterable[int | RelicEntry],
                           estimator: BattlePassEstimator, catalog: FeatureCatalog | None = None, *,
                           settings: ScoreSettings = DEFAULT_SETTINGS, shelf_model: ShopShelfModel = DEFAULT_SHELVES,
                           shop_curve: ShopCurve | None = None):
    return _collection_choices(state, candidates, estimator, settings, shelf_model, shop_curve, catalog, True)


def evaluate_event_options(state: PlanningState, options: int | Sequence[EventOption | EventChoiceEntry],
                           estimator: BattlePassEstimator, deduct_condition_principal: bool | None = None, *,
                           event_tree: Mapping[int, Sequence[EventOption]] | None = None,
                           settings: ScoreSettings = DEFAULT_SETTINGS, catalog: FeatureCatalog | None = None,
                           shelf_model: ShopShelfModel = DEFAULT_SHELVES, shop_curve: ShopCurve | None = None,
                           shop_distribution_fn: Callable[[PlanningState], PointDistribution] | None = None,
                           future_score_distribution: PointDistribution | None = None,
                           target_score: int = 0, lambda_rate: float = 0.0,
                           first_continuation: bool = False) -> list[EvaluatedCandidate]:
    """Expand observations before choosing each continuation; rank feasible root choices.

    future_score_distribution is the remaining non-shop, non-event score *conditional on clear*.
    By default shop payouts use their full PMF. A custom scalar curve alone specifies a
    deterministic payout scenario. Supply shop_distribution_fn for uncertain custom payouts.
    """
    tree = default_event_tree()
    if event_tree:
        tree.update(event_tree)
    deduct = True if deduct_condition_principal is None else deduct_condition_principal
    if lambda_rate < 0:
        raise ValueError("negative time price")
    future = {0: 1.0} if future_score_distribution is None else future_score_distribution

    def choices_at(event_id, s):
        rules = tree[event_id]
        if event_id in (1105, 1109, 1113, 1117) and event_id not in (event_tree or {}):
            calibrated = []
            for rule, power in zip(rules, (1.0, 2.5)):
                p = estimator.estimate(s, BattleExam(rule.choice_id, 0, power)).lcb_prob
                win, loss = rule.branches
                calibrated.append(replace(rule, branches=(replace(win, probability=p), replace(loss, probability=1-p))))
            rules = calibrated
        return rules

    def catalog_rule(entry):
        # The catalog contains result ids, not joint outcome probabilities. Resolve these
        # through the protocol rules or a supplied empirical tree before pricing a choice.
        for rule in choices_at(entry.event_id, state):
            if rule.choice_id == entry.choice_id:
                return replace(rule, condition_value=entry.condition_value if entry.condition_type == 1 else 0)
        raise ValueError(f"missing outcome rule for catalog event choice {entry.choice_id}")

    def checked(pmf):
        if not isclose(fsum(pmf.values()), 1.0, abs_tol=1e-12) or any(not 0 <= p <= 1 for p in pmf.values()):
            raise ValueError("event forecast PMF must sum to one")
        return pmf

    checked(future)

    def wallet(s):
        if shop_distribution_fn:
            return fsum(v * p for v, p in checked(shop_distribution_fn(s)).items())
        return _wallet_value(s, settings, shelf_model, shop_curve)

    initial_shop = wallet(state)
    initial_battle = battle_value(state, estimator)

    def evaluate_leaves(ident, leaves):
        direct = shop = battle = hit = time = 0.0
        for leaf in leaves:
            s, w = leaf.state, leaf.probability
            direct += w * (s.confirmed_pts - state.confirmed_pts)
            balance = wallet(s)
            shop += w * (balance - initial_shop)
            passes, survival, fight_time = pass_profile(s, estimator)
            battle += w * (fsum(e.weight_pts * p for e, p in zip(s.remaining_battles, passes)) - initial_battle)
            time += w * (leaf.elapsed_sec + fight_time)
            if target_score > 0:
                if shop_distribution_fn:
                    payouts = checked(shop_distribution_fn(s))
                elif shop_curve:
                    payouts = {balance: 1.0}
                else:
                    payouts = shop_score_distribution(s.gold, s.remaining_shops, settings, s.difficulty,
                                                       discount_model=s.discount, shelf_model=shelf_model,
                                                       owned_unit_ids=s.owned_unit_ids, score_remainder=s.score_remainder)
                hit += w * survival * fsum(p * q for pts, p in payouts.items() for extra, q in future.items()
                                          if s.confirmed_pts + pts + extra >= target_score)
        value = hit - lambda_rate * time if target_score > 0 else direct + shop + battle
        return EvaluatedCandidate(ident, value, direct, shop, battle,
                                  leaves[0].state if len(leaves) == 1 else None, hit, time, tuple(leaves))

    def expand(s, choices, elapsed, history, branch_path, visiting):
        ranked = []
        for option in choices:
            if s.gold < option.condition_value:
                continue
            leaves = []
            for index, branch in enumerate(option.branches):
                if not branch.probability:
                    continue
                gold = s.gold - (option.condition_value if deduct else 0) + branch.gold_delta
                after = award_collections(replace(s, gold=gold), settings, unit_ids=branch.unit_ids,
                                          relic_ids=branch.relic_ids, new_units=branch.new_units,
                                          new_relics=branch.new_relics, catalog=catalog)
                time = elapsed + option.time_sec + branch.elapsed_sec
                path = (*history, option.choice_id)
                indices = (*branch_path, index)
                child = branch.next_event_id
                if child is None:
                    leaves.append(EventLeaf(branch.probability, after, time, path, indices))
                    continue
                if child in visiting:
                    raise ValueError(f"cycle in event chain at {child}")
                continuations = expand(after, choices_at(child, after), time, path, indices, visiting | {child})
                if continuations:
                    chosen = continuations[0] if first_continuation else _rank(continuations)[0]
                    leaves.extend(replace(o, probability=o.probability * branch.probability) for o in chosen.outcomes)
                else:
                    leaves.append(EventLeaf(branch.probability, after, time, path, indices))
            ranked.append(evaluate_leaves(option.choice_id, leaves))
        return ranked

    roots = choices_at(options, state) if isinstance(options, int) else [
        catalog_rule(option) if isinstance(option, EventChoiceEntry) else option for option in options]
    return _rank(expand(state, roots, 0.0, (), (), {options} if isinstance(options, int) else set()))
