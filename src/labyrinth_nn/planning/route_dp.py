"""Finite-horizon route planning on recorded map edges, with observable score and revive state."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import cache
from graphlib import CycleError, TopologicalSorter
from math import fsum, gcd, isclose, isfinite

from labyrinth_nn.catalog.schema import ScoreSettings
from .scoring import ScoreCounts, base_score, scale_pts, score_gcd


class BlockType(IntEnum):
    # Wire values checked against LabyrinthNodeTypes.kt and BilibiliLabyrinthApi.kt.
    UNKNOWN = 0
    START = 1
    NORMAL_BATTLE = 2
    EX_BATTLE = 3
    LINK = 4
    ROLE_REWARD = 4
    EVENT = 5
    RELIC = 6
    SHOP = 7
    BOSS = 8
    CLEAR = -1


@dataclass(frozen=True)
class MapNode:
    block_id: int
    area: int
    column: int
    row: int
    block_type: int
    quest_id: int | None = None
    next_block_ids: tuple[int, ...] = ()
    is_area_last_point: bool = False

    def __post_init__(self):
        object.__setattr__(self, "next_block_ids", tuple(self.next_block_ids))


@dataclass(frozen=True)
class NodeOutcomeSpec:
    """Reward on success, in final points (integer or point-to-probability PMF).

    Times are conditional on the result of one try. Gold is a replay resource,
    never automatically added as points. A supplied spec replaces the whole node.
    """
    win_prob: float = 1.0
    gold_delta: int = 0
    direct_pts_buckets: int | dict[int, float] = 0
    expected_time_sec: float = 0.0
    fail_time_sec: float | None = None
    is_boss: bool | None = None
    is_non_boss_battle: bool | None = None


@dataclass(frozen=True)
class _OfficialOutcome(NodeOutcomeSpec):
    # Keep the fractional part of default rewards until the one final settlement floor.
    remainder: int = 0


@dataclass
class RouteGraph:
    nodes: dict[int, MapNode]
    children: dict[int, tuple[int, ...]]
    starts: tuple[int, ...]
    order: tuple[int, ...]
    area_ends: set[int]
    clear_ends: set[int]


def build_route_graph(nodes: Iterable[MapNode]) -> RouteGraph:
    by_id = {}
    areas = defaultdict(list)
    for node in nodes:
        if node.block_id in by_id:
            raise ValueError(f"duplicate block_id {node.block_id}")
        by_id[node.block_id] = node
        areas[node.area].append(node)
    if not by_id:
        raise ValueError("empty map")
    floors = sorted(areas)
    starts, ends, children, clear_ends = {}, set(), {}, set()
    for area in floors:
        group = areas[area]
        first = min(n.column for n in group)
        last = max((n.column for n in group if n.block_type not in (BlockType.CLEAR, BlockType.UNKNOWN)), default=first)
        starts[area] = tuple(n.block_id for n in group if n.column == first)
        ends.update(n.block_id for n in group if n.block_type not in (BlockType.CLEAR, BlockType.UNKNOWN) and
                    (n.is_area_last_point or n.column == last))
    parents = {i: [] for i in by_id}
    for index, area in enumerate(floors):
        for n in areas[area]:
            edges = [i for i in n.next_block_ids if i != -1 or i in by_id]
            for child in edges:
                if child not in by_id:
                    raise ValueError(f"missing block {child} referenced by {n.block_id}")
            crosses_area = any(by_id[i].area != area for i in edges)
            if crosses_area:
                ends.add(n.block_id)
            if index + 1 < len(floors) and n.block_id in ends and not crosses_area:
                edges.extend(starts[floors[index + 1]])
            children[n.block_id] = tuple(dict.fromkeys(edges))
            for child in children[n.block_id]:
                parents[child].append(n.block_id)
            if not edges and area == floors[-1] and (
                n.block_id in ends or n.block_type == BlockType.CLEAR
            ):
                clear_ends.add(n.block_id)
    try:
        order = tuple(TopologicalSorter(parents).static_order())
    except CycleError as exc:
        raise ValueError("cyclic labyrinth map") from exc
    return RouteGraph(by_id, children, starts[floors[0]], order, ends, clear_ends)


def node_score_counts(node: MapNode, area_end: bool) -> ScoreCounts:
    """Known default collections assume a new role on LINK and one relic on RELIC."""
    kind = node.block_type
    if kind in (BlockType.CLEAR, BlockType.UNKNOWN):
        return ScoreCounts()
    return ScoreCounts((node.area,) if area_end else (), (node.area,),
                       int(kind == BlockType.LINK), int(kind == BlockType.RELIC),
                       int(kind == BlockType.NORMAL_BATTLE), int(kind == BlockType.EX_BATTLE),
                       int(kind == BlockType.BOSS))


def default_node_outcome_specs(nodes: Iterable[MapNode] | RouteGraph, settings: ScoreSettings,
                               difficulty: int, *, win_probs: Mapping[int, float] | None = None,
                               times_sec: Mapping[int, float] | None = None,
                               reward_units: Mapping[int, int] | None = None,
                               reward_relics: Mapping[int, int] | None = None,
                               only_ids: Iterable[int] | None = None) -> dict[int, NodeOutcomeSpec]:
    graph = nodes if isinstance(nodes, RouteGraph) else build_route_graph(nodes)
    specs = {}
    for i in graph.nodes if only_ids is None else only_ids:
        node = graph.nodes[i]
        counts = node_score_counts(node, i in graph.area_ends)
        counts = replace(counts, units=counts.units + (reward_units or {}).get(i, 0),
                         relics=counts.relics + (reward_relics or {}).get(i, 0))
        base = base_score(settings, counts)
        t = node.block_type
        battle = t in (BlockType.NORMAL_BATTLE, BlockType.EX_BATTLE, BlockType.BOSS)
        # Timing defaults are planning priors. Recorded conditional times can override them.
        seconds = 30.0 if battle else float(t != BlockType.CLEAR)
        specs[i] = _OfficialOutcome(
            win_prob=(win_probs or {}).get(i, 1.0), expected_time_sec=(times_sec or {}).get(i, seconds),
            direct_pts_buckets=scale_pts(settings, difficulty, base),
            remainder=base * settings.score_rates[difficulty] % 100)
    return specs


def battle_flags(node: MapNode, spec: NodeOutcomeSpec) -> tuple[bool, bool]:
    boss = node.block_type == BlockType.BOSS if spec.is_boss is None else spec.is_boss
    nonboss = node.block_type in (BlockType.NORMAL_BATTLE, BlockType.EX_BATTLE)
    if spec.is_non_boss_battle is not None:
        nonboss = spec.is_non_boss_battle
    return boss, nonboss and not boss


@dataclass
class RouteDPResult:
    best_route: list[MapNode]
    score_distribution: dict[int, float]
    expected_score: float
    clear_prob: float
    prob_reach_target: float
    expected_time_sec: float
    hard_upper_bound: int
    num_buckets: int
    bucket_size: float
    # Choice AFTER the node yields a reward, keyed by (block, score bucket, remaining charge).
    policy: dict[tuple, int]


@dataclass(frozen=True)
class WalletOutcome:
    """Joint incremental collection and wallet after a shop/event, conditional on node success."""
    extra_base_pts: int
    gold_after: int
    probability: float
    elapsed_sec: float = 0.0


# Node, score after its direct reward, carried hundredth, charge and post-node gold.
WalletModel = Callable[[MapNode, int, int, bool, int], Sequence[WalletOutcome]]


@dataclass(frozen=True)
class _Metrics:
    score: float = 0.0
    clear: float = 0.0
    hit: float = 0.0
    time: float = 0.0
    upper: int = -1  # highest terminal bucket on ANY feasible continuation; -1 means no clear


def solve_route_dp(nodes: Iterable[MapNode] | RouteGraph, settings: ScoreSettings, difficulty: int,
                   outcome_specs: Mapping[int, NodeOutcomeSpec] | None = None,
                   target_score: int = 0, lambda_rate: float = 0.0,
                   revive_available: bool = False, *, confirmed_pts: int = 0,
                   score_remainder: int = 0, start_node_ids: Iterable[int] | None = None,
                   wallet_model: WalletModel | None = None, initial_gold: int = 0) -> RouteDPResult:
    """Reverse-topological Bellman DP on reachable buckets, then forward probability propagation.

    Wipes settle at zero. A prevented defeat spends the charge and retries the same battle
    once. best_route is one representative successful trajectory of the adaptive policy.
    score_remainder is the carried hundredth of a point from a partial official settlement.
    The optional replay wallet model carries a fourth key (coins) when shop/event rewards
    depend on the path's spending history. It is absent from ordinary three-key policies.
    """
    if lambda_rate < 0 or not isfinite(lambda_rate) or not 0 <= score_remainder < 100:
        raise ValueError("invalid time price or partial score remainder")
    graph = nodes if isinstance(nodes, RouteGraph) else build_route_graph(nodes)
    specs = dict(outcome_specs or {})
    if specs.keys() - graph.nodes.keys():
        raise ValueError("outcome spec references an unknown block")
    specs.update(default_node_outcome_specs(graph, settings, difficulty, only_ids=graph.nodes.keys() - specs.keys()))
    starts = graph.starts if start_node_ids is None else tuple(start_node_ids)
    if not starts or any(i not in graph.nodes for i in starts):
        raise ValueError("supply at least one valid entry block")

    ticks = max(1, score_gcd(settings, difficulty)) * 100
    rate = settings.score_rates[difficulty]
    if wallet_model is not None:
        ticks = gcd(ticks, settings.unit_pts * rate, settings.relic_pts * rate)
    rewards = {}
    for i, spec in specs.items():
        raw = spec.direct_pts_buckets
        pmf = {raw: 1.0} if isinstance(raw, int) else raw
        times = (spec.expected_time_sec, spec.fail_time_sec)
        if (not 0 <= spec.win_prob <= 1 or any(t is not None and (not isfinite(t) or t < 0) for t in times)
                or not isclose(fsum(pmf.values()), 1, abs_tol=1e-12)
                or any(not isinstance(s, int) or s < 0 or not 0 <= p <= 1 for s, p in pmf.items())):
            raise ValueError(f"invalid outcome distribution or time on block {i}")
        fraction = spec.remainder if isinstance(spec, _OfficialOutcome) else 0
        rewards[i] = [(s * 100 + fraction, p) for s, p in pmf.items() if p]
        for reward, _ in rewards[i]:
            ticks = gcd(ticks, reward)
    initial_bucket, offset = divmod(confirmed_pts * 100 + score_remainder, ticks)

    def points(bucket):
        return (bucket * ticks + offset) // 100

    @cache
    def battle_steps(i, charge):
        spec = specs[i]
        p = spec.win_prob
        retry = charge and battle_flags(graph.nodes[i], spec)[1]
        fail_sec = spec.expected_time_sec if spec.fail_time_sec is None else spec.fail_time_sec
        one_try_time = p * spec.expected_time_sec + (1 - p) * fail_sec
        time = one_try_time * (1 + (1 - p) * retry)
        passes = [(charge, p)]
        if retry:
            passes.append((False, (1 - p) * p))
        steps = [(reward // ticks, after, mass * prob) for after, mass in passes if mass
                 for reward, prob in rewards[i]]
        return steps, (1 - p) ** (2 if retry else 1), time

    @cache
    def transitions(i, bucket, charge, gold):
        steps, wipe, time = battle_steps(i, charge)
        if wallet_model is None:
            return [(bucket + d, c, 0, p) for d, c, p in steps], wipe, time
        extended = []
        wallet = gold + specs[i].gold_delta
        for delta, remaining, prob in steps:
            b = bucket + delta
            pts, rem = divmod(b * ticks + offset, 100)
            outcomes = wallet_model(graph.nodes[i], pts, rem, remaining, wallet)
            if not isclose(fsum(o.probability for o in outcomes), 1.0, abs_tol=1e-12):
                raise ValueError("wallet rewards must have unit probability mass")
            for o in outcomes:
                extra, residue = divmod(o.extra_base_pts * rate, ticks)
                if residue or o.gold_after < 0 or o.elapsed_sec < 0 or not 0 <= o.probability <= 1:
                    raise ValueError("invalid conditional wallet reward")
                if o.probability:
                    mass = prob * o.probability
                    extended.append((b + extra, remaining, o.gold_after, mass))
                    time += mass * o.elapsed_sec
        return extended, wipe, time

    def key(metrics):
        objective = metrics.hit - lambda_rate * metrics.time if target_score > 0 else metrics.score
        return objective, metrics.score, metrics.clear, -metrics.time

    policy = {}
    entries_at = defaultdict(set)
    initial = (initial_bucket, revive_available, initial_gold)
    for i in starts:
        entries_at[i].add(initial)
    for i in graph.order:
        # A prefix is reachable under any choice here, before fixing the optimal policy.
        exits = {step[:3] for args in entries_at.get(i, ()) for step in transitions(i, *args)[0]}
        for child in graph.children[i]:
            entries_at[child].update(exits)
    values = {}

    def state_key(i, bucket, charge, gold):
        return (i, bucket, charge, gold) if wallet_model is not None else (i, bucket, charge)

    @cache
    def after_node(i, bucket, charge, gold):
        children = graph.children[i]
        if children:
            options = [(c, values[c, bucket, charge, gold]) for c in children]
            child, chosen = max(options, key=lambda pair: key(pair[1]))
            policy[state_key(i, bucket, charge, gold)] = child
            return replace(chosen, upper=max(m.upper for _, m in options))
        if i in graph.clear_ends:
            score = points(bucket)
            return _Metrics(score, 1, float(score >= target_score), upper=bucket)
        return _Metrics()

    for i in reversed(graph.order):
        for bucket, charge, gold in entries_at.pop(i, ()):
            steps, _, time = transitions(i, bucket, charge, gold)
            score = clear = hit = 0.0
            upper = -1
            for b, remaining, wallet, prob in steps:
                continuation = after_node(i, b, remaining, wallet)
                score += prob * continuation.score
                clear += prob * continuation.clear
                hit += prob * continuation.hit
                time += prob * continuation.time
                upper = max(upper, continuation.upper)
            values[i, bucket, charge, gold] = _Metrics(score, clear, hit, time, upper)

    entries = [(i, values[i, *initial]) for i in starts]
    entry, metrics = max(entries, key=lambda pair: key(pair[1]))
    bound = max(m.upper for _, m in entries)
    mass = defaultdict(lambda: defaultdict(float))
    mass[entry][initial_bucket, revive_available, initial_gold] = 1.0
    distribution = defaultdict(float)
    for i in graph.order:
        for (bucket, charge, gold), incoming in mass.pop(i, {}).items():
            steps, wipe, _ = transitions(i, bucket, charge, gold)
            if wipe:
                distribution[0] += incoming * wipe
            for b, remaining, wallet, prob in steps:
                child = policy.get(state_key(i, b, remaining, wallet))
                if child is not None:
                    mass[child][b, remaining, wallet] += incoming * prob
                else:
                    distribution[points(b) if i in graph.clear_ends else 0] += incoming * prob
    route, i, b, charge, gold = [], entry, initial_bucket, revive_available, initial_gold
    while i is not None:
        route.append(graph.nodes[i])
        steps, _, _ = transitions(i, b, charge, gold)
        if not steps:
            break
        b, charge, gold, _ = max(steps, key=lambda step: step[3] * after_node(i, *step[:3]).clear)
        i = policy.get(state_key(i, b, charge, gold))
    top = 0 if bound < 0 else points(bound)
    return RouteDPResult(route, dict(sorted(distribution.items())), metrics.score, metrics.clear,
                         metrics.hit, metrics.time, top, max(300, bound),
                         ticks / 100, policy)


@dataclass(frozen=True)
class EarlyStopDecision:
    decision: str
    reason: str

    @property
    def should_retire(self) -> bool:
        return self.decision == "RETIRE"


def evaluate_early_stop(confirmed_pts: int, max_achievable_remaining_pts: int, target_score: int,
                        prob_reach_target: float = 1.0, expected_time_sec: float = 0.0,
                        lambda_rate: float = 0.0) -> EarlyStopDecision:
    top = confirmed_pts + max_achievable_remaining_pts
    if target_score > 0 and top < target_score:
        return EarlyStopDecision("RETIRE", f"单条可达路径的分数上界 {top} 低于目标 {target_score}")
    if target_score > 0 and lambda_rate > 0 and prob_reach_target < lambda_rate * expected_time_sec:
        return EarlyStopDecision("RETIRE", f"剩余命中率 {prob_reach_target:.4g} 低于时间成本 "
                                 f"{lambda_rate:.4g} × {expected_time_sec:.4g} 秒")
    return EarlyStopDecision("ACCEPT", "当前后缀通过分数上界与时间收益门禁")


@dataclass
class OpeningMapEvaluation(RouteDPResult):
    should_reroll: bool
    decision: str
    reason: str
    attempt_time_sec: float


def score_opening_map(nodes, settings, difficulty, outcome_specs=None, target_score=0,
                      lambda_rate=0.0, restart_time_sec=1.0, revive_available=False,
                      *, confirmed_pts=0, score_remainder=0) -> OpeningMapEvaluation:
    if restart_time_sec < 0 or not isfinite(restart_time_sec):
        raise ValueError("invalid restart duration")
    result = solve_route_dp(nodes, settings, difficulty, outcome_specs, target_score, lambda_rate,
                            revive_available, confirmed_pts=confirmed_pts, score_remainder=score_remainder)
    gate = evaluate_early_stop(confirmed_pts, result.hard_upper_bound - confirmed_pts, target_score,
                               result.prob_reach_target, result.expected_time_sec, lambda_rate)
    return OpeningMapEvaluation(**vars(result), should_reroll=gate.should_retire,
                                decision=gate.decision, reason=gate.reason,
                                attempt_time_sec=restart_time_sec if gate.should_retire else result.expected_time_sec)
