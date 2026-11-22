"""这是基于真实地图回放、官方计分对账与策略敏感性评测的离线沙盒，绝不用于 MBRL 训练。

Load a recorded enter/resume map_list with MapReplayEpisode.from_json. Recorded
observations, calibrated node priors and empirical shelves are separate inputs.
Every input attempt produces exactly one outcome, including rejected opening maps.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from functools import cache
from math import fsum, inf
from pathlib import Path
from random import Random
from typing import Protocol

from labyrinth_nn.catalog.schema import FeatureCatalog, ScoreSettings
from labyrinth_nn.data.sample_schema import BattleSample, OUTCOME_WIN
from labyrinth_nn.planning.marginal_eval import (
    BattleExam, BattlePassEstimator, EvaluatedCandidate, EventBranch, EventOption,
    HeuristicBattlePassEstimator, PlanningState, award_collections, battle_value, default_event_tree,
    evaluate_event_options, evaluate_relic_choices, evaluate_role_choices,
)
from labyrinth_nn.planning.route_dp import (
    BlockType, MapNode, NodeOutcomeSpec, RouteGraph, WalletOutcome, battle_flags, build_route_graph,
    default_node_outcome_specs, evaluate_early_stop, node_score_counts, solve_route_dp,
)
from labyrinth_nn.planning.scoring import DEFAULT_SETTINGS, ScoreCounts, base_score, compute_final_score
from labyrinth_nn.planning.shop_dp import (
    DEFAULT_SHELVES, ShopItem, ShopPlanResult, ShopShelfModel, clearance_ladder,
    plan_shop_visit, shop_visit_distribution,
)


@dataclass(frozen=True)
class BattleObservation:
    outcome_class: int
    elapsed_sec: float | None = None

    def __post_init__(self):
        if self.outcome_class not in (0, 1, 2) or (self.elapsed_sec is not None and self.elapsed_sec < 0):
            raise ValueError("invalid recorded battle observation")


@dataclass(frozen=True)
class NodeReplayData:
    role_choices: tuple[int, ...] = ()
    relic_choices: tuple[int, ...] = ()
    shop_shelves: tuple[tuple[ShopItem, ...], ...] = ()  # visible shelf, then successive realized refreshes
    refresh_available: bool = True
    event_id: int | None = None
    event_rolls: tuple[float, ...] = ()  # successive observed branch quantiles, in [0, 1)
    battle_results: tuple[BattleObservation, ...] = ()  # first try, then retry
    gold_delta: int | None = None
    exam: BattleExam | None = None


def _item(raw):
    return ShopItem(raw) if isinstance(raw, str) else ShopItem(**raw)


def _node_data(raw):
    data = dict(raw)
    for name in ("role_choices", "relic_choices", "event_rolls"):
        if name in data:
            data[name] = tuple(data[name])
    if "shop_shelves" in data:
        data["shop_shelves"] = tuple(tuple(_item(item) for item in shelf) for shelf in data["shop_shelves"])
    if "battle_results" in data:
        data["battle_results"] = tuple(BattleObservation(r["outcome_class"], r.get("elapsed_sec"))
                                       for r in data["battle_results"])
    if data.get("exam") is not None:
        data["exam"] = BattleExam(**data["exam"])
    return NodeReplayData(**data)


@dataclass(frozen=True)
class MapReplayEpisode:
    """One opening attempt. Initial official counts are authoritative, defaulting to inventory counts."""
    nodes: tuple[MapNode, ...]
    episode_id: str = ""
    difficulty: int = 5
    initial_state: PlanningState = field(default_factory=PlanningState)
    initial_counts: ScoreCounts | None = None
    node_data: Mapping[int, NodeReplayData] = field(default_factory=dict)
    outcome_specs: Mapping[int, NodeOutcomeSpec] = field(default_factory=dict)  # forecasts, not extra score
    battle_samples: tuple[BattleSample, ...] = ()
    shelf_model: ShopShelfModel = DEFAULT_SHELVES
    event_tree: Mapping[int, Sequence[EventOption]] = field(default_factory=dict)
    restart_time_sec: float = 1.0
    max_refreshes: int | None = None

    @classmethod
    def from_dict(cls, payload: dict, **overrides) -> MapReplayEpisode:
        data = {**payload, **payload.get("data", {})}
        # Saved Desktop routes contain both a selected path (nodes) and the complete map (allNodes).
        rows = next((data[key] for key in ("map_list", "allNodes", "nodes") if key in data), None)
        if rows is None:
            raise ValueError("recording has no map_list, allNodes or nodes")
        nodes = []
        for r in rows:
            ident = r.get("block_id", r.get("blockId"))
            kind = r.get("block_type", r.get("blockType"))
            quest = int(r.get("quest_id", r.get("questId")) or 0)
            edges = r.get("next_block_id_list", r.get("nextBlockIds", r.get("next_block_ids", ())))
            last = r.get("IsAreaLastPoint", r.get("isAreaLastPoint", r.get("is_area_last_point", False)))
            nodes.append(MapNode(int(ident), int(r["area"]), int(r["column"]), int(r["row"]), int(kind),
                                 quest if quest > 0 else None, tuple(map(int, edges)), last in (True, "true", "1")))
        initial = dict(data.get("initial_state", {}))
        initial["remaining_battles"] = tuple(BattleExam(**e) for e in initial.get("remaining_battles", ()))
        counts = dict(data["initial_counts"]) if data.get("initial_counts") is not None else None
        if counts and isinstance(counts.get("blocks_cleared"), dict):
            counts["blocks_cleared"] = {int(k): v for k, v in counts["blocks_cleared"].items()}
        specs = {}
        for key, value in data.get("outcome_specs", {}).items():
            spec = dict(value)
            if isinstance(spec.get("direct_pts_buckets"), dict):
                spec["direct_pts_buckets"] = {int(k): p for k, p in spec["direct_pts_buckets"].items()}
            specs[int(key)] = NodeOutcomeSpec(**spec)
        events = {int(k): tuple(EventOption(o["choice_id"], tuple(EventBranch(**b) for b in o["branches"]),
                                            o.get("condition_value", 0), o.get("time_sec", 0.0)) for o in opts)
                  for k, opts in data.get("event_tree", {}).items()}
        kwargs = dict(nodes=tuple(nodes), episode_id=str(data.get("episode_id", data.get("enter_id", data.get("enterId", "")))),
                      difficulty=int(data.get("difficulty", initial.get("difficulty", 5))),
                      initial_state=PlanningState(**initial), initial_counts=ScoreCounts(**counts) if counts is not None else None,
                      node_data={int(k): _node_data(v) for k, v in data.get("node_data", {}).items()},
                      outcome_specs=specs, event_tree=events, restart_time_sec=float(data.get("restart_time_sec", 1)),
                      max_refreshes=data.get("max_refreshes"),
                      battle_samples=tuple(BattleSample.from_dict(s) for s in data.get("battle_samples", ())))
        if "shelf_model" in data:
            kwargs["shelf_model"] = ShopShelfModel(tuple((p, tuple(_item(it) for it in shelf))
                                                         for p, shelf in data["shelf_model"]))
        return cls(**{**kwargs, **overrides})

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> MapReplayEpisode:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")), **overrides)


def load_replays(path: str | Path, **overrides) -> list[MapReplayEpisode]:
    """Read enter/resume JSON, JSONL, or a Desktop database snapshot with its WAL, if any."""
    path = Path(path)
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        path = path.resolve()
        uri = path.as_uri() + "?mode=ro"
        try:
            wal_size = path.with_name(path.name + "-wal").stat().st_size
        except FileNotFoundError:
            wal_size = 0
        # Immutable reads ignore committed rows that are still in the WAL.
        if wal_size == 0:
            uri += "&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            return [MapReplayEpisode.from_dict(json.loads(row[0]), **overrides)
                    for row in conn.execute("SELECT route_json FROM labyrinth_routes ORDER BY rowid")]
    with path.open(encoding="utf-8-sig") as f:
        if path.suffix.lower() == ".jsonl":
            return [MapReplayEpisode.from_dict(json.loads(line), **overrides) for line in f if line.strip()]
        data = json.load(f)
        return [MapReplayEpisode.from_dict(row, **overrides) for row in (data if isinstance(data, list) else [data])]


@dataclass(frozen=True)
class RouteChoice:
    route: tuple[MapNode, ...]
    retire: bool = False
    reason: str = ""


@dataclass
class ReplayContext:
    episode: MapReplayEpisode
    graph: RouteGraph
    settings: ScoreSettings
    catalog: FeatureCatalog | None
    truth_estimator: BattlePassEstimator
    defaults: dict[int, NodeOutcomeSpec]
    events: dict[int, Sequence[EventOption]]

    def data(self, node):
        return self.episode.node_data.get(node.block_id, NodeReplayData())

    def exam(self, node, state):
        data = self.data(node)
        if data.exam:
            return data.exam
        for exam in self.episode.initial_state.remaining_battles:
            if node.quest_id is not None and node.quest_id == exam.quest_id:
                return exam
        kind = node.block_type
        pts = self.defaults[node.block_id].direct_pts_buckets
        requirement = {BlockType.NORMAL_BATTLE: 1.0, BlockType.EX_BATTLE: 2.5, BlockType.BOSS: 3.5}
        return BattleExam(node.quest_id or node.block_id, pts, requirement.get(kind, 2.5),
                          kind == BlockType.BOSS, self.defaults[node.block_id].expected_time_sec)

    def matching_runs(self, node, state):
        # A counterfactual inventory must not inherit another roster's observed wins.
        return [run for s in self.episode.battle_samples
                if (s.quest_id == node.quest_id and s.area == node.area and s.difficulty == state.difficulty
                    and frozenset(s.owned_unit_ids) == state.owned_unit_ids and s.owned_unit_count == state.unit_count
                    and sorted(s.relic_ids) == sorted(state.relic_ids) and tuple(s.mark_stacks) == state.mark_stacks
                    and (s.alpha_rupee is None or s.alpha_rupee == state.gold)
                    and s.revive_available == bool(state.revive.remaining_charges))
                for run in s.runs]

    def battle_spec(self, node, state, estimator):
        i = node.block_id
        if i in self.episode.outcome_specs:
            return self.episode.outcome_specs[i]
        spec = self.defaults[i]
        boss, battle = battle_flags(node, spec)
        if not (battle or boss):
            return spec
        runs = self.matching_runs(node, state)
        if runs:
            wins = [r.elapsed_sec for r in runs if r.outcome_class == OUTCOME_WIN]
            losses = [r.elapsed_sec for r in runs if r.outcome_class != OUTCOME_WIN]
            return replace(spec, win_prob=len(wins) / len(runs),
                           expected_time_sec=fsum(wins) / len(wins) if wins else spec.expected_time_sec,
                           fail_time_sec=fsum(losses) / len(losses) if losses else spec.expected_time_sec)
        return replace(spec, win_prob=estimator.estimate(state, self.exam(node, state)).mean_prob)

    def future_nonshop_points(self, suffix):
        """Conditional score PMF on the selected suffix; reserve all future shop payouts for f_shop."""
        pmf = {0: 1.0}
        for node in suffix:
            spec = self.episode.outcome_specs.get(node.block_id, self.defaults[node.block_id])
            if node.block_type in (BlockType.SHOP, BlockType.EVENT):
                spec = self.defaults[node.block_id]
            raw = spec.direct_pts_buckets
            rewards = {raw: 1.0} if isinstance(raw, int) else raw
            combined = defaultdict(float)
            for a, p in pmf.items():
                for b, q in rewards.items():
                    combined[a + b] += p * q
            pmf = combined
        return pmf

    def inventory_horizon(self, state, suffix):
        exams = tuple(self.exam(n, state) for n in suffix if any(battle_flags(
            n, self.episode.outcome_specs.get(n.block_id, self.defaults[n.block_id]))))
        last_area = max(n.area for n in self.graph.nodes.values())
        flags = tuple(n.area != last_area and self.data(n).refresh_available for n in suffix if n.block_type == BlockType.SHOP)
        return replace(state, remaining_battles=exams, remaining_shops=len(flags), future_refresh_available=flags)


class ReplayPolicy(Protocol):
    def choose_route(self, ctx: ReplayContext, state: PlanningState, starts: tuple[int, ...]) -> RouteChoice: ...
    def choose_reward(self, ctx: ReplayContext, state: PlanningState, choices: Sequence[int], relic: bool) -> int: ...
    def choose_shop(self, ctx: ReplayContext, state: PlanningState, shelf: tuple[ShopItem, ...],
                    refresh_count: int, allow: bool) -> ShopPlanResult: ...
    def choose_event(self, ctx: ReplayContext, state: PlanningState, event: int,
                     suffix: Sequence[MapNode]) -> EvaluatedCandidate | None: ...


@dataclass
class PathBReplayPolicy:
    target_score: int = 0
    lambda_rate: float = 0.0
    estimator: BattlePassEstimator = field(default_factory=HeuristicBattlePassEstimator)
    deduct_condition_principal: bool = True

    def retention_value(self, state):
        if state.remaining_battles and any(r in (50003, 50009) for r in state.relic_ids):
            return lambda coins: battle_value(replace(state, gold=coins), self.estimator)
        return None

    def _wallet_model(self, ctx, state, specs):
        # Use one feasible suffix to supply exposure weights and the future shop count
        # to local purchase/event policies. Bellman still compares every actual successor.
        suffixes, forecast = {}, {}
        for i in reversed(ctx.graph.order):
            children = ctx.graph.children[i]
            child = max(children, key=forecast.__getitem__) if children else None
            suffix = (ctx.graph.nodes[child], *suffixes[child]) if child is not None else ()
            suffixes[i] = suffix
            spec = specs[i]
            raw = spec.direct_pts_buckets
            gain = raw if isinstance(raw, int) else fsum(k * p for k, p in raw.items())
            forecast[i] = spec.win_prob * (gain + (forecast[child] if child is not None else 0))
        last_area = max(n.area for n in ctx.graph.nodes.values())

        @cache
        def horizon(i):
            return ctx.inventory_horizon(state, suffixes[i])

        @cache
        def checkout(i, wallet, remainder):
            s = horizon(i)
            allow = ctx.graph.nodes[i].area != last_area and ctx.data(ctx.graph.nodes[i]).refresh_available
            pmf = shop_visit_distribution(wallet, 1 + s.remaining_shops, ctx.settings, s.difficulty,
                                          discount_model=s.discount, shelf_model=ctx.episode.shelf_model,
                                          max_refreshes=ctx.episode.max_refreshes, refresh_available=allow,
                                          owned_unit_ids=s.owned_unit_ids, score_remainder=remainder,
                                          coin_retention_value_fn=self.retention_value(s),
                                          future_refresh_available=s.future_refresh_available)
            return tuple(WalletOutcome(u * ctx.settings.unit_pts + r * ctx.settings.relic_pts, coins, p)
                         for (u, r, coins), p in pmf.items())

        def outcomes(node, points, remainder, charge, wallet):
            i = node.block_id
            if i in ctx.episode.outcome_specs:
                return (WalletOutcome(0, wallet, 1),)
            if node.block_type == BlockType.SHOP:
                return checkout(i, wallet, remainder)
            event = ctx.data(node).event_id or node.quest_id
            if node.block_type == BlockType.EVENT and event in ctx.events:
                s = replace(horizon(i), gold=wallet, confirmed_pts=points, score_remainder=remainder,
                            revive_used_count=int(state.revive.has_relic and not charge) or state.revive_used_count)
                choice = self.choose_event(ctx, s, event, suffixes[i])
                if choice:
                    return tuple(WalletOutcome((o.state.unit_count - s.unit_count) * ctx.settings.unit_pts
                                               + (o.state.relic_count - s.relic_count) * ctx.settings.relic_pts,
                                               o.state.gold, o.probability, o.elapsed_sec) for o in choice.outcomes)
            return (WalletOutcome(0, wallet, 1),)
        return outcomes

    def choose_route(self, ctx, state, starts):
        specs = {i: ctx.battle_spec(n, state, self.estimator) for i, n in ctx.graph.nodes.items()}
        needs_wallet = any(i not in ctx.episode.outcome_specs and n.block_type in (BlockType.EVENT, BlockType.SHOP)
                           for i, n in ctx.graph.nodes.items())
        wallet_model = self._wallet_model(ctx, state, specs) if needs_wallet else None
        result = solve_route_dp(ctx.graph, ctx.settings, state.difficulty, specs, self.target_score,
                                self.lambda_rate, bool(state.revive.remaining_charges), confirmed_pts=state.confirmed_pts,
                                score_remainder=state.score_remainder, start_node_ids=starts,
                                wallet_model=wallet_model, initial_gold=state.gold)
        gate = evaluate_early_stop(state.confirmed_pts, result.hard_upper_bound - state.confirmed_pts,
                                   self.target_score, result.prob_reach_target, result.expected_time_sec, self.lambda_rate)
        return RouteChoice(tuple(result.best_route), gate.should_retire, gate.reason)

    def choose_reward(self, ctx, state, choices, relic):
        kwargs = dict(settings=ctx.settings, shelf_model=ctx.episode.shelf_model)
        if relic:
            return evaluate_relic_choices(state, choices, self.estimator, ctx.catalog, **kwargs)[0].candidate_id
        return evaluate_role_choices(state, choices, self.estimator, **kwargs)[0].candidate_id

    def choose_shop(self, ctx, state, shelf, refresh_count, allow):
        return plan_shop_visit(shelf, state.gold, ctx.settings, state.difficulty, discount_model=state.discount,
                               shelf_model=ctx.episode.shelf_model, refresh_count=refresh_count,
                               max_refreshes=ctx.episode.max_refreshes, refresh_available=allow,
                               remaining_shops=1 + state.remaining_shops, coin_retention_value_fn=self.retention_value(state),
                               owned_unit_ids=state.owned_unit_ids, score_remainder=state.score_remainder,
                               future_refresh_available=state.future_refresh_available)

    def choose_event(self, ctx, state, event, suffix):
        choices = evaluate_event_options(state, event, self.estimator, self.deduct_condition_principal,
                                         event_tree=ctx.episode.event_tree, settings=ctx.settings, catalog=ctx.catalog,
                                         shelf_model=ctx.episode.shelf_model, target_score=self.target_score,
                                         lambda_rate=self.lambda_rate,
                                         future_score_distribution=ctx.future_nonshop_points(suffix))
        return choices[0] if choices else None


@dataclass
class BaselineHeuristicPolicy:
    """Always enters, favors easy fights, takes the first offered reward and never refreshes."""
    deduct_condition_principal: bool = True

    def choose_route(self, ctx, state, starts):
        priorities = (BlockType.LINK, BlockType.RELIC, BlockType.NORMAL_BATTLE, BlockType.SHOP, BlockType.EVENT)
        rank = {kind: i for i, kind in enumerate(priorities)}
        route = []
        while starts:
            i = min(starts, key=lambda n: (rank.get(ctx.graph.nodes[n].block_type, len(rank)), n))
            route.append(ctx.graph.nodes[i])
            starts = ctx.graph.children[i]
        return RouteChoice(tuple(route))

    def choose_reward(self, ctx, state, choices, relic):
        return choices[0]

    def choose_shop(self, ctx, state, shelf, refresh_count, allow):
        purchases = clearance_ladder(shelf, state.gold, ctx.settings, state.difficulty, discount_model=state.discount)
        remaining = state.gold - sum(state.discount.price(it.base_price) for it in purchases)
        indices = list(range(len(shelf)))
        used = []
        for item in purchases:
            i = next(i for i in indices if shelf[i] == item)
            used.append(i)
            indices.remove(i)
        points = sum((ctx.settings.relic_pts if it.is_relic else ctx.settings.unit_pts) * it.collection_count
                     for it in purchases) * ctx.settings.score_rates[state.difficulty] // 100
        return ShopPlanResult(purchases, tuple(used), remaining, False, 0, points, points)

    def choose_event(self, ctx, state, event, suffix):
        choices = evaluate_event_options(state, event, ctx.truth_estimator, self.deduct_condition_principal,
                                         event_tree=ctx.episode.event_tree, settings=ctx.settings, catalog=ctx.catalog,
                                         shelf_model=ctx.episode.shelf_model, first_continuation=True)
        feasible = {choice.candidate_id: choice for choice in choices}
        return next((feasible[o.choice_id] for o in ctx.events[event] if o.choice_id in feasible), None)


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_id: str
    status: str
    final_score: int  # settled score, zero for any non-clear
    ledger_score: int  # audit value of the counters even on unsuccessful attempts
    elapsed_sec: float
    counts: ScoreCounts
    state: PlanningState
    visited_blocks: tuple[int, ...]
    event_choices: tuple[int, ...]
    shop_refreshes: int
    reason: str


def _draw(weighted, roll):
    if not 0 <= roll < 1:
        raise ValueError("recorded random quantile outside [0, 1)")
    pos = roll * fsum(p for p, _ in weighted)
    for weight, value in weighted:
        pos -= weight
        if pos < 0:
            return value
    return weighted[-1][1]


def _event_leaf(candidate, rolls, rng):
    leaves = candidate.outcomes
    depth = 0
    while len(leaves) > 1:
        groups = defaultdict(list)
        for leaf in leaves:
            groups[leaf.branch_indices[depth]].append(leaf)
        weighted = [(fsum(o.probability for o in group), group) for group in groups.values()]
        leaves = _draw(weighted, rolls[depth] if depth < len(rolls) else rng.random())
        depth += 1
    return leaves[0]


def _wallet_shelf(shelf, state):
    return tuple(replace(it, collection_count=0) if it.unit_id in state.owned_unit_ids and not it.is_relic
                 else it for it in shelf)


def _shop(ctx, policy, node, state, rng):
    data = ctx.data(node)
    allow = data.refresh_available and node.area != max(n.area for n in ctx.graph.nodes.values())
    count = 0
    while True:
        if count < len(data.shop_shelves):
            shelf = data.shop_shelves[count]
        else:
            shelf = _draw(ctx.episode.shelf_model.shelves, rng.random())
        shelf = _wallet_shelf(shelf, state)
        plan = policy.choose_shop(ctx, state, shelf, count, allow)
        state = replace(state, gold=plan.gold_remaining)
        for it in plan.purchases:
            if it.is_relic:
                ids = (it.relic_id,) if it.relic_id is not None and it.collection_count else ()
                state = award_collections(state, ctx.settings, relic_ids=ids, new_relics=it.collection_count - len(ids),
                                          catalog=ctx.catalog)
            else:
                ids = (it.unit_id,) if it.unit_id is not None and it.collection_count else ()
                state = award_collections(state, ctx.settings, unit_ids=ids, new_units=it.collection_count - len(ids))
        if not plan.should_refresh:
            return state, count
        if not allow or count >= len(ctx.settings.shop_refresh_costs):
            raise ValueError("policy chose an unavailable shop refresh")
        fee = ctx.settings.shop_refresh_costs[count]
        state = replace(state, gold=state.gold - fee)
        count += 1


def _initial_ledger(episode, settings):
    state = replace(episode.initial_state, difficulty=episode.difficulty)
    counts = episode.initial_counts or ScoreCounts(units=state.unit_count, relics=state.relic_count)
    if counts.units < state.unit_count or counts.relics < state.relic_count:
        raise ValueError("initial ledger omits owned inventory")
    base = base_score(settings, counts)
    pts, rem = divmod(base * settings.score_rates[state.difficulty], 100)
    state = replace(state, unidentified_units=counts.units - len(state.owned_unit_ids),
                    unidentified_relics=counts.relics - len(state.relic_ids), confirmed_pts=pts, score_remainder=rem)
    blocks = counts.blocks_cleared
    if isinstance(blocks, int):
        # Integer counts were accepted by base_score only for a uniform per-block price.
        blocks = {min(settings.block_pts): blocks} if blocks else {}
    elif not isinstance(blocks, dict):
        blocks = dict(Counter(blocks))
    return state, replace(counts, areas_cleared=counts.area_ids(settings), blocks_cleared=dict(blocks))


def run_episode(episode: MapReplayEpisode, policy: ReplayPolicy, settings: ScoreSettings = DEFAULT_SETTINGS,
                *, seed: int = 0, catalog: FeatureCatalog | None = None,
                truth_estimator: BattlePassEstimator | None = None) -> EpisodeOutcome:
    graph = build_route_graph(episode.nodes)
    if episode.restart_time_sec < 0:
        raise ValueError("negative restart duration")
    # A recorded offer fixes the reward category. Its identity is chosen only on arrival.
    extra_units = {i: 1 for i, data in episode.node_data.items()
                   if i in graph.nodes and data.role_choices and graph.nodes[i].block_type != BlockType.LINK}
    extra_relics = {i: 1 for i, data in episode.node_data.items()
                    if i in graph.nodes and data.relic_choices and graph.nodes[i].block_type != BlockType.RELIC}
    defaults = default_node_outcome_specs(graph, settings, episode.difficulty,
                                          reward_units=extra_units, reward_relics=extra_relics)
    ctx = ReplayContext(episode, graph, settings, catalog, truth_estimator or HeuristicBattlePassEstimator(catalog),
                        defaults, {**default_event_tree(), **episode.event_tree})
    state, counts = _initial_ledger(episode, settings)
    if catalog and state.relic_ids and not any(state.mark_stacks):
        state = replace(state, mark_stacks=tuple(catalog.mark_stacks_for_relics(list(state.relic_ids))))
    visited, events = [], []
    refreshes = 0
    elapsed = 0.0
    starts = graph.starts

    def finish(status, reason=""):
        ledger = compute_final_score(settings, episode.difficulty, counts)
        return EpisodeOutcome(episode.episode_id, status, ledger if status == "clear" else 0, ledger,
                              elapsed, counts, state, tuple(visited), tuple(events), refreshes, reason)

    while starts:
        decision = policy.choose_route(ctx, state, starts)
        if decision.retire:
            elapsed += episode.restart_time_sec
            return finish("midrun_retire" if visited else "opening_reroll", decision.reason)
        node, *suffix = decision.route
        i = node.block_id
        if i not in starts:
            raise ValueError(f"policy selected unreachable block {i}")
        visited.append(i)
        data = ctx.data(node)
        state = ctx.inventory_horizon(state, suffix)
        spec = ctx.battle_spec(node, state, ctx.truth_estimator)
        boss, nonboss = battle_flags(node, spec)
        # Node-local streams keep shared observations paired when two policies take different paths.
        stream = f"{seed}/{episode.episode_id}/{i}/"
        rng = Random(stream + "battle")
        attempt = 0
        while True:
            run = None
            if attempt < len(data.battle_results):
                run = data.battle_results[attempt]
            elif boss or nonboss:
                observed = ctx.matching_runs(node, state)
                if observed:
                    run = rng.choice(observed)
            won = run.outcome_class == OUTCOME_WIN if run else rng.random() < spec.win_prob
            seconds = spec.expected_time_sec if won or spec.fail_time_sec is None else spec.fail_time_sec
            elapsed += run.elapsed_sec if run and run.elapsed_sec is not None else seconds
            if won:
                break
            if not nonboss or not state.revive.can_prevent_defeat(boss):
                return finish("wipe", f"节点 {i} 未通过")
            state = replace(state, revive_used_count=state.revive.apply_battle_loss(boss).used_count)
            attempt += 1

        delta = node_score_counts(node, i in graph.area_ends)
        blocks = dict(counts.blocks_cleared)
        if node.block_type not in (BlockType.CLEAR, BlockType.UNKNOWN):
            blocks[node.area] = blocks.get(node.area, 0) + 1
        counts = replace(counts, areas_cleared=counts.areas_cleared | delta.area_ids(settings), blocks_cleared=blocks,
                         normal_wins=counts.normal_wins + delta.normal_wins, ex_wins=counts.ex_wins + delta.ex_wins,
                         boss_wins=counts.boss_wins + delta.boss_wins)
        pts, rem = divmod(base_score(settings, counts) * settings.score_rates[state.difficulty], 100)
        gold = spec.gold_delta if data.gold_delta is None else data.gold_delta
        state = replace(state, confirmed_pts=pts, score_remainder=rem, gold=state.gold + gold)
        for choices, relic, default_count in ((data.role_choices, False, delta.units), (data.relic_choices, True, delta.relics)):
            if choices:
                chosen = policy.choose_reward(ctx, state, choices, relic)
                if chosen not in choices:
                    raise ValueError("policy selected an unavailable reward")
                state = award_collections(state, settings, relic_ids=(chosen,) if relic else (),
                                          unit_ids=() if relic else (chosen,), catalog=catalog)
            elif default_count:
                state = award_collections(state, settings, new_relics=int(relic), new_units=int(not relic))
        if node.block_type == BlockType.SHOP:
            state, n = _shop(ctx, policy, node, state, Random(stream + "shop"))
            refreshes += n
        event = data.event_id or node.quest_id
        if node.block_type == BlockType.EVENT and event in ctx.events:
            candidate = policy.choose_event(ctx, state, event, suffix)
            if candidate is not None:
                leaf = _event_leaf(candidate, data.event_rolls, Random(stream + "event"))
                events.extend(leaf.choices)
                state, elapsed = leaf.state, elapsed + leaf.elapsed_sec
        counts = replace(counts, units=state.unit_count, relics=state.relic_count)
        if i in graph.clear_ends:
            return finish("clear")
        starts = graph.children[i]
    return finish("wipe", "地图在通关出口前中断")


@dataclass(frozen=True)
class BenchmarkReport:
    total_attempts: int
    clears: int
    wipes: int
    opening_rerolls: int
    midrun_retires: int
    expected_score: float
    p_ge_50k: float
    p_ge_53k: float
    clear_rate: float
    abandon_rate: float
    mean_time_per_target_hit_sec: float
    mean_time_per_attempt_sec: float
    target_score: int
    target_hits: int

    @classmethod
    def from_outcomes(cls, outcomes: Iterable[EpisodeOutcome], target_score: int = 53000) -> BenchmarkReport:
        rows = tuple(outcomes)
        n = len(rows)
        counts = Counter(o.status for o in rows)
        if counts.keys() - {"clear", "wipe", "opening_reroll", "midrun_retire"}:
            raise ValueError("unknown attempt status")
        total_time = fsum(o.elapsed_sec for o in rows)
        settled = [o.final_score for o in rows if o.status == "clear"]
        hits = sum(s >= target_score for s in settled)
        denominator = n or 1
        abandons = counts["opening_reroll"] + counts["midrun_retire"]
        return cls(n, counts["clear"], counts["wipe"], counts["opening_reroll"], counts["midrun_retire"],
                   sum(settled) / denominator, sum(s >= 50000 for s in settled) / denominator,
                   sum(s >= 53000 for s in settled) / denominator, counts["clear"] / denominator,
                   abandons / denominator, total_time / hits if hits else inf, total_time / denominator,
                   target_score, hits)


def run_benchmark(episodes: Iterable[MapReplayEpisode], policy: ReplayPolicy, settings: ScoreSettings = DEFAULT_SETTINGS,
                  *, repeats: int = 1, seed: int = 0, target_score: int | None = None,
                  catalog: FeatureCatalog | None = None) -> BenchmarkReport:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if target_score is None:
        target_score = getattr(policy, "target_score", 0) or 53000
    maps = tuple(episodes)
    outcomes = (run_episode(ep, policy, settings, seed=seed + r * len(maps) + i, catalog=catalog)
                for r in range(repeats) for i, ep in enumerate(maps))
    return BenchmarkReport.from_outcomes(outcomes, target_score)


def compare_policies(episodes: Iterable[MapReplayEpisode], path_b: PathBReplayPolicy | None = None,
                     settings: ScoreSettings = DEFAULT_SETTINGS, **kwargs) -> dict[str, BenchmarkReport]:
    maps = tuple(episodes)
    path_b = path_b or PathBReplayPolicy()
    kwargs.setdefault("target_score", path_b.target_score or 53000)
    return {name: run_benchmark(maps, policy, settings, **kwargs) for name, policy in
            (("path_b", path_b), ("baseline", BaselineHeuristicPolicy(path_b.deduct_condition_principal)))}
