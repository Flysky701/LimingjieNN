"""Numerical oracles and end-to-end replay checks. Uses only unittest and the standard library."""
import json
import math
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from labyrinth_nn.catalog.schema import EventChoiceEntry, FeatureCatalog, RelicEntry, ScoreSettings
from labyrinth_nn.data.sample_schema import BattleSample, SingleRunResult
from labyrinth_nn.env.run_env import (
    BaselineHeuristicPolicy, BattleObservation, BenchmarkReport, MapReplayEpisode, NodeReplayData,
    PathBReplayPolicy, compare_policies, load_replays, run_episode,
)
from labyrinth_nn.planning.marginal_eval import (
    BattleExam, EventBranch, EventOption, HeuristicBattlePassEstimator, PlanningState,
    award_collections, default_event_tree, evaluate_event_options, evaluate_relic_choices,
    evaluate_role_choices, pass_profile,
)
from labyrinth_nn.planning.route_dp import (
    BlockType as B, MapNode, NodeOutcomeSpec, build_route_graph, evaluate_early_stop,
    score_opening_map, solve_route_dp,
)
from labyrinth_nn.planning.scoring import (
    DEFAULT_SETTINGS as SS, REVIVE_RELIC_ID, ReviveState, ScoreCounts, area_clear_delta_pts, base_score,
    block_delta_pts, boss_win_delta_pts, compute_final_score, delta_ex_vs_normal_pts, ex_win_delta_pts,
    normal_win_delta_pts, relic_delta_pts, scale_pts, score_check_gate, score_gcd, unit_delta_pts,
)
from labyrinth_nn.planning.shop_dp import (
    ShopDiscountModel, ShopItem, ShopItemKind, ShopShelfModel, clearance_ladder, gold_to_pts_curve,
    marginal_gold_pts, plan_shop_visit, shop_score_distribution,
)


def node(i, kind=B.START, edges=(), *, area=1, col=None, last=False, quest=None):
    return MapNode(i, area, i if col is None else col, 0, kind, quest, edges, last)


def spec(pts, seconds=1, win=1):
    return NodeOutcomeSpec(win_prob=win, direct_pts_buckets=pts, expected_time_sec=seconds)


def catalog_of(*entries):
    return FeatureCatalog(1, "", "", "", 5, 10, {}, {}, {}, {e.relic_id: e for e in entries},
                          {}, SS, [], [], {})


class OfficialScoringTests(unittest.TestCase):
    def test_reference_scores_and_gate(self):
        counts = ScoreCounts(5, 45, 30, 15, 5, 4, 5)
        self.assertEqual(base_score(SS, counts), 25000)
        self.assertEqual(compute_final_score(SS, 5, counts), 50000)
        self.assertEqual(compute_final_score(SS, 5, replace(counts, units=35, ex_wins=5)), 53000)
        self.assertEqual(score_gcd(SS, 5), 200)
        before = replace(counts, areas_cleared=4, blocks_cleared=44, boss_wins=4)
        gate = score_check_gate(SS, 5, before, 50000)
        self.assertEqual((gate.current_score, gate.projected_clear_score, gate.passed), (45800, 50000, True))
        gate = score_check_gate(SS, 5, replace(before, areas_cleared={1, 2, 3, 4, 5}), 53000, 5)
        self.assertEqual((gate.projected_clear_score, gate.shortfall, gate.should_retire), (50000, 3000, True))

    def test_different_floor_prices_and_integer_floor(self):
        settings = replace(SS, area_clear_pts={a: 10 * a for a in range(1, 11)},
                           block_pts={a: a for a in range(1, 11)}, score_rates={7: 135})
        counts = ScoreCounts(10, {1: 2, 10: 3})
        self.assertEqual(base_score(settings, counts), 582)
        self.assertEqual(compute_final_score(settings, 7, counts), 785)
        self.assertEqual(score_gcd(settings, 7), 1)
        self.assertEqual(base_score(settings, ScoreCounts([2, 10], [1, 10, 10])), 141)
        with self.assertRaises(ValueError):
            base_score(settings, ScoreCounts(blocks_cleared=3))
        self.assertEqual(scale_pts(settings, 7, 10**20 + 1), (10**20 + 1) * 135 // 100)

    def test_incremental_values(self):
        self.assertEqual([f(SS, 5) for f in (unit_delta_pts, relic_delta_pts, normal_win_delta_pts,
                                              ex_win_delta_pts, boss_win_delta_pts)], [200, 400, 200, 2000, 3000])
        self.assertEqual((block_delta_pts(SS, 5, 3), area_clear_delta_pts(SS, 5, 3)), (200, 1000))
        self.assertEqual(delta_ex_vs_normal_pts(SS, 5), 1800)
        self.assertEqual(delta_ex_vs_normal_pts(SS, 5, extra_relics=1, extra_units=-1), 2000)

    def test_single_nonboss_charge(self):
        full = ReviveState.from_relics([REVIVE_RELIC_ID] * 2)
        self.assertTrue(full.can_prevent_defeat(False))
        self.assertFalse(full.can_prevent_defeat(True))
        self.assertIs(full.apply_battle_loss(True), full)
        used = full.apply_battle_loss(False)
        self.assertEqual(used, ReviveState.from_relics([50002], used_count=1))
        self.assertFalse(used.can_prevent_defeat(False))
        self.assertEqual(used.apply_battle_loss(False).used_count, 1)
        self.assertFalse(ReviveState.from_relics([]).has_relic)


class RouteTests(unittest.TestCase):
    def test_full_300_bucket_convolution(self):
        nodes = [node(1, edges=[2]), node(2, edges=[3]), node(3)]
        pmf = {200 * i: 1 / 101 for i in range(101)}
        result = solve_route_dp(nodes, SS, 5, {n.block_id: spec(pmf) for n in nodes})
        self.assertEqual((result.bucket_size, result.num_buckets, result.hard_upper_bound), (200, 300, 60000))
        self.assertEqual(len(result.score_distribution), 301)
        for k in range(301):
            # Coefficients of (1-x^101)^3 / (1-x)^3, independent of the DP.
            ways = sum((-1)**j * math.comb(3, j) * math.comb(k - 101*j + 2, 2)
                       for j in range(k // 101 + 1))
            self.assertAlmostEqual(result.score_distribution[200*k], ways / 101**3, places=13)
        self.assertAlmostEqual(result.expected_score, 30000)
        self.assertAlmostEqual(result.clear_prob, 1)

    def test_floor_stitching_expansion_and_fractional_settlement(self):
        settings = replace(SS, area_clear_pts=dict.fromkeys(range(1, 11), 3000),
                           block_pts=dict.fromkeys(range(1, 11), 100))
        ten = [node(i, B.BOSS, area=i, col=1) for i in range(1, 11)]
        result = solve_route_dp(ten, settings, 5)
        self.assertEqual([n.area for n in result.best_route], list(range(1, 11)))
        self.assertEqual(result.score_distribution, {92000: 1.0})
        self.assertEqual(result.num_buckets, 460)
        long_map = [node(i, area=(i-1)//30 + 1, col=(i-1)%30, edges=[i+1] if i % 30 else [])
                    for i in range(1, 301)]
        self.assertEqual(solve_route_dp(long_map, settings, 5).score_distribution, {120000: 1.0})
        fractional = ScoreSettings(area_clear_pts={1: 0}, block_pts={1: 1}, score_rates={1: 125})
        nodes = [node(i, edges=[i+1] if i < 4 else []) for i in range(1, 5)]
        self.assertEqual(solve_route_dp(nodes, fractional, 1).score_distribution, {5: 1.0})
        graph = build_route_graph([node(1, edges=[3], last=True), node(2, area=2, col=1), node(3, area=2, col=1)])
        self.assertEqual(graph.children[1], (3,))
        with self.assertRaises(ValueError):
            build_route_graph([node(1, edges=[2]), node(2, edges=[1])])

    def test_ex_and_exclusive_bounds_at_opening(self):
        nodes = [node(1, edges=[2, 3]), node(2, B.NORMAL_BATTLE, [4], col=2),
                 node(3, B.EX_BATTLE, [4], col=2), node(4, B.BOSS)]
        result = score_opening_map(nodes, SS, 5)
        self.assertEqual([n.block_id for n in result.best_route], [1, 3, 4])
        self.assertEqual(result.hard_upper_bound, 200 + 2200 + 4200)
        self.assertEqual(result.expected_score, result.hard_upper_bound)
        reject = score_opening_map(nodes, SS, 5, target_score=6800, restart_time_sec=7)
        self.assertEqual((reject.decision, reject.should_reroll, reject.attempt_time_sec), ("RETIRE", True, 7))
        self.assertFalse(evaluate_early_stop(200, 400, 600).should_retire)

    def test_state_dependent_choice_and_unconditional_mass(self):
        nodes = [node(1, edges=[2, 3]), node(2, col=2), node(3, col=2)]
        specs = {1: spec({0: .5, 400: .5}), 2: spec(600), 3: spec(1000, win=.5)}
        result = solve_route_dp(nodes, SS, 5, specs, target_score=1000)
        self.assertEqual(result.policy[1, 0, False], 3)
        self.assertEqual(result.policy[1, 2, False], 2)
        self.assertEqual(result.score_distribution, {0: .25, 1000: .75})
        self.assertEqual((result.expected_score, result.clear_prob, result.prob_reach_target), (750, .75, .75))
        self.assertEqual(result.hard_upper_bound, 1400)
        # Expected-score optimization must also protect the score already earned.
        result = solve_route_dp(nodes[1:], SS, 5, {2: spec(600), 3: spec(1600, win=.5)}, confirmed_pts=1000)
        self.assertEqual(result.best_route[0].block_id, 2)

    def test_time_objective_and_rate_gate(self):
        nodes = [node(1, col=0), node(2, col=0)]
        specs = {1: spec(1000, 100, .9), 2: spec(1000, 2, .75)}
        fast = solve_route_dp(nodes, SS, 5, specs, target_score=1000, lambda_rate=.01)
        self.assertEqual((fast.best_route[0].block_id, fast.expected_time_sec, fast.prob_reach_target), (2, 2, .75))
        self.assertTrue(evaluate_early_stop(0, 1000, 1000, .75, 2, .4).should_retire)
        self.assertFalse(evaluate_early_stop(0, 1000, 1000, .75, 2, .375).should_retire)
        self.assertTrue(score_opening_map(nodes, SS, 5, specs, 1000, .4).should_reroll)
        self.assertFalse(evaluate_early_stop(0, 1000, 0, 0, 100, 5).should_retire)

    def test_revive_changes_survival_and_expected_attempt_time(self):
        nodes = [node(1, B.EX_BATTLE, [2]), node(2, B.BOSS)]
        specs = {1: NodeOutcomeSpec(.5, direct_pts_buckets=200, expected_time_sec=2, fail_time_sec=6),
                 2: spec(1000, 4, .5)}
        result = solve_route_dp(nodes, SS, 5, specs, revive_available=True)
        self.assertEqual(result.score_distribution, {0: .625, 1200: .375})
        self.assertEqual(result.expected_time_sec, 6 + .75 * 4)
        self.assertEqual(solve_route_dp(nodes, SS, 5, specs).clear_prob, .25)
        twice = [node(1, B.EX_BATTLE, [2]), node(2, B.NORMAL_BATTLE)]
        self.assertEqual(solve_route_dp(twice, SS, 5, specs, revive_available=True).clear_prob, .5)

    def test_full_override_without_default_floor_values_and_off_grid_scores(self):
        settings = ScoreSettings(score_rates={5: 200})
        result = solve_route_dp([node(1, edges=[2]), node(2)], settings, 5, {1: spec(125), 2: spec(50)})
        self.assertEqual((result.bucket_size, result.score_distribution), (25, {175: 1.0}))


class ShopTests(unittest.TestCase):
    discount = ShopDiscountModel(True, True)
    cheap = ShopItem("RELIC_1STAR")
    shelf = ShopShelfModel(((1, (cheap,)),))

    def test_price_floors_and_cap(self):
        items = [ShopItem(kind) for kind in ShopItemKind]
        self.assertEqual([self.discount.price(it.base_price) for it in items], [780, 1820, 1170, 1820, 2470, 6499])
        self.assertEqual([ShopDiscountModel().price(it.base_price) for it in items], [900, 2100, 1350, 2100, 2850, 7499])
        self.assertEqual(ShopDiscountModel(True, True, 90).price(1200), 120)

    def test_exact_subset_and_final_clearance(self):
        random = ShopItem("RANDOM_IMPRINT")
        items = (ShopItem("RELIC_3STAR"), random, self.cheap)
        opts = dict(discount_model=self.discount, shelf_model=self.shelf)
        self.assertEqual(clearance_ladder(items, 1950, discount_model=self.discount), (self.cheap, random))
        for dark in (dict(is_final_area=True), dict(refresh_available=False)):
            plan = plan_shop_visit(items, 1950, **opts, **dark)
            self.assertEqual((plan.purchases, plan.gold_remaining, plan.should_refresh), ((self.cheap, random), 0, False))
        # The best feasible subset can differ from a greedy ratio ordering.
        premium = ShopItem("RELIC_2STAR", extra_battle_pts=310)
        plan = plan_shop_visit([premium, self.cheap, self.cheap], 2340, **opts, is_final_area=True)
        self.assertEqual(plan.purchases, (self.cheap, self.cheap))
        self.assertEqual(plan.expected_pts, 800)
        # The second, also dark, shop cannot use a refresh to make up for skipping this shelf.
        plan = plan_shop_visit([self.cheap], 2640, **opts, is_final_area=True, remaining_shops=2)
        self.assertEqual((plan.purchases, plan.expected_pts), ((self.cheap,), 800))

    def test_increasing_fees_visible_then_empirical(self):
        opts = dict(discount_model=self.discount, shelf_model=self.shelf, max_refreshes=2)
        self.assertEqual(plan_shop_visit([], 3240, **opts).expected_pts, 800)
        self.assertEqual(plan_shop_visit([], 3239, **opts).expected_pts, 400)
        self.assertEqual(plan_shop_visit([], 3240, **opts).refresh_cost, 300)
        self.assertEqual(plan_shop_visit([], 1770, refresh_count=1, **opts).refresh_cost, 600)
        custom = replace(SS, shop_refresh_costs=[400, 900])
        self.assertEqual(plan_shop_visit([], 1490, custom, **opts).expected_pts, 0)
        empirical = ShopShelfModel.from_observations([[], [self.cheap], [self.cheap], [self.cheap]])
        self.assertEqual(plan_shop_visit([], 1470, discount_model=self.discount, shelf_model=empirical,
                                        max_refreshes=1).expected_pts, 300)

    def test_coin_retention_and_multishop_transfer(self):
        opts = dict(discount_model=self.discount, shelf_model=self.shelf, max_refreshes=0)
        for rid in (50003, 50009):
            state = PlanningState(relic_ids=(rid,), remaining_battles=(BattleExam(9, 90000, 3.5),))
            estimator = HeuristicBattlePassEstimator()
            def coin_value(wallet):
                return 90000 * estimator.estimate(replace(state, gold=wallet), state.remaining_battles[0]).lcb_prob
            self.assertEqual(plan_shop_visit([self.cheap], 10000, coin_retention_value_fn=coin_value, **opts).purchases, ())
            self.assertEqual(plan_shop_visit([self.cheap], 11170, coin_retention_value_fn=coin_value, **opts).expected_pts, 400)
        self.assertEqual([gold_to_pts_curve(2340, stores, **opts) for stores in (0, 1, 2)], [0, 400, 800])
        curve = [gold_to_pts_curve(g, 2, **opts) for g in range(0, 2900, 75)]
        self.assertEqual(curve, sorted(curve))
        self.assertEqual(curve, [gold_to_pts_curve(g, 2, **opts) for g in range(0, 2900, 75)])
        self.assertEqual(marginal_gold_pts(2339, 1, 2, **opts), 400)
        self.assertEqual(shop_score_distribution(2340, 2, **opts), {800: 1.0})
        plan = plan_shop_visit([ShopItem("RELIC_3STAR")], 2470, remaining_shops=3, **opts)
        self.assertEqual((plan.purchases, plan.expected_pts), ((), 800))

    def test_named_duplicate_imprints_and_single_final_floor(self):
        named = ShopItem("RANDOM_IMPRINT", unit_id=77)
        model = ShopShelfModel(((1, (named,)),))
        self.assertEqual(gold_to_pts_curve(3000, 3, shelf_model=model), 200)
        self.assertEqual(gold_to_pts_curve(3000, 3, shelf_model=model, owned_unit_ids={77}), 0)
        fractional = replace(SS, unit_pts=1, score_rates={5: 125})
        shelf = (ShopItem("RANDOM_IMPRINT"),) * 4
        self.assertEqual(plan_shop_visit(shelf, 3600, fractional, is_final_area=True).expected_pts, 5)


class MarginalTests(unittest.TestCase):
    estimator = HeuristicBattlePassEstimator()

    def test_duplicate_relic_awards_score_and_mark_only_once(self):
        marked = RelicEntry(101, "印记", 1, 1, 1, 1800)
        awarded = award_collections(PlanningState(), SS, relic_ids=(101, 101), entry=marked)
        self.assertEqual((awarded.relic_ids, awarded.confirmed_pts, awarded.mark_stacks),
                         ((101,), 400, (1, 0, 0, 0, 0)))
        duplicate, = evaluate_relic_choices(awarded, [marked], self.estimator)
        self.assertEqual((duplicate.direct_pts, duplicate.next_state), (0, awarded))

    def test_unmarked_orb_thresholds_and_saturation(self):
        orb = RelicEntry(30001, "元素涌动宝珠", 3, 0, 0, 3800, element_bonus_amp=1)
        marked = RelicEntry(101, "印记", 1, 1, 1, 1800)
        catalog = catalog_of(orb, marked)
        state = PlanningState(owned_unit_ids={7}, remaining_battles=(BattleExam(1, 6000),))
        choices = evaluate_relic_choices(state, [marked, 30001], self.estimator, catalog)
        self.assertEqual([c.candidate_id for c in choices], [30001, 101])
        self.assertGreater(choices[0].battle_pts, 0)
        self.assertEqual(choices[0].next_state.mark_stacks, (0,)*5)
        for mark in (3, 8, 14):
            s = replace(state, mark_stacks=(mark, 0, 0, 0, 0))
            rated, = evaluate_relic_choices(s, [101], self.estimator, catalog)
            self.assertGreater(rated.battle_pts, 0)
        strong = replace(state, battle_power=25)
        self.assertGreaterEqual(self.estimator.estimate(strong, strong.remaining_battles[0]).lcb_prob, .99)
        choices = evaluate_role_choices(strong, [7, 8], self.estimator)
        self.assertEqual([(c.candidate_id, c.value, c.direct_pts) for c in choices], [(8, 200, 200), (7, 0, 0)])

    def test_discounts_revive_and_scaling(self):
        shelf = ShopShelfModel(((1, (ShopItem("RELIC_1STAR"),)),))
        state = PlanningState(gold=1170, remaining_shops=1, remaining_battles=(BattleExam(12, 10000),))
        results = {c.candidate_id: c for c in evaluate_relic_choices(state, [40003, 50002], self.estimator, shelf_model=shelf)}
        self.assertEqual((results[40003].direct_pts, results[40003].shop_pts, results[40003].value), (400, 400, 800))
        self.assertGreater(results[50002].battle_pts, 0)
        boss = replace(state, remaining_battles=(BattleExam(12, 10000, is_boss=True),))
        self.assertEqual(evaluate_relic_choices(boss, [50002], self.estimator)[0].battle_pts, 0)
        for rid in (50003, 50009):
            probabilities = [pass_profile(replace(state, gold=g, relic_ids=(rid,)), self.estimator)[1]
                             for g in (0, 10000, 20000)]
            self.assertLess(probabilities[0], probabilities[1])
            self.assertEqual(probabilities[1], probabilities[2])
        populous = replace(state, owned_unit_ids=frozenset(range(10)))
        self.assertGreater(evaluate_relic_choices(populous, [50001], self.estimator)[0].battle_pts, 0)

    @staticmethod
    def curve(s):
        return min(s.gold, 4000)*.5 + min(4000, max(0, s.gold - 4000))*.1

    def test_slots_target_position_and_principal(self):
        state = PlanningState(gold=4000, confirmed_pts=50100, remaining_shops=1)
        def options(s, **kw):
            return evaluate_event_options(s, 3316, self.estimator, shop_curve=self.curve, **kw)
        self.assertEqual(options(state)[0].candidate_id, 33162)
        self.assertEqual(options(state, deduct_condition_principal=False)[0].candidate_id, 33161)
        behind = options(state, target_score=52500, lambda_rate=.01)[0]
        self.assertEqual((behind.candidate_id, behind.prob_reach_target, behind.value), (33161, .5, .49))
        ahead = options(replace(state, confirmed_pts=50500), target_score=52500, lambda_rate=.01)[0]
        self.assertEqual((ahead.candidate_id, ahead.prob_reach_target), (33162, 1))
        self.assertEqual([c.candidate_id for c in options(replace(state, gold=3999), deduct_condition_principal=False)], [33162])

    def test_slots_protect_gold_scaled_battle_power_and_use_catalog_principal(self):
        state = PlanningState(gold=10000, remaining_shops=1, remaining_battles=(BattleExam(7, 20000, 3.5),))
        def rank(s):
            return evaluate_event_options(s, 3316, self.estimator, shop_curve=lambda st: st.gold * .2)
        self.assertEqual(rank(state)[0].candidate_id, 33161)
        for rid in (50003, 50009):
            self.assertEqual(rank(replace(state, relic_ids=(rid,)))[0].candidate_id, 33162)
        entry = EventChoiceEntry(3316, 33161, 1, 6000, 0, "继续")
        c, = evaluate_event_options(state, [entry], self.estimator)
        self.assertEqual(sorted(o.state.gold for o in c.outcomes), [4500, 12000])

    def test_multistage_probability_and_observed_continuations(self):
        state = PlanningState(confirmed_pts=50100, remaining_shops=1)
        play, = evaluate_event_options(state, 1316, self.estimator, shop_curve=self.curve, target_score=52500)
        self.assertEqual(play.prob_reach_target, .125)
        self.assertEqual(play.expected_time_sec, 1.75)
        self.assertEqual(sorted((o.state.gold, o.probability) for o in play.outcomes),
                         [(500, .125), (500, .25), (500, .5), (8000, .125)])
        self.assertIn((13161, 23161, 33161), [o.choices for o in play.outcomes])
        tree = {77: (EventOption(771, (EventBranch(.25, next_event_id=78), EventBranch(.75, 500))),),
                78: (EventOption(781, (EventBranch(relic_ids=(30001,)),)),)}
        result, = evaluate_event_options(PlanningState(), 77, self.estimator, event_tree=tree)
        self.assertEqual(result.direct_pts, 100)
        self.assertTrue(any(30001 in o.state.relic_ids for o in result.outcomes))
        self.assertTrue({1107, 1105, 1109, 1113, 1117, 3314, 3315} <= default_event_tree().keys())

    def test_threshold_integrates_pmf_not_threshold_of_mean(self):
        class Dist:
            def __call__(self, state):
                return {0: .75, 800: .25} if state.gold else {200: 1.0}
        state = PlanningState(confirmed_pts=52100)
        choices = [EventOption(1), EventOption(2, (EventBranch(gold_delta=1),))]
        rated = evaluate_event_options(state, choices, self.estimator, target_score=52500,
                                      shop_distribution_fn=Dist(), future_score_distribution={0: .5, 200: .5})
        self.assertEqual([(c.candidate_id, c.prob_reach_target) for c in rated], [(1, .5), (2, .25)])


class ReplayTests(unittest.TestCase):
    def test_event_prizes_and_a_single_wallet_across_future_shops(self):
        boss = node(3, B.BOSS)
        stone = MapReplayEpisode((node(1, B.EVENT, [3], quest=1107), boss), initial_counts=ScoreCounts(units=227),
                                 initial_state=PlanningState(gold=2000, battle_power=30))
        result = run_episode(stone, PathBReplayPolicy(50000))
        self.assertEqual((result.event_choices, result.status, result.final_score), ((11071,), "clear", 50200))
        unfunded = replace(stone, initial_state=replace(stone.initial_state, gold=0))
        result = run_episode(unfunded, PathBReplayPolicy())
        self.assertEqual((result.event_choices, result.status, result.final_score), ((11073,), "clear", 49800))
        self.assertEqual(run_episode(unfunded, PathBReplayPolicy(50000)).status, "opening_reroll")
        battle_reward = replace(stone, nodes=(node(1, B.NORMAL_BATTLE, [3]), boss),
                                initial_counts=ScoreCounts(units=226),
                                node_data={1: NodeReplayData(relic_choices=(30001,))})
        self.assertEqual(run_episode(battle_reward, PathBReplayPolicy(50000)).final_score, 50200)
        shops = (node(1, B.SHOP, [2]), node(2, B.SHOP, [3]), boss)
        shelf = ShopShelfModel(((1, (ShopItem("RELIC_1STAR"),)),))
        budget = replace(stone, nodes=shops, shelf_model=shelf, initial_counts=ScoreCounts(units=226),
                         initial_state=PlanningState(gold=1350, battle_power=30))
        self.assertEqual(run_episode(budget, PathBReplayPolicy(50200)).final_score, 50200)
        self.assertEqual(run_episode(budget, PathBReplayPolicy(50400)).status, "opening_reroll")
        funded = replace(budget, nodes=(node(1, B.EVENT, [2], quest=77), *shops[1:]),
                         event_tree={77: (EventOption(771, (EventBranch(.5, 1350), EventBranch(.5, 500))),)},
                         initial_counts=ScoreCounts(units=225), initial_state=PlanningState(battle_power=30),
                         node_data={1: NodeReplayData(event_rolls=(.25,))})
        self.assertEqual(run_episode(funded, PathBReplayPolicy(50000)).final_score, 50000)
        lost = replace(funded, node_data={1: NodeReplayData(event_rolls=(.75,))})
        self.assertEqual(run_episode(lost, PathBReplayPolicy(50000)).status, "midrun_retire")

    def test_desktop_full_map_preferred_to_saved_path_and_sample_timing(self):
        nodes = [dict(blockId=i, blockType=int(kind), area=1, column=col, row=0, nextBlockIds=edges,
                      questId=91 if kind == B.EX_BATTLE else None, isAreaLastPoint=False)
                 for i, kind, col, edges in ((1, B.START, 1, [2, 3]), (2, B.NORMAL_BATTLE, 2, []),
                                            (3, B.EX_BATTLE, 2, []))]
        run = SingleRunResult(13, 0, 0, 420, 90, 7, .9, 50, 1)
        sample = BattleSample(sample_id="v", round_id="r", source="live", db_version="", quest_id=91,
                              canonical_wave_sig="w", bucket="EX", area=1, difficulty=5, guild_id=1,
                              team_unit_ids=[], owned_unit_ids=[], owned_unit_count=0, owned_role_counts=[0]*8,
                              owned_talent_counts=[0]*5, relic_ids=[], mark_stacks=[0]*5, alpha_rupee=0,
                              outcome_counts=[1, 0, 0], runs=[run])
        episode = MapReplayEpisode.from_dict({"nodes": nodes[:2], "allNodes": nodes, "difficulty": 5,
                                              "battle_samples": [sample.to_dict()]})
        outcome = run_episode(episode, PathBReplayPolicy())
        self.assertEqual((outcome.visited_blocks, outcome.elapsed_sec, outcome.final_score), ((1, 3), 8, 3400))

    def test_desktop_database_loads_uncheckpointed_wal_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "desktop.db"
            nodes = [dict(blockId=1, area=1, column=1, row=0, blockType=int(B.START))]
            with closing(sqlite3.connect(path)) as writer:
                self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone(), ("wal",))
                writer.execute("CREATE TABLE labyrinth_routes (route_json TEXT NOT NULL)")
                writer.execute("INSERT INTO labyrinth_routes VALUES (?)", (json.dumps(dict(enterId=1, nodes=nodes)),))
                writer.commit()
                writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                self.assertEqual([ep.episode_id for ep in load_replays(path)], ["1"])
                writer.execute("INSERT INTO labyrinth_routes VALUES (?)", (json.dumps(dict(enterId=2, nodes=nodes)),))
                writer.commit()
                self.assertGreater(path.with_suffix(".db-wal").stat().st_size, 0)
                self.assertEqual([ep.episode_id for ep in load_replays(path)], ["1", "2"])

    def test_four_outcomes_and_paired_benchmark(self):
        nodes = (node(1, edges=[2, 3]), node(2, B.NORMAL_BATTLE, [4], col=2),
                 node(3, B.EX_BATTLE, [4], col=2), node(4, B.BOSS))
        specs = {1: spec(200), 2: spec(400, 2), 3: spec(2200, 3), 4: spec(4200, 5)}
        good = MapReplayEpisode(nodes, "clear", initial_counts=ScoreCounts(units=240), outcome_specs=specs, restart_time_sec=7)
        wipe = replace(good, episode_id="wipe", node_data={4: NodeReplayData(battle_results=(BattleObservation(2, 4),))})
        opening = replace(good, episode_id="opening", nodes=(nodes[-1],), outcome_specs={4: specs[4]})
        mid = replace(good, episode_id="mid", nodes=(node(1, B.LINK, [4]), nodes[-1]),
                      initial_state=PlanningState(owned_unit_ids={10}), outcome_specs={1: spec(1000, 2), 4: specs[4]},
                      node_data={1: NodeReplayData(role_choices=(10,))})
        maps = [good, wipe, opening, mid]
        path = PathBReplayPolicy(53000)
        outcomes = [run_episode(ep, path) for ep in maps]
        self.assertEqual([o.status for o in outcomes], ["clear", "wipe", "opening_reroll", "midrun_retire"])
        self.assertEqual([o.final_score for o in outcomes], [54600, 0, 0, 0])
        self.assertEqual([o.elapsed_sec for o in outcomes], [9, 8, 7, 9])
        self.assertEqual(outcomes[1].ledger_score, 50400)
        reports = compare_policies(maps, path, repeats=2, seed=91)
        for r in reports.values():
            self.assertEqual(r.total_attempts, 8)
            self.assertEqual(r.clears + r.wipes + r.opening_rerolls + r.midrun_retires, r.total_attempts)
        b = reports["path_b"]
        self.assertEqual((b.expected_score, b.p_ge_50k, b.p_ge_53k, b.clear_rate, b.abandon_rate),
                         (54600/4, .25, .25, .25, .5))
        self.assertEqual((b.mean_time_per_target_hit_sec, b.mean_time_per_attempt_sec), (33, 8.25))
        baseline = reports["baseline"]
        self.assertEqual((baseline.expected_score, baseline.p_ge_50k, baseline.p_ge_53k,
                          baseline.clear_rate, baseline.abandon_rate), (39350, .75, 0, .75, 0))
        self.assertTrue(math.isinf(baseline.mean_time_per_target_hit_sec))
        self.assertEqual(BenchmarkReport.from_outcomes([]).total_attempts, 0)

    def test_protocol_map_replay_rewards_shops_event_and_retry(self):
        rows = [dict(block_id=i, area=1 if i <= 3 else 2, column=i, row=0, block_type=int(kind),
                     next_block_id_list=[i+1] if i not in (3, 7) else [], IsAreaLastPoint=i in (3, 7))
                for i, kind in enumerate((B.RELIC, B.NORMAL_BATTLE, B.SHOP, B.EVENT, B.LINK, B.SHOP, B.BOSS), 1)]
        packet = dict(data=dict(enter_id=19, map_list=rows), initial_state=dict(relic_ids=[50002], gold=10000),
                      max_refreshes=1, shelf_model=[[1, ["RELIC_1STAR"]]], node_data={
                          "1": dict(relic_choices=[101, 30001, 102]),
                          "2": dict(battle_results=[dict(outcome_class=2, elapsed_sec=4), dict(outcome_class=0, elapsed_sec=3)]),
                          "3": dict(shop_shelves=[[], [dict(kind="RELIC_1STAR", relic_id=44)]]),
                          "4": dict(event_id=1316, event_rolls=[0, 0, 0]),
                          "5": dict(role_choices=[8, 9, 10]),
                          "6": dict(shop_shelves=[[dict(kind="RANDOM_IMPRINT", unit_id=25)]]),
                          "7": dict(battle_results=[dict(outcome_class=0, elapsed_sec=2)])})
        episode = MapReplayEpisode.from_dict(packet)
        result = run_episode(episode, PathBReplayPolicy())
        self.assertEqual(result.status, "clear")
        self.assertEqual(result.visited_blocks, tuple(range(1, 8)))
        self.assertEqual(result.event_choices[:2], (13161, 23162))
        self.assertEqual(result.shop_refreshes, 1)
        self.assertEqual(result.state.revive_used_count, 1)
        self.assertEqual(result.state.relic_ids, (50002, 30001, 44))
        self.assertEqual(result.state.owned_unit_ids, {8, 25})
        self.assertEqual((result.counts.normal_wins, result.counts.boss_wins), (1, 1))
        self.assertEqual(result.final_score, compute_final_score(SS, 5, result.counts))
        self.assertEqual(result.state.confirmed_pts, result.final_score)
        self.assertEqual(result.elapsed_sec, 15)
        boss = replace(episode, nodes=(episode.nodes[-1],), node_data={7: NodeReplayData(battle_results=(BattleObservation(1, 7),))})
        failed = run_episode(boss, BaselineHeuristicPolicy())
        self.assertEqual((failed.status, failed.state.revive_used_count, failed.elapsed_sec), ("wipe", 0, 7))


if __name__ == "__main__":
    unittest.main()
