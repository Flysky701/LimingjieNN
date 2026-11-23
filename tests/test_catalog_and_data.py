"""Worker A tests using SQLite, loopback HTTP and any local CN master copies."""
from __future__ import annotations

import errno
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import statistics
import tempfile
import threading
import unittest
import uuid
from collections import Counter, defaultdict
from contextlib import closing, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch
from urllib.error import URLError

from labyrinth_nn.catalog.build_catalog import build_catalog
from labyrinth_nn.catalog.schema import ENEMY_STATIC_FEAT_NAMES, UNIT_MECH_FLAG_NAMES, UNIT_STATIC_FEAT_NAMES
from labyrinth_nn.data import probe_simulator, sampler
from labyrinth_nn.data.dataset import split_samples
from labyrinth_nn.data.sample_schema import BattleSample, OUTCOME_TIMEOUT, OUTCOME_WIN, SingleRunResult, read_samples
from labyrinth_nn.data.sim_client import (
    PcrCalculatorClient, SimClient, SimulationConfig, SimulationError, SimulationTimeoutError, build_simulate_payload,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/sim_profile_highscore_cn.json"
SAMPLING = ROOT / "configs/sampling_round0.json"
REAL_DATABASES = (
    Path(r"D:\Project\LimingjieAssistant\data\files\labyrinth\master-cn\master_cn.db"),
    Path(r"D:\Project\LimingjieDesktop\out\便携包测试\47d3e92\LimingjieAssistant\data\files\labyrinth\master-cn\master_cn.db"),
)
NORMAL = 770110101
SAME_WAVE = 770110102
REORDERED_WAVE = 770110201
ROBOT_EX = 770120201
ENVIRONMENT = 770900501
PROFILE_ENVIRONMENT = 770777777


def create_fixture(path: Path) -> None:
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            PRAGMA user_version = 7;
            CREATE TABLE labyrinth_npc_unit_data (unit_id INT, enhance_id INT);
            CREATE TABLE labyrinth_npc_unit_enhance_data
                (enhance_id INT, rarity INT, unit_level INT, promotion_level INT);
            INSERT INTO labyrinth_npc_unit_enhance_data VALUES (1001,3,60,7),(1002,6,70,8);
            CREATE TABLE experience_unit (unit_level INT);
            INSERT INTO experience_unit VALUES (50),(100);
            CREATE TABLE unit_data (unit_id INT, unit_name TEXT, search_area_width INT, atk_type INT);
            CREATE TABLE unit_talent (unit_id INT, talent_id INT);
            CREATE TABLE unit_role_data (unit_id INT, unit_role_id INT);
            CREATE TABLE unit_rarity
                (unit_id INT, rarity INT, hp REAL, hp_growth REAL, atk REAL, magic_str REAL);
            INSERT INTO unit_rarity VALUES (1,6,100,2,100,0),(1,3,9999,0,9999,0);
            CREATE TABLE unit_promotion_status
                (unit_id INT, promotion_level INT, hp REAL, def REAL, magic_def REAL);
            INSERT INTO unit_promotion_status VALUES (1,9,50,100,150),(1,8,9999,9999,9999);
            CREATE TABLE unit_unique_equipment (unit_id INT, equip_slot INT, equip_id INT);
            INSERT INTO unit_unique_equipment VALUES (1,1,101),(2,1,0);
            CREATE TABLE unit_unique_equip (unit_id INT, equip_slot INT, equip_id INT);
            INSERT INTO unit_unique_equip VALUES (1,2,102);
            CREATE TABLE unit_skill_data
                (unit_id INT, union_burst INT, main_skill_10 INT, sp_skill_evolution_3 INT, ex_skill_5 INT);
            INSERT INTO unit_skill_data VALUES (1,10,20,30,0),(2,10,0,0,0),(3,0,0,0,40);
            CREATE TABLE skill_data (
                skill_id INT, action_1 INT, action_2 INT, action_3 INT, action_4 INT,
                action_5 INT, action_6 INT, action_7 INT, action_8 INT, action_9 INT, action_10 INT
            );
            INSERT INTO skill_data VALUES
                (10,101,0,0,0,0,0,0,0,0,0),(20,102,103,104,105,106,107,0,0,0,108),
                (30,109,0,0,0,0,0,0,0,0,0),(40,110,111,0,0,0,0,0,0,0,0);
            CREATE TABLE skill_action
                (action_id INT, action_type INT, action_detail_1 INT, description TEXT, target_range INT, target_area INT);
            INSERT INTO skill_action VALUES
                (101,6,2,'降低敌人技能值',300,2),(102,116,0,'',0,0),(103,73,0,'',0,0),
                (104,52,0,'',0,0),(105,16,2,'',0,0),(106,48,0,'',0,0),(107,58,0,'',0,0),
                (108,36,0,'',0,2),(109,99,0,'削减目标TP',0,0),
                (110,99,0,'免疫自身技能值降低效果量',0,0),(111,1,0,'',0,1);
            CREATE TABLE labyrinth_skill_action_replace (action_id INT, action_type INT);
            INSERT INTO labyrinth_skill_action_replace VALUES (101,79);
            CREATE TABLE labyrinth_enemy_parameter (
                enemy_id INT, unit_id INT, name TEXT, level INT, hp INT, virtual_hp INT,
                atk INT, magic_str INT, def INT, magic_def INT, dodge INT, accuracy INT, resist_status_id INT
            );
            INSERT INTO labyrinth_enemy_parameter VALUES
                (201,301,'甲',100,100,7,1,2,10,11,0,0,1),
                (202,302,'乙',100,300,0,2,3,20,21,0,0,2),
                (211,301,'甲',200,250,7,1,2,10,11,0,0,1),
                (203,303,'黄蜂女王',300,500,0,2,3,30,31,0,0,0),
                (204,304,'狙击·遗物',200,500,0,2,3,30,31,0,0,0),
                (205,305,'普通EX',200,500,0,2,3,30,31,0,0,0),
                (206,306,'3F首领',399,600,0,2,3,30,31,0,0,0),
                (207,306,'3F首领',400,800,0,2,3,30,31,0,0,0),
                (208,307,'奇美拉',500,900,0,2,3,30,31,0,0,0),
                (209,307,'奇美拉',550,3000000000,0,2,3,30,31,0,0,0),
                (770900501,1,'环境',1000,99999999,0,999,400,500,300,0,0,0);
            CREATE TABLE ailment_data (ailment_id INT, ailment_action INT, ailment_detail_1 INT);
            INSERT INTO ailment_data VALUES (90,79,1),(91,8,2),(92,8,1);
            CREATE TABLE resist_data (resist_status_id INT, ailment_90 REAL, ailment_91 REAL, ailment_92 REAL, ailment_10 REAL, ailment_14 REAL);
            INSERT INTO resist_data VALUES (1,1,1,0,1,0),(2,-1,0,1,0,1);
            CREATE TABLE labyrinth_difficulty_enemy_data (difficulty INT, enemy_id_5 INT);
            INSERT INTO labyrinth_difficulty_enemy_data VALUES (5,770900501);
            CREATE TABLE labyrinth_wave_group_data (wave_group_id INT, enemy_id_1 INT, enemy_id_2 INT, enemy_id_3 INT);
            INSERT INTO labyrinth_wave_group_data VALUES
                (1,201,770900501,202),(2,202,211,770900501),(3,203,0,0),(4,204,0,0),
                (5,205,0,0),(6,206,0,0),(7,207,0,0),(8,208,0,0),(9,209,770900501,0),(10,770900501,0,0);
            CREATE TABLE labyrinth_quest_data (quest_id INT, wave_group_id INT, quest_type INT, limit_time INT, reward_id INT);
            INSERT INTO labyrinth_quest_data VALUES
                (770110101,1,1,90,1),(770110102,1,1,90,1),(770110201,2,1,90,1),
                (770120101,3,2,90,2),(770120201,4,2,90,2),(770120301,5,2,90,2),(770220301,5,2,90,2),
                (770330101,6,3,90,3),(770331601,7,3,90,3),(770530101,8,3,90,3),(770531601,9,3,90,3),
                (770110301,10,1,90,1),(770110401,999,1,90,1);
            CREATE TABLE labyrinth_ex_quest_data_display (quest_id INT, quest_name TEXT);
            INSERT INTO labyrinth_ex_quest_data_display VALUES (770120101,'专用显示');
            CREATE TABLE labyrinth_quest_data_display (quest_id INT, quest_name TEXT);
            INSERT INTO labyrinth_quest_data_display VALUES (770120101,'通用显示'),(770110101,'普通关卡');
            CREATE TABLE labyrinth_relic (
                relic_id INT, name TEXT, rarity INT, relic_mark_id INT, relic_mark_count INT, price INT, effect_description TEXT
            );
            INSERT INTO labyrinth_relic VALUES
                (30001,'元素',3,0,0,0,'属性加成＋100％'),(40002,'增幅',3,4,1,0,''),
                (40003,'折扣',3,1,1,0,'商店的贩卖价格－10％'),(40009,'初始金币',3,3,1,0,'初始阿尔法金币+3000'),
                (50001,'人数',3,4,1,0,'攻击力提升（加入角色数×4％）'),(50002,'复活',3,5,1,0,''),
                (50003,'金币',3,3,1,0,'阿尔法金币持有数×0.01'),(50009,'金币2',3,3,1,0,'金币×0.01'),
                (50010,'人数2',3,2,1,0,'加入角色数'),(34001,'物防',3,4,3,0,'前排角色的物理防御力×15'),
                (34002,'魔防',3,4,3,0,'前排魔防 ×10'),(35001,'元素2',3,5,3,0,'属性加成＋25％');
            CREATE TABLE labyrinth_enter_guild (guild_id INT, guild_name TEXT, guild_master_id INT, add_relic_id INT);
            INSERT INTO labyrinth_enter_guild VALUES (1,'美食殿堂',8,50001),(2,'破晓之星',9,50002),(3,'咲恋救济院',10,50003);
            CREATE TABLE labyrinth_enhance_node (node_id INT);
            INSERT INTO labyrinth_enhance_node VALUES (11),(12);
            CREATE TABLE labyrinth_setting (setting_id INT, value INT);
            INSERT INTO labyrinth_setting VALUES (101,330),(201,550),(211,150),(221,101),(222,202),(223,103),(224,1005),(225,1550);
            CREATE TABLE labyrinth_difficulty_data (difficulty INT, score_rate INT);
            INSERT INTO labyrinth_difficulty_data VALUES (1,100),(5,200);
            CREATE TABLE labyrinth_event (event_id INT, choice_id_1 INT, choice_id_2 INT);
            INSERT INTO labyrinth_event VALUES (42,421,422);
            CREATE TABLE labyrinth_event_choice (choice_id INT, condition_type INT, condition_value INT, unit_id INT, description TEXT);
            INSERT INTO labyrinth_event_choice VALUES (421,1,2,1,'选项'),(422,0,0,0,'离开');
            CREATE TABLE labyrinth_event_result (event_result_id INT);
            INSERT INTO labyrinth_event_result VALUES (4212),(4211),(4221);
            CREATE TABLE talent (talent_id INT);
            INSERT INTO talent VALUES (1),(2);
            CREATE TABLE experience_talent_level (talent_level INT);
            INSERT INTO experience_talent_level VALUES (2),(1000);
            CREATE TABLE talent_skill_node (node_id INT, enhance_level_id_1 INT, enhance_level_id_2 INT, enhance_level_id_3 INT);
            INSERT INTO talent_skill_node VALUES (11,12,0,0),(12,13,14,0);
            CREATE TABLE unit_role_mastery_id (unit_role_id INT, mastery_id INT);
            INSERT INTO unit_role_mastery_id VALUES (1,10102),(0,11111),(4,99999);
            CREATE TABLE unit_role_mastery_level (mastery_id INT, slot_level INT, enhance_level INT);
            INSERT INTO unit_role_mastery_level VALUES (10102,3,2),(10102,4,5),(11111,5,5),(99999,5,5);
        """)
        for uid in range(1, 41):
            db.execute("INSERT INTO labyrinth_npc_unit_data VALUES (?,?)", (uid, 1002 if uid == 1 else 1001))
            db.execute("INSERT INTO unit_data VALUES (?,?,?,?)", (uid, f"角色{uid}", uid * 10, uid % 2 + 1))
            db.execute("INSERT INTO unit_talent VALUES (?,?)", (uid, uid % 5 + 1))
            db.execute("INSERT INTO unit_role_data VALUES (?,?)", (uid, uid % 8 + 1))
            if uid != 1:
                db.execute("INSERT INTO unit_rarity VALUES (?,?,?,?,?,?)", (uid, 5, 200, 0, uid, uid * 2))
        db.executemany("INSERT INTO labyrinth_relic VALUES (?,?,?,?,?,?,?)", [
            (10000 + mark * 100 + index, f"遗物{mark}-{index}", 1, mark, width, 100, "")
            for mark in range(1, 6) for index, width in enumerate((1, 1, 1, 2, 2, 2, 3, 3, 5), 1)
        ])


@contextmanager
def mock_server(catalog, *, respond=None, withhold_response=None):
    requests = []
    lock = threading.Lock()
    next_seed = 4000

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            nonlocal next_seed
            if self.path != "/simulate":
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append(payload)
                result = []
                for query in payload:
                    settings = query["Settings"]
                    next_seed += 1
                    seed = settings["RandomSeed"] if settings["UseFixedRandomSeed"] else next_seed
                    enemies = [{"enemyid": eid, "currentHp": catalog.enemies[eid].hp // 2}
                               for eid in reversed(query["GuildEnemyList"])]
                    enemies.extend({"enemyId": eid, "currentHp": 1}
                                   for eid in (ENVIRONMENT, PROFILE_ENVIRONMENT, 770999901))
                    row = {
                        "Success": True, "EnemyList": enemies, "Result": 2, "BattleFrames": 5399,
                        "CurrentSeed": seed, "Stats": {"TotalDamage": seed * 100},
                    }
                    result.append(respond(query, row, len(requests)) if respond is not None else row)
            if withhold_response is not None:
                withhold_response.wait(timeout=5)
                return
            encoded = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            thread.join()


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Python 3.13's Windows 0700 ACL can exclude the sandbox identity.
        # Inherit the system temp ACL using the default mkdir mode.
        cls.root = Path(tempfile.gettempdir()) / f"labyrinth-nn-test-{uuid.uuid4().hex}"
        cls.root.mkdir()

        def cleanup():
            for file in cls.root.iterdir():
                file.unlink()
            cls.root.rmdir()

        cls.addClassCleanup(cleanup)
        cls.db_path = cls.root / "合成迷宫资料.sqlite"
        create_fixture(cls.db_path)
        cls.catalog = build_catalog(cls.db_path)
        cls.profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        cls.profile["exclude_environment_enemy_ids"].append(PROFILE_ENVIRONMENT)
        cls.config = json.loads(SAMPLING.read_text(encoding="utf-8"))


class CatalogTests(FixtureCase):
    def test_unit_replacements_mechanisms_and_panel(self):
        c = self.catalog
        first, second, third = (c.units[i] for i in (1, 2, 3))
        self.assertEqual(tuple(first.mech_flags), UNIT_MECH_FLAG_NAMES)
        self.assertTrue(all(first.mech_flags.values()))
        self.assertTrue(first.has_6star and first.has_ue1 and first.has_ue2 and first.has_labyrinth_replace)
        self.assertEqual([key for key, value in second.mech_flags.items() if value], ["dot_slip"])
        self.assertTrue(second.has_labyrinth_replace)
        self.assertFalse(any(third.mech_flags.values()))
        self.assertEqual(first.panel_hp, 368.0)
        self.assertEqual(first.panel_def, 100.0)
        self.assertEqual(first.panel_energy_reduce_rate, 0.0)
        self.assertEqual(first.static_feats[:7], [0.01, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(first.static_feats[-10:], [1.0] * 10)
        self.assertTrue(all(len(u.static_feats) == len(UNIT_STATIC_FEAT_NAMES) == 27 for u in c.units.values()))
        self.assertAlmostEqual(statistics.fmean(u.static_feats[7] for u in c.units.values()), 0)
        self.assertAlmostEqual(statistics.pstdev(u.static_feats[7] for u in c.units.values()), 1)
        self.assertTrue(all(u.static_feats[12] == 0.0 for u in c.units.values()))

    def test_rf_skill_actions_include_replacements_and_multiple_evolutions(self):
        path = self.root / "rf-skills.sqlite"
        create_fixture(path)
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE unit_skill_data_rf (id INT, skill_id INT, rf_skill_id INT);
                INSERT INTO unit_skill_data_rf VALUES (400,40,41),(401,40,42),(402,0,20),(403,40,0);
                INSERT INTO skill_data (skill_id, action_10) VALUES (41,112),(42,107);
                INSERT INTO skill_action VALUES (112,6,0,'',0,0);
                INSERT INTO labyrinth_skill_action_replace VALUES (112,79);
            """)
        self.assertFalse(self.catalog.units[3].has_labyrinth_replace)
        unit = build_catalog(path).units[3]
        self.assertTrue(unit.has_labyrinth_replace)
        self.assertEqual([key for key, value in unit.mech_flags.items() if value], ["dot_slip", "dispel"])
        self.assertEqual(unit.static_feats[6], 1.0)

    def test_quest_identity_ex_index_and_environment_stats(self):
        c = self.catalog
        self.assertEqual(c.environment_enemy_ids, [ENVIRONMENT])
        self.assertEqual(c.quests[NORMAL].wave_total_hp, 400)
        signatures = {c.quests[qid].canonical_wave_sig for qid in (NORMAL, SAME_WAVE, REORDERED_WAVE)}
        pairing = json.dumps([(301, "甲"), (302, "乙")], ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(signatures, {hashlib.sha256(pairing.encode()).hexdigest()})
        self.assertEqual(c.quests[770530101].canonical_wave_sig, c.quests[770531601].canonical_wave_sig)
        self.assertNotIn(770110301, c.quests)
        self.assertNotIn(770110401, c.quests)
        self.assertEqual(c.ex_disambiguation_index, {
            "1:黄蜂女王": [770120101], "1:狙击·遗物": [770120201],
            "1:普通EX": [770120301], "2:普通EX": [770220301],
        })
        self.assertEqual(c.quests[770120101].quest_name, "专用显示")
        self.assertEqual(c.quests[NORMAL].quest_name, "普通关卡")
        self.assertEqual([q.bucket for q in c.quests.values() if q.is_difficulty5_boss], ["boss_3f_d5", "boss_5f_d5"])
        self.assertTrue(c.enemies[206].is_boss_unit)
        self.assertTrue(c.enemies[201].has_dot_resist)
        self.assertFalse(c.enemies[201].has_cc_resist)
        self.assertFalse(c.enemies[202].has_dot_resist)
        self.assertTrue(c.enemies[202].has_cc_resist)
        self.assertTrue(all(len(e.static_feats) == len(ENEMY_STATIC_FEAT_NAMES) == 11 for e in c.enemies.values()))
        zscores = [e.static_feats[1] for eid, e in c.enemies.items() if eid != ENVIRONMENT]
        self.assertAlmostEqual(statistics.fmean(zscores), 0)
        self.assertAlmostEqual(statistics.pstdev(zscores), 1)
        visible_log_hps = [math.log1p(e.hp) for eid, e in c.enemies.items() if eid != ENVIRONMENT]
        env_log_hp = math.log1p(c.enemies[ENVIRONMENT].hp)
        self.assertAlmostEqual(c.enemies[ENVIRONMENT].static_feats[1],
                               (env_log_hp - statistics.mean(visible_log_hps)) / statistics.pstdev(visible_log_hps))

    def test_ailment_fallback(self):
        path = self.root / "fallback.sqlite"
        create_fixture(path)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("DROP TABLE ailment_data")
        fallback = build_catalog(path)
        self.assertTrue(fallback.enemies[201].has_dot_resist)
        self.assertTrue(fallback.enemies[202].has_cc_resist)

    def test_relic_training_scores_and_events(self):
        c = self.catalog
        self.assertEqual(c.db_version, "7")
        self.assertEqual(c.catalog_sha256, c.compute_sha256())
        self.assertEqual({r.relic_id for r in c.relics.values() if r.gold_scaling}, {50003, 50009})
        self.assertEqual({r.relic_id for r in c.relics.values() if r.owned_unit_scaling}, {50001, 50010})
        self.assertEqual(c.relics[34001].front_pdef_to_atk_mult, 15)
        self.assertEqual(c.relics[34002].front_mdef_to_atk_mult, 10)
        self.assertEqual([c.relics[i].element_bonus_amp for i in (30001, 40002, 35001)], [1.0, 0.5, 0.25])
        self.assertTrue(c.relics[50002].is_revive)
        self.assertTrue(c.relics[40003].is_shop_discount)
        self.assertEqual(c.relics[40003].shop_discount_rate, 0.1)
        self.assertEqual(c.enhance_node_ids, [11, 12])
        self.assertEqual(c.guilds[3].add_relic_id, 50003)
        self.assertEqual(c.score_settings.area_clear_pts, {1: 550})
        self.assertEqual(c.score_settings.block_pts, {1: 150})
        self.assertEqual(c.score_settings.shop_refresh_costs[0], 330)
        self.assertEqual(c.score_settings.score_rates[5], 200)
        self.assertEqual(c.event_choices[421].result_ids, [4211, 4212])
        self.assertEqual(c.event_choices[421].event_id, 42)
        self.assertEqual(c.max_account_training, {
            "talent_skill_levels": {"1": 1000, "2": 1000}, "talent_skill_node_levels": {"11": 1, "12": 2},
            "role_mastery": {"10102": [4, 5]},
        })


class RealCatalogTests(unittest.TestCase):
    def test_real_cn_catalog(self):
        candidates = [ROOT / "artifacts/colab_upload/LimingjieNN/master_cn.db", *REAL_DATABASES]
        if configured := os.environ.get("CN_DB_PATH"):
            candidates.insert(0, Path(configured))
        paths = dict.fromkeys(path.resolve() for path in candidates if path.is_file())
        if not paths:
            self.skipTest("no supported CN master database is present")
        for path in paths:
            with self.subTest(database=str(path)):
                c = build_catalog(path)
                with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
                    roster = {r[0] for r in db.execute("SELECT unit_id FROM labyrinth_npc_unit_data")}
                    enemies = {r[0] for r in db.execute("SELECT enemy_id FROM labyrinth_enemy_parameter")}
                    quests = {r[0] for r in db.execute("SELECT quest_id FROM labyrinth_quest_data")}
                    relics = {r[0] for r in db.execute("SELECT relic_id FROM labyrinth_relic")}
                    mastery_ids = {r[0] for r in db.execute("""
                        SELECT DISTINCT i.mastery_id
                        FROM unit_role_mastery_id i JOIN unit_role_mastery_level l USING (mastery_id)
                        WHERE i.unit_role_id != 0 AND i.mastery_id != 99999
                    """)}
                    for row in db.execute("""
                        SELECT q.quest_id, w.enemy_id_1, w.enemy_id_2, w.enemy_id_3, w.enemy_id_4, w.enemy_id_5
                        FROM labyrinth_quest_data q JOIN labyrinth_wave_group_data w USING (wave_group_id)
                    """):
                        self.assertEqual(c.quests[row[0]].enemy_ids, [eid for eid in row[1:] if eid])
                self.assertEqual(set(c.units), roster)
                self.assertEqual(set(c.enemies), enemies)
                self.assertEqual(set(c.quests), quests)
                self.assertEqual(set(c.relics), relics)
                self.assertEqual(len(c.ex_disambiguation_index), 64)
                self.assertTrue(all(len(ids) == 1 for ids in c.ex_disambiguation_index.values()))
                self.assertEqual({q for ids in c.ex_disambiguation_index.values() for q in ids},
                                 {q.quest_id for q in c.quests.values() if q.quest_type == 2})
                clusters = defaultdict(set)
                for q in c.quests.values():
                    clusters[q.canonical_wave_sig].add(q.quest_type)
                    self.assertEqual(q.wave_total_hp, sum(c.enemies[e].hp for e in q.enemy_ids
                                                        if e not in c.environment_enemy_ids))
                self.assertTrue(all(len(types) == 1 for types in clusters.values()))
                for first, second in ((770330101, 770330102), (770530101, 770531601)):
                    self.assertEqual(c.quests[first].canonical_wave_sig, c.quests[second].canonical_wave_sig)
                self.assertEqual(c.environment_enemy_ids, [770900101, 770900201, 770900301, 770900401, 770900501])
                self.assertEqual(Counter(q.bucket for q in c.quests.values() if q.is_difficulty5_boss),
                                 {"boss_3f_d5": 5, "boss_5f_d5": 5})
                self.assertEqual(len(c.enhance_node_ids), 41)
                expected_gold = {50003}
                if c.db_version != "202611021739" or 50009 in relics:
                    expected_gold.add(50009)
                self.assertEqual({r.relic_id for r in c.relics.values() if r.gold_scaling}, expected_gold)
                self.assertTrue(c.relics[50001].owned_unit_scaling)
                self.assertEqual(set(c.max_account_training), {
                    "talent_skill_levels", "talent_skill_node_levels", "role_mastery",
                })
                self.assertEqual(len(c.max_account_training["talent_skill_node_levels"]), 638)
                self.assertTrue(any(level < 5 for level in c.max_account_training["talent_skill_node_levels"].values()))
                self.assertEqual(set(map(int, c.max_account_training["role_mastery"])), mastery_ids)
                self.assertTrue(all(len(u.static_feats) == 27 for u in c.units.values()))
                self.assertTrue(all(len(e.static_feats) == 11 for e in c.enemies.values()))


class SimClientTests(FixtureCase):
    def test_http_timeout_retries_only_when_enabled(self):
        config = SimulationConfig(self.catalog.quests[NORMAL], [1, 2])
        options = {"timeout_sec": 0.2, "max_retries": 2}
        for kind, expected_requests in (("default", 1), ("explicit", 3), ("profile", 3)):
            with self.subTest(kind=kind):
                release = threading.Event()
                with mock_server(self.catalog, withhold_response=release) as (url, requests):
                    if kind == "profile":
                        client = SimClient(url, self.catalog, self.profile | options | {"retry_on_timeout": True})
                    else:
                        kwargs = {"retry_on_timeout": True} if kind == "explicit" else {}
                        client = PcrCalculatorClient(url, catalog=self.catalog, **options, **kwargs)
                    try:
                        with self.assertRaises(SimulationTimeoutError) as error:
                            client.simulate_batch_single_runs([config])
                        self.assertIsInstance(error.exception, SimulationError)
                        self.assertEqual(len(requests), expected_requests)
                    finally:
                        release.set()

    def test_wrapped_timeout_is_not_retried_by_default(self):
        error = URLError(TimeoutError("timed out"))
        client = PcrCalculatorClient("http://127.0.0.1:8080", max_retries=2, catalog=self.catalog)
        config = SimulationConfig(self.catalog.quests[NORMAL], [1, 2])
        with patch("labyrinth_nn.data.sim_client.urlopen", side_effect=error) as post:
            with self.assertRaisesRegex(SimulationTimeoutError, "timed out") as raised:
                client.simulate_batch_single_runs([config])
        post.assert_called_once()
        self.assertIs(raised.exception.__cause__, error)

    def test_public_payload_builder_and_raw_batch_with_catalog(self):
        first = build_simulate_payload(
            catalog=self.catalog, profile=self.profile, quest_id=NORMAL, team_unit_ids=[1, 2],
            owned_unit_ids=[4, 1, 4], relic_ids=[50003], alpha_rupee=10000, enhance_node_ids=[],
            difficulty=5, calc_stats=2, need_once_result=False, use_fixed_random_seed=True, seed=101,
        )
        second = build_simulate_payload(
            self.catalog, self.profile, quest=self.catalog.quests[ROBOT_EX], team_unit_ids=[3],
            owned_unit_ids=None, relic_ids=None, use_fixed_random_seed=True, random_seed=102,
        )
        self.assertEqual(first["LabyrinthOwnedUnitIds"], [4, 1, 2])
        self.assertEqual(first["LabyrinthEnhancedNodeIds"], [])
        self.assertEqual(first["LabyrinthRelicIds"], [50003])
        self.assertEqual(first["LabyrinthAlphaRupee"], 10000)
        self.assertEqual(first["LabyrinthDifficulty"], 5)
        self.assertEqual(first["GuildEnemyList"], [201, 202])
        self.assertEqual(first["CalcStats"], 2)
        self.assertFalse(first["NeedOnceResult"])
        self.assertNotIn("MasterSkillNodeLevels", first)
        self.assertEqual(second["LabyrinthOwnedUnitIds"], [3])
        self.assertEqual(second["LabyrinthEnhancedNodeIds"], self.catalog.enhance_node_ids)
        first.update(RepeatCount=7, GuildEnemyList=[201, ENVIRONMENT, PROFILE_ENVIRONMENT, 202])
        with mock_server(self.catalog) as (url, requests):
            client = PcrCalculatorClient(url, timeout_sec=3.5, max_retries=0)
            self.assertEqual((client.timeout, client.max_retries), (3.5, 0))
            runs = client.simulate_batch_single_runs(
                [(MappingProxyType(first), 2), ({}, 0), (second, 1)],
                catalog=self.catalog, environment_enemy_ids=[PROFILE_ENVIRONMENT],
            )
        self.assertEqual([[run.seed for run in group] for group in runs], [[101, 101], [], [102]])
        self.assertEqual(len(requests), 1)
        self.assertEqual([query["RepeatCount"] for query in requests[0]], [1, 1, 1])
        self.assertEqual([query["GuildEnemyList"] for query in requests[0]], [[201, 202], [201, 202], [204]])
        self.assertEqual(first["RepeatCount"], 7)
        self.assertEqual(first["GuildEnemyList"], [201, ENVIRONMENT, PROFILE_ENVIRONMENT, 202])
        self.assertEqual([(run.enemy_final_hps, run.damage_ratio) for run in runs[0]], [([50, 150], 0.5)] * 2)
        self.assertEqual(runs[2][0].damage_ratio, 0.5)

    def test_raw_batch_without_catalog_uses_enemy_list_hp(self):
        payload = build_simulate_payload(self.catalog, self.profile, NORMAL, [1, 2])

        def respond(query, row, batch_number):
            if batch_number == 2:
                return row | {"TotalDamage": 40, "EnemyList": [
                    {"enemyId": enemy_id, "hp": self.catalog.enemies[enemy_id].hp}
                    for enemy_id in (201, 202, ENVIRONMENT)
                ]}
            return row | {"EnemyList": [
                {"enemyId": ENVIRONMENT, "maxHp": 99999999, "finalHp": 1},
                {"enemyId": 202, "HP": 300, "finalHp": 50},
                {"enemyId": 201, "max_hp": 100, "hp": 90},
            ]}

        with mock_server(self.catalog, respond=respond) as (url, _):
            client = PcrCalculatorClient(url)
            (run,), = client.simulate_batch_single_runs([(payload, 1)])
            (aggregate,), = client.simulate_batch_single_runs([(payload, 1)])
        self.assertEqual(run.enemy_final_hps, [90, 50])
        self.assertEqual(run.damage_ratio, 0.65)
        self.assertEqual(aggregate.enemy_final_hps, [])
        self.assertEqual(aggregate.damage_ratio, 0.1)

    def test_expand_and_regroup_http_requests(self):
        profile = self.profile | {"talent_skill_levels_override": {"1": "5"}, "role_mastery_override": {"10102": [4, 3]}}
        quest = self.catalog.quests[NORMAL]
        with mock_server(self.catalog) as (url, requests):
            client = SimClient(url, self.catalog, profile)
            runs = client.simulate_batch_single_runs(configs=[
                SimulationConfig(quest, [1, 2, 3], [1, 4, 4], runs=2, run_seeds=[10, 20]),
                SimulationConfig(quest, [2], runs=0),
                SimulationConfig(quest, [3, 4, 5], relic_ids=[50003], alpha_rupee=10001, runs=1, run_seeds=[30]),
            ])
        self.assertEqual([[run.seed for run in group] for group in runs], [[10, 20], [], [30]])
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(requests[0]), 3)
        first, second, last = requests[0]
        self.assertEqual(first, {
            "PlayerList": [{"unitId": 1}, {"unitId": 2}, {"unitId": 3}], "GuildEnemyList": [201, 202],
            "WaveGroupId": 1, "IsGuildBattle": True, "IsSkipping": True, "BattleTime": 90,
            "ToMaxUnitAttr": True, "IsAutoMode": True, "IsSetMode": False, "IsSemanMode": False,
            "CalcStats": 1, "NeedOnceResult": True, "RepeatCount": 1, "Settings": {
                "UseFixedRandomSeed": True, "RandomSeed": 10, "ForceNoCritical_player": False,
                "ForceNoCritical_enemy": False, "ForceIgnoreDodge_player": False, "ForceIgnoreDodge_enemy": False,
                "ForceCritical_player": False, "randomSpecialDatas": [],
            },
            "TalentSkillLevels": {"1": 5, "2": 1000}, "TalentSkillNodeLevels": {"11": 1, "12": 2},
            "RoleMastery": {"10102": [4, 3]}, "LabyrinthEnhancedNodeIds": [11, 12],
            "LabyrinthRelicIds": [], "LabyrinthOwnedUnitIds": [1, 4, 2, 3], "LabyrinthAlphaRupee": 0,
            "LabyrinthQuestId": NORMAL, "LabyrinthDifficulty": 5,
        })
        for query in (first, second, last):
            self.assertEqual(query["RepeatCount"], 1)
            self.assertNotIn("MasterSkillNodeLevels", query)
            self.assertNotIn("EnemyMasterSkillNodeLevels", query)
            self.assertTrue({p["unitId"] for p in query["PlayerList"]} <= set(query["LabyrinthOwnedUnitIds"]))
        self.assertEqual(last["LabyrinthAlphaRupee"], 10001)
        self.assertEqual(last["LabyrinthRelicIds"], [50003])
        run = runs[0][0]
        self.assertEqual(run.enemy_final_hps, [50, 150])
        self.assertEqual(run.damage_ratio, 0.5)
        self.assertEqual(run.total_damage, 1000)
        self.assertEqual(run.outcome_class, OUTCOME_TIMEOUT)
        self.assertAlmostEqual(run.elapsed_sec, 5399 / 60)
        self.assertEqual(runs[0][1].total_damage, 2000)

    def test_once_result_without_final_hps_and_server_failure(self):
        def respond(query, row, batch_number):
            if batch_number == 2:
                return {"Success": False, "Message": "bad fixture"}
            return row | {"Stats": None, "EnemyList": [{"enemyid": 201}, {"enemyid": 202}, {"enemyid": ENVIRONMENT}],
                          "OnceResult": {"currentDamage": 9000}}

        with mock_server(self.catalog, respond=respond) as (url, _):
            client = SimClient(url, self.catalog, self.profile)
            config = SimulationConfig(self.catalog.quests[NORMAL], [1, 2, 3])
            runs, = client.simulate_batch_single_runs([config])
            self.assertEqual(runs[0].enemy_final_hps, [])
            self.assertEqual(runs[0].total_damage, 9000)
            self.assertEqual(runs[0].damage_ratio, 1.0)
            with self.assertRaisesRegex(SimulationError, "bad fixture"):
                client.simulate_batch_single_runs([config])

    def test_probe_cli_over_loopback(self):
        catalog_path = self.catalog.save(self.root / "probe-catalog.json")
        profile_path = self.root / "probe-profile.json"
        profile_path.write_text(json.dumps(self.profile), encoding="utf-8")
        with mock_server(self.catalog) as (url, requests), redirect_stdout(io.StringIO()) as output:
            code = probe_simulator.main(["--catalog", str(catalog_path), "--profile", str(profile_path),
                                         "--url", url, "--bench-count", "64"])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 0, report)
        self.assertTrue(report["passed"])
        self.assertTrue(all(check["passed"] for check in report["checks"]))
        checks = {check["name"]: check["details"] for check in report["checks"]}
        self.assertLessEqual({
            "npc_3star_vs_6star", "labyrinth_skill_action_replace", "auto_vs_set", "env_enemy_770900501",
            "gold_relic_50003", "owned_relic_50001", "boss_5f_d5_damage_ratio", "order_swap_invariance",
            "difficulty5_quest_strata", "independent_single_runs", "enhancement_and_account_training",
            "relic_owned_gold_boundaries",
        }, set(checks))
        npc = checks["npc_3star_vs_6star"]
        self.assertEqual(npc["npc_3star"]["unit_id"], 2)
        self.assertEqual(npc["npc_6star"]["unit_id"], 1)
        npc_payloads = [query for batch in requests for query in batch if "rarity" in query["PlayerList"][0]]
        self.assertEqual([(q["PlayerList"][0]["rarity"], q["ToMaxUnitAttr"]) for q in npc_payloads],
                         [(3, False), (6, False)])
        unit_id = checks["labyrinth_skill_action_replace"]["unit_id"]
        self.assertTrue(self.catalog.units[unit_id].has_labyrinth_replace)
        auto, set_mode = next(batch for batch in requests if any(q["IsSetMode"] for q in batch))
        self.assertEqual((auto["IsAutoMode"], auto["IsSetMode"]), (True, False))
        self.assertEqual(set_mode, auto | {"IsAutoMode": False, "IsSetMode": True})
        environment = checks["env_enemy_770900501"]
        self.assertEqual(environment["excluded_enemy_id"], ENVIRONMENT)
        self.assertEqual(environment["non_environment_initial_hp"], 400)
        self.assertEqual(environment["enemy_final_hps"], [50, 150])
        self.assertEqual(environment["damage_ratio"], 0.5)
        self.assertTrue(all(ENVIRONMENT not in q["GuildEnemyList"] for batch in requests for q in batch))
        for name, field, values in (("gold_relic_50003", "alpha_rupee", [0, 10000]),
                                    ("owned_relic_50001", "owned_unit_count", [5, 25])):
            results = checks[name]["cases"]
            self.assertEqual([row[field] for row in results], values)
            self.assertEqual(results[0]["seed"], results[1]["seed"])
        boss = checks["boss_5f_d5_damage_ratio"]
        self.assertEqual(boss["quest_id"], 770531601)
        self.assertEqual(boss["non_environment_initial_hp"], 3_000_000_000)
        self.assertEqual(boss["enemy_final_hps"], [1_500_000_000])
        self.assertEqual(boss["damage_ratio"], 0.5)
        order = checks["order_swap_invariance"]
        self.assertEqual(order["forward"], order["reverse"][::-1])
        self.assertEqual(order["forward"], [[12401, 2, 5399, 1240100], [12402, 2, 5399, 1240200]])
        self.assertEqual([[q["Settings"]["RandomSeed"] for q in batch] for batch in requests[-3:-1]],
                         [[12401, 12402], [12402, 12401]])
        self.assertEqual(report["benchmark"]["completed_runs"], 64)
        self.assertGreater(report["benchmark"]["sims_per_sec"], 0)
        self.assertEqual(len(requests[-1]), 64)

    def test_probe_rejects_order_dependent_damage_even_when_seeds_match(self):
        second_seed_runs = 0

        def respond(query, row, batch_number):
            nonlocal second_seed_runs
            if row["CurrentSeed"] == 12402:
                second_seed_runs += 1
                if second_seed_runs == 2:
                    return row | {"Stats": {"TotalDamage": row["Stats"]["TotalDamage"] + 1}}
            return row

        with mock_server(self.catalog, respond=respond) as (url, _):
            client = PcrCalculatorClient(url, catalog=self.catalog, profile=self.profile)
            report = probe_simulator.run_probe(client, bench_count=0)
        failed = [check for check in report["checks"] if not check["passed"]]
        self.assertFalse(report["passed"])
        self.assertEqual([check["name"] for check in failed], ["order_swap_invariance"])
        self.assertIn("request order was swapped", failed[0]["error"])

    def test_probe_cli_failure_exit(self):
        catalog_path = self.catalog.save(self.root / "failed-probe-catalog.json")
        with mock_server(self.catalog, respond=lambda *args: {"Success": False, "Message": "bad battle"}) as (url, _):
            with redirect_stdout(io.StringIO()) as output:
                code = probe_simulator.main(["--catalog", str(catalog_path), "--profile", str(PROFILE),
                                             "--url", url, "--bench-count", "64"])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["benchmark"]["status"], "skipped")
        self.assertTrue(any("bad battle" in c.get("error", "") for c in report["checks"]))


class SamplerTests(FixtureCase):
    def test_sample_context_id_and_round_independent_physical_state(self):
        sample, _ = sampler.HighScoreSampler(self.catalog, self.profile, self.config).draw()
        legacy = sample.to_dict()
        del legacy["context_id"]
        self.assertEqual(BattleSample.from_dict(legacy), sample)
        paired = replace(sample, context_id="shared", round_id="round1", sample_id="new-id",
                         team_unit_ids=sample.team_unit_ids[::-1], owned_unit_ids=sample.owned_unit_ids[::-1],
                         relic_ids=sample.relic_ids[::-1])
        self.assertEqual(BattleSample.from_dict(paired.to_dict()), paired)
        identity = (sample.db_version, sample.quest_id, sample.difficulty, sample.guild_id,
                    sorted(sample.team_unit_ids), sorted(sample.owned_unit_ids), sorted(sample.relic_ids),
                    sample.alpha_rupee)
        expected = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertEqual(sample.physical_state_key(), expected)
        self.assertEqual(paired.physical_state_key(), expected)
        self.assertNotEqual(replace(paired, alpha_rupee=None).physical_state_key(), expected)

    def test_round1_config_and_adaptive_repeat_triggers(self):
        config = json.loads((ROOT / "configs/sampling_round1.json").read_text(encoding="utf-8"))
        profile = json.loads((ROOT / "configs/sim_profile_round1_cn.json").read_text(encoding="utf-8"))
        self.assertEqual(set(config["guild_weights"]), set(map(str, range(1, 11))))
        self.assertTrue(all(w > 0 for w in config["guild_weights"].values()))
        for name in ("area_state_distributions", "boundary_probe_probabilities"):
            self.assertEqual(config[name], self.config[name])
        repeat = profile["adaptive_repeat"]
        self.assertEqual((repeat["initial_runs"], repeat["borderline_runs"]), (2, 6))
        planner = sampler.HighScoreSampler(self.catalog, profile, config)
        base, _ = planner.draw()
        loss = SingleRunResult(1, 2, OUTCOME_TIMEOUT, 5400, 90, 90, 0.0, 100, 0.5)
        win = replace(loss, result=0, outcome_class=OUTCOME_WIN, damage_ratio=1.0, rem_time_ratio=0.5)
        # SHA256("a") starts above the 0.15 cutoff; SHA256("d") starts below it.
        base = replace(base, quest_id=NORMAL, sample_id="a")
        for run, expected in ((loss, False), (win, False),
                              (replace(loss, damage_ratio=0.75), True),
                              (replace(win, rem_time_ratio=0.15), True)):
            with self.subTest(run=run):
                self.assertEqual(planner._needs_repeat(replace(base, runs=[run, run])), expected)
        random_pick = replace(base, sample_id="d", runs=[loss, loss])
        rng_before = planner.rng.getstate()
        self.assertTrue(planner._needs_repeat(random_pick))
        self.assertTrue(planner._needs_repeat(random_pick))
        self.assertEqual(planner.rng.getstate(), rng_before)
        legacy = sampler.HighScoreSampler(self.catalog, self.profile, self.config)
        self.assertFalse(legacy._needs_repeat(random_pick))

    def test_paired_contexts_underfull_teams_and_context_safe_split(self):
        config = self.config | {"paired_teams_per_context": 4, "underfull_team_prob": 1.0,
                                "team_mix_weights": {"prior": 1}}
        planner = sampler.HighScoreSampler(self.catalog, self.profile, config)
        rows = [s for s, _ in planner._unique(40, set())]
        self.assertEqual(len({s.sample_id for s in rows}), len(rows))
        contexts = defaultdict(list)
        for sample in rows:
            self.assertTrue(sample.context_id)
            self.assertIn(len(sample.team_unit_ids), (3, 4))
            self.assertTrue(set(sample.team_unit_ids) <= set(sample.owned_unit_ids))
            contexts[sample.context_id].append(sample)
        self.assertEqual(len(contexts), 10)
        for group in contexts.values():
            self.assertEqual(len({s.canonical_team_sig() for s in group}), 4)
            self.assertEqual(len({replace(s, team_unit_ids=[]).physical_state_key() for s in group}), 1)
        split = split_samples(rows, val_ratio=0.2, test_ratio=0.2, group_by="context")
        self.assertEqual([len(part) for part in split], [24, 8, 8])
        sets = [{s.context_id for s in part} for part in split]
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
        fresh = sampler.HighScoreSampler(self.catalog, self.profile, config)
        self.assertEqual([s.sample_id for s in rows], [fresh.draw()[0].sample_id for _ in rows])

    def test_batch_isolation_quarantine_resume_and_manifest(self):
        config = self.config | {"isolate_batch_errors": True, "batch_size": 4, "chunk_size": 2}
        planner = sampler.HighScoreSampler(self.catalog, self.profile, config)
        bad_id = list(planner._unique(2, set()))[-1][0].sample_id
        path = self.root / "isolation.jsonl"
        failure_log = self.root / "isolation.failed.jsonl"
        sync_dir = self.root / "isolation-sync"
        sync_dir.mkdir()
        self.addCleanup(shutil.rmtree, sync_dir)

        def simulate(_client, batch):
            if any(s.sample_id == bad_id for s in batch):
                raise SimulationError("malformed battle result")
            return batch

        planner = sampler.HighScoreSampler(self.catalog, self.profile, config)
        with patch.object(planner, "simulate_samples", side_effect=simulate):
            result = planner.collect(None, path, 4, failure_log=failure_log, sync_dir=sync_dir)
        self.assertEqual((result["new_battles"], result["failed_battles"], result["new_shards"]), (3, 1, 2))
        self.assertEqual([json.loads(line) for line in failure_log.read_text(encoding="utf-8").splitlines()],
                         [{"sample_id": bad_id, "error": "malformed battle result"}])
        self.assertEqual((sync_dir / failure_log.name).read_bytes(), failure_log.read_bytes())
        resume = sampler.HighScoreSampler(self.catalog, self.profile, config)
        with patch.object(resume, "simulate_samples", side_effect=simulate):
            result = resume.collect(None, path, 4, resume=True, failure_log=failure_log)
        self.assertEqual((result["existing_battles"], result["new_battles"], result["failed_battles"]), (3, 1, 0))
        manifest = json.loads((self.root / "isolation.manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["total_unique_battles"], 4)
        self.assertEqual(manifest["round_id"], config["round_id"])
        self.assertEqual(manifest["db_version"], self.catalog.db_version)
        self.assertEqual(manifest["catalog_sha256"], self.catalog.catalog_sha256)
        self.assertEqual([e["row_count"] for e in manifest["shards"]], [2, 1, 1])
        ids = set()
        for entry in manifest["shards"]:
            shard = self.root / entry["filename"]
            self.assertEqual(entry["sha256"], hashlib.sha256(shard.read_bytes()).hexdigest())
            ids.update(sample.sample_id for sample in read_samples(shard))
        self.assertEqual(len(ids), 4)
        self.assertNotIn(bad_id, ids)

    def test_timeout_flushes_and_syncs_without_isolation_and_can_resume(self):
        config = self.config | {"isolate_batch_errors": True, "batch_size": 2, "chunk_size": 10, "concurrency": 1}
        path = self.root / "timeout.jsonl"
        manifest = self.root / "timeout.manifest.json"
        failure_log = self.root / "timeout.failed.jsonl"
        sync_dir = self.root / "timeout-sync"
        sync_dir.mkdir()
        self.addCleanup(shutil.rmtree, sync_dir)
        sync_order = []
        sync_file = sampler._sync_file_atomic

        def sync(source, destination):
            sync_order.append(source.name)
            if source == manifest:
                entries = json.loads(source.read_text(encoding="utf-8"))["shards"]
                for entry in entries:
                    shard = destination / entry["filename"]
                    self.assertEqual(hashlib.sha256(shard.read_bytes()).hexdigest(), entry["sha256"])
            return sync_file(source, destination)

        def simulate(_client, batch):
            if simulated.call_count == 2:
                raise SimulationTimeoutError("timed out")
            return batch

        planner = sampler.HighScoreSampler(self.catalog, self.profile, config)
        with (patch.object(planner, "simulate_samples", side_effect=simulate) as simulated,
              patch.object(sampler, "_sync_file_atomic", side_effect=sync)):
            with self.assertRaisesRegex(SimulationTimeoutError, "timed out"):
                planner.collect(None, path, 6, failure_log=failure_log, sync_dir=sync_dir)
        self.assertEqual([len(call.args[1]) for call in simulated.call_args_list], [2, 2])
        self.assertEqual(sync_order, [path.name, manifest.name])
        self.assertFalse(failure_log.exists())
        self.assertFalse((sync_dir / failure_log.name).exists())
        self.assertEqual(len(read_samples(path)), 2)
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["total_unique_battles"], 2)
        self.assertEqual((sync_dir / manifest.name).read_bytes(), manifest.read_bytes())

        resume = sampler.HighScoreSampler(self.catalog, self.profile, config)
        with (patch.object(resume, "simulate_samples", side_effect=lambda _client, batch: batch),
              patch.object(sampler, "_sync_file_atomic", side_effect=sync)):
            result = resume.collect(None, path, 6, resume=True, failure_log=failure_log, sync_dir=sync_dir)
        self.assertEqual((result["existing_battles"], result["total_battles"], result["failed_battles"]), (2, 6, 0))
        records = read_samples(path) + read_samples(self.root / "timeout.part-00001.jsonl")
        expected = sampler.HighScoreSampler(self.catalog, self.profile, config)
        self.assertEqual([row.sample_id for row in records], [expected.draw()[0].sample_id for _ in range(6)])
        self.assertEqual((sync_dir / manifest.name).read_bytes(), manifest.read_bytes())

    def test_keyboard_interrupt_during_sync_retries_shard_before_manifest(self):
        config = self.config | {"batch_size": 2, "chunk_size": 2, "concurrency": 1}
        planner = sampler.HighScoreSampler(self.catalog, self.profile, config)
        path = self.root / "keyboard-interrupt.jsonl"
        manifest = self.root / "keyboard-interrupt.manifest.json"
        sync_dir = self.root / "keyboard-interrupt-sync"
        sync_dir.mkdir()
        self.addCleanup(shutil.rmtree, sync_dir)
        sync_file = sampler._sync_file_atomic

        def sync(source, destination):
            if sync_attempt.call_count == 1:
                raise KeyboardInterrupt
            if source == manifest:
                self.assertEqual((destination / path.name).read_bytes(), path.read_bytes())
            return sync_file(source, destination)

        with (patch.object(planner, "simulate_samples", side_effect=lambda _client, batch: batch),
              patch.object(sampler, "_sync_file_atomic", side_effect=sync) as sync_attempt):
            with self.assertRaises(KeyboardInterrupt):
                planner.collect(None, path, 2, sync_dir=sync_dir)
        self.assertEqual([call.args[0] for call in sync_attempt.call_args_list], [path, path, manifest])
        self.assertEqual((sync_dir / manifest.name).read_bytes(), manifest.read_bytes())

    def test_sync_file_skips_matches_and_replaces_atomically(self):
        source = self.root / "atomic-source.jsonl"
        source.write_bytes(b"old data")
        parent = self.root / "atomic-sync"
        parent.mkdir()
        self.addCleanup(shutil.rmtree, parent)
        sync_dir = parent / "destination"
        self.assertFalse(sampler._sync_file_atomic(self.root / "missing.jsonl", sync_dir))
        self.assertFalse(sync_dir.exists())
        self.assertTrue(sampler._sync_file_atomic(source, sync_dir))
        destination = sync_dir / source.name
        with patch.object(shutil, "copy2") as copy:
            self.assertFalse(sampler._sync_file_atomic(source, source.parent / "."))
            self.assertFalse(sampler._sync_file_atomic(source, sync_dir))
            copy.assert_not_called()

        source.write_bytes(b"new data")
        replace_file = os.replace

        def publish(temporary, dst):
            self.assertEqual(temporary.parent, sync_dir)
            self.assertEqual(dst, destination)
            self.assertEqual(dst.read_bytes(), b"old data")
            self.assertEqual(temporary.read_bytes(), b"new data")
            replace_file(temporary, dst)

        with patch.object(os, "replace", side_effect=publish) as publish_file:
            self.assertTrue(sampler._sync_file_atomic(source, sync_dir))
        publish_file.assert_called_once()
        self.assertEqual(destination.read_bytes(), b"new data")
        source.write_bytes(b"unpublished data")
        with patch.object(os, "replace", side_effect=OSError(errno.EIO, "I/O error")):
            with self.assertRaisesRegex(OSError, "I/O error"):
                sampler._sync_file_atomic(source, sync_dir)
        self.assertEqual(destination.read_bytes(), b"new data")
        self.assertEqual(list(sync_dir.iterdir()), [destination])

    def test_cli_time_limit_telemetry_overrides_and_resume(self):
        catalog_path = self.catalog.save(self.root / "timed-catalog.json")
        profile_path = self.root / "timed-profile.json"
        profile = self.profile | {"adaptive_repeat": {"initial_runs": 1, "borderline_runs": 3}}
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        path = self.root / "timed.jsonl"
        sync_dir = self.root / "timed-sync"
        sync_dir.mkdir()
        self.addCleanup(shutil.rmtree, sync_dir)
        args = ["--catalog", str(catalog_path), "--profile", str(profile_path), "--sampling", str(SAMPLING),
                "--out", str(path), "--target-battles", "7", "--chunk-size", "2", "--batch-size", "2",
                "--concurrency", "2", "--timeout-sec", "3.25", "--max-seconds", "1", "--progress"]
        run = SingleRunResult(1, 0, OUTCOME_WIN, 1800, 90, 30, 2 / 3, 100, 1.0)

        def simulate(_planner, client, batch):
            self.assertEqual(client.timeout, 3.25)
            return [replace(sample, runs=[run] if i else [replace(run, seed=s) for s in (1, 2, 3)])
                    for i, sample in enumerate(batch)]

        def invoke(extra_args, ticks):
            with (patch.object(sampler, "time") as clock,
                  redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as progress):
                clock.monotonic.side_effect = ticks
                self.assertEqual(sampler.main(args + extra_args), 0)
            return json.loads(output.getvalue()), progress.getvalue().splitlines()

        with (patch.object(sampler.HighScoreSampler, "simulate_samples", autospec=True, side_effect=simulate) as calls,
              patch.object(sampler.HighScoreSampler, "draw", autospec=True,
                           side_effect=sampler.HighScoreSampler.draw) as draw):
            first, progress = invoke(["--sync-dir", str(sync_dir)], [100, 100, 102, 102])
            self.assertEqual(draw.call_count, 4)
            self.assertEqual([len(call.args[2]) for call in calls.call_args_list], [2, 2])
            calls.reset_mock()
            resumed, _ = invoke(["--sync-dir", str(sync_dir), "--resume"], [200, 200, 202])
            self.assertCountEqual([len(call.args[2]) for call in calls.call_args_list], [2, 1])
            calls.reset_mock()
            new_sync_dir = sync_dir / "complete"
            complete, _ = invoke(["--sync-dir", str(new_sync_dir), "--resume"], [300, 302])
            calls.assert_not_called()

        self.assertTrue(first["stopped_by_time_limit"])
        self.assertFalse(resumed["stopped_by_time_limit"])
        self.assertFalse(complete["stopped_by_time_limit"])
        self.assertEqual(len(progress), 2)
        self.assertIn("Collected 2/7 battles", progress[0])
        self.assertIn("Collected 4/7 battles", progress[1])
        self.assertEqual(first["existing_battles"], 0)
        self.assertEqual(first["new_battles"], 4)
        for field, expected in {"existing_subruns": 0, "new_subruns": 8, "total_subruns": 8,
                                "repeated_battles": 2, "repeat_rate": 0.5, "elapsed_sec": 2.0,
                                "rows_per_sec": 2.0, "subruns_per_sec": 4.0}.items():
            with self.subTest(field=field):
                self.assertEqual(first[field], expected)
        self.assertEqual((resumed["existing_battles"], resumed["new_battles"], resumed["total_battles"]), (4, 3, 7))
        self.assertEqual((resumed["existing_subruns"], resumed["new_subruns"], resumed["total_subruns"]), (8, 7, 15))
        self.assertEqual((complete["new_battles"], complete["existing_subruns"], complete["new_subruns"]), (0, 15, 0))

        manifest = self.root / "timed.manifest.json"
        entries = json.loads(manifest.read_text(encoding="utf-8"))["shards"]
        records = []
        for name in [entry["filename"] for entry in entries] + [manifest.name]:
            source = self.root / name
            self.assertEqual((sync_dir / name).read_bytes(), source.read_bytes())
            self.assertEqual((new_sync_dir / name).read_bytes(), source.read_bytes())
            if source != manifest:
                records.extend(read_samples(source))
        fresh = sampler.HighScoreSampler(self.catalog, profile, self.config)
        self.assertEqual([row.sample_id for row in records], [fresh.draw()[0].sample_id for _ in range(7)])
        self.assertEqual(first["buckets_new"], dict(Counter(row.bucket for row in records[:4])))
        self.assertEqual(first["guilds_new"], dict(Counter(str(row.guild_id) for row in records[:4])))

    def test_collect_and_cli_validate_positive_limits(self):
        planner = sampler.HighScoreSampler(self.catalog, self.profile, self.config)
        with self.assertRaisesRegex(ValueError, "max_seconds must be positive"):
            planner.collect(None, self.root / "invalid.jsonl", 1, max_seconds=0)
        for flag, name in (("--batch-size", "batch_size"), ("--concurrency", "concurrency"),
                           ("--timeout-sec", "timeout_sec"), ("--max-seconds", "max_seconds")):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    sampler.main(["--catalog", "unused.json", "--dry-run", "1", flag, "0"])
                self.assertEqual(raised.exception.code, 1)
                self.assertIn(f"{name} must be positive", error.getvalue())

    def test_shard_publish_fallback_checks_errno_and_existing_path(self):
        path = self.root / "fallback.jsonl"
        sample, _ = sampler.HighScoreSampler(self.catalog, self.profile, self.config).draw()
        with patch.object(os, "link", side_effect=OSError(errno.EINVAL, "unsupported hardlink")):
            sampler._write_shard(path, [sample])
            with self.assertRaises(FileExistsError):
                sampler._write_shard(path, [replace(sample, sample_id="must-not-overwrite")])
        self.assertEqual(read_samples(path), [sample])
        other = self.root / "fallback-error.jsonl"
        with patch.object(os, "link", side_effect=OSError(errno.EIO, "I/O error")):
            with self.assertRaisesRegex(OSError, "I/O error"):
                sampler._write_shard(other, [sample])
        self.assertFalse(other.exists())
        self.assertFalse(list(self.root.glob(".fallback-*.jsonl")))

    def test_select_hard_negatives_ranks_prediction_errors_and_close_battles(self):
        base, _ = sampler.HighScoreSampler(self.catalog, self.profile, self.config).draw()
        samples = [
            replace(base, sample_id=name, outcome_counts=counts, mean_damage_ratio=damage)
            for name, counts, damage in (
                ("unlabeled", [0, 0, 0], 0.99), ("clear_win", [4, 0, 0], 1.0),
                ("clear_loss", [0, 4, 0], 0.0), ("loss", [0, 4, 0], 0.2),
                ("near_win", [0, 4, 0], 0.95), ("mixed", [1, 3, 0], 0.8), ("balanced", [2, 2, 0], 0.7),
            )
        ]
        heuristic = sampler.select_hard_negatives(iter(samples))
        self.assertEqual([s.sample_id for s in heuristic], ["balanced", "mixed", "near_win", "loss"])
        predictions = {"clear_win": 0.25, "clear_loss": 0.875, "balanced": 0.5, "loss": 0.125}
        ranked = sampler.select_hard_negatives(samples, predictions, limit=3)
        self.assertEqual([s.sample_id for s in ranked], ["clear_loss", "clear_win", "loss"])
        tied = sampler.select_hard_negatives(samples, {"clear_win": 0.75, "mixed": 0.5}, limit=2)
        self.assertEqual([s.sample_id for s in tied], ["mixed", "clear_win"])
        self.assertEqual(sampler.select_hard_negatives(samples, limit=0), [])

    def test_dry_run_boundaries_and_strata(self):
        catalog_path = self.catalog.save(self.root / "dry-catalog.json")
        count = 4000
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(sampler.main(["--catalog", str(catalog_path), "--profile", str(PROFILE),
                                          "--sampling", str(SAMPLING), "--dry-run", str(count)]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["planned_battles"], count)
        self.assertEqual(result["unique_quests"], 9)
        for weights, stats in (("quest_type_weights", "strata"), ("team_mix_weights", "team_modes")):
            self.assertEqual(set(self.config[weights]), set(result[stats]))
            for key, expected in self.config[weights].items():
                self.assertAlmostEqual(result[stats][key] / count, expected, delta=0.03)
        boundary = result["boundaries"]
        for name, values, lower_bound in (
            ("mark_thresholds", (4, 9, 15), 0.045),
            ("mark_neighborhoods", (4, 9, 15), 0.08),
            ("owned_25", (24, 25, 26), 0.045),
            ("gold_10k", (9999, 10000, 10001), 0.05),
        ):
            for value in values:
                self.assertGreater(boundary[name][str(value)] / count, lower_bound)
        self.assertAlmostEqual(boundary["empty_relic_early"] / count, 0.05, delta=0.015)

    def test_illegal_relics_never_mix_and_future_boss_uses_top_weight(self):
        future = replace(self.catalog.quests[770531601], area=10, bucket="boss_10f_d5")
        c = replace(self.catalog, quests=self.catalog.quests | {future.quest_id: future})
        planner = sampler.HighScoreSampler(c, self.profile, self.config)
        self.assertIn(future, planner.quests["boss_5f_d5"])
        for _ in range(800):
            sample, _ = planner.draw()
            guild = c.guilds[sample.guild_id]
            self.assertEqual(sample.difficulty, 5)
            self.assertNotEqual(sample.bucket, "boss_other")
            self.assertEqual(len(set(sample.team_unit_ids)), 5)
            self.assertTrue(set(sample.team_unit_ids) <= set(sample.owned_unit_ids))
            self.assertIn(guild.guild_master_id, sample.owned_unit_ids)
            self.assertEqual(len(sample.relic_ids), len(set(sample.relic_ids)))
            self.assertEqual(sample.mark_stacks, c.mark_stacks_for_relics(sample.relic_ids))
            if not sample.relic_ids:
                self.assertEqual(sample.area, 1)
                continue
            self.assertIn(guild.add_relic_id, sample.relic_ids)
            for rid in sample.relic_ids:
                if 40000 < rid < 41000 or 50000 < rid < 51000:
                    self.assertIn(rid, {40000 + sample.guild_id, guild.add_relic_id})

    def test_adaptive_repeat_keyword_with_one_initial_run(self):
        def respond(query, row, batch_number):
            return row | {"Result": 0, "BattleFrames": 1800, "EnemyList": []} if batch_number == 1 else row

        planner = sampler.HighScoreSampler(self.catalog, self.profile, self.config)
        base, _ = planner.draw()
        pair = [replace(base, quest_id=qid, sample_id=str(qid)) for qid in (NORMAL, ROBOT_EX)]
        with mock_server(self.catalog, respond=respond) as (url, batches):
            samples = planner.simulate_samples(SimClient(url, self.catalog, self.profile), pair)
        self.assertEqual([len(batch) for batch in batches], [2, 5])
        self.assertEqual([q["LabyrinthQuestId"] for q in batches[1]], [ROBOT_EX] * 5)
        self.assertTrue(all(q["RepeatCount"] == 1 for batch in batches for q in batch))
        self.assertEqual([len(s.runs) for s in samples], [1, 6])
        self.assertEqual(len({r.seed for r in samples[1].runs}), 6)
        self.assertEqual(samples[0].runs[0].outcome_class, OUTCOME_WIN)
        self.assertEqual(samples[1].outcome_counts, [1, 5, 0])
        self.assertAlmostEqual(samples[1].mean_damage_ratio, (1 + 5 * 0.5) / 6)

    def test_adaptive_repeat_win_rate_with_explicit_distinct_seeds(self):
        def respond(query, row, batch_number):
            nonlocal initial_count
            if batch_number == 1:
                initial_count += 1
            if batch_number == initial_count == 1:
                return row | {"Result": 0, "BattleFrames": 1800, "EnemyList": []}
            return row

        initial_count = 0
        profile = self.profile | {"use_fixed_random_seed": True,
                                  "adaptive_repeat": {"initial_runs": 2, "borderline_runs": 6, "high_rng_keywords": []}}
        planner = sampler.HighScoreSampler(self.catalog, profile, self.config)
        base, _ = planner.draw()
        sample = replace(base, quest_id=NORMAL)
        with mock_server(self.catalog, respond=respond) as (url, batches):
            planner.simulate_samples(SimClient(url, self.catalog, profile), [sample])
        self.assertEqual([len(batch) for batch in batches], [2, 4])
        self.assertEqual(len({r.seed for r in sample.runs}), 6)
        self.assertEqual(sample.outcome_counts, [1, 5, 0])

    def test_incremental_shards_resume_to_total_without_duplicates(self):
        config = self.config | {"batch_size": 2, "chunk_size": 3, "concurrency": 2}
        path = self.root / "battles.jsonl"
        with mock_server(self.catalog) as (url, requests):
            client = SimClient(url, self.catalog, self.profile)
            first = sampler.HighScoreSampler(self.catalog, self.profile, config).collect(client, path, 4)
            resumed = sampler.HighScoreSampler(self.catalog, self.profile, config).collect(client, path, 8, resume=True)
            before = len(requests)
            complete = sampler.HighScoreSampler(self.catalog, self.profile, config).collect(client, path, 8, resume=True)
        self.assertEqual((first["new_battles"], resumed["existing_battles"], resumed["new_battles"]), (4, 4, 4))
        self.assertEqual(complete["new_battles"], 0)
        self.assertEqual(len(requests), before)
        shards = [path, *sorted(self.root.glob("battles.part-*.jsonl"))]
        records = [s for shard in shards for s in read_samples(shard)]
        self.assertEqual([len(read_samples(shard)) for shard in shards], [3, 1, 3, 1])
        self.assertEqual(len({s.sample_id for s in records}), 8)
        fresh = sampler.HighScoreSampler(self.catalog, self.profile, config)
        self.assertEqual([s.sample_id for s in records], [fresh.draw()[0].sample_id for _ in range(8)])

    def test_resume_keeps_completed_batch_when_later_request_fails(self):
        def respond(query, row, batch_number):
            return {"Success": False, "Message": "temporary failure"} if batch_number == 2 else row

        config = self.config | {"batch_size": 2, "chunk_size": 10, "concurrency": 1}
        profile = self.profile | {"adaptive_repeat": {"initial_runs": 1, "borderline_runs": 1}}
        path = self.root / "interrupted.jsonl"
        with mock_server(self.catalog, respond=respond) as (url, _):
            client = SimClient(url, self.catalog, profile)
            planner = sampler.HighScoreSampler(self.catalog, profile, config)
            with self.assertRaisesRegex(SimulationError, "temporary failure"):
                planner.collect(client, path, 6)
            self.assertEqual(len(read_samples(path)), 2)
            resumed = sampler.HighScoreSampler(self.catalog, profile, config).collect(client, path, 6, resume=True)
        self.assertEqual((resumed["existing_battles"], resumed["new_battles"]), (2, 4))
        records = read_samples(path) + read_samples(self.root / "interrupted.part-00001.jsonl")
        fresh = sampler.HighScoreSampler(self.catalog, profile, config)
        self.assertEqual([s.sample_id for s in records], [fresh.draw()[0].sample_id for _ in range(6)])


if __name__ == "__main__":
    unittest.main()
