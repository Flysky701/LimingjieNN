# TASK · LimingjieNN 云端训练与自动监督任务书

本任务书面向 **Google Colab / 云端 GPU** 运行环境及自动监督 Agent，记录云端目录约定、Round 0/1 完成基线、阶段门禁与 **20 CC 储备**。

> 执行脚本与 Cell 入口以 [notebooks/colab_labyrinth_pipeline.ipynb](notebooks/colab_labyrinth_pipeline.ipynb)、[scripts/colab_setup_simulator.sh](scripts/colab_setup_simulator.sh) 及 `configs/*.json` 为准；执行规则见 [自动监督规则](docs/AGENT_SUPERVISION_RULES.md)。
>
> **跨仓定位**：双仓统一里程碑（分支收口、视觉方案冗余审查、模型内置分发及 Path B 客户端落地）以 `LimingjieDesktop/docs/ROADMAP.md` 为唯一权威源。

---

## 1. 云端目录约定

| 项 | 约定路径 / 地址 | 说明 |
| --- | --- | --- |
| **Git 仓库** | `https://github.com/Flysky701/LimingjieNN`（`main`） | 存放源码、配置、脚本、Notebook 与通过 S5 验收的发布包 |
| **Colab 工作目录** | `/content/LimingjieNN`（或 `-zip` / `-drive`） | 对应锁定提交的本地运行副本 |
| **Drive 根目录** | `/content/drive/MyDrive/LimingjieNN/`（`$DRIVE_ROOT`） | 持久化存储 DB、采样分片、Checkpoint 与导出包 |
| ├ `master_cn.db` | `$DRIVE_ROOT/master_cn.db` | 离线锁定国服主库（`202611131613`）；**严禁入库 Git** |
| ├ `cache/` | `$DRIVE_ROOT/cache/` | 缓存模拟器 ZIP、`wheelhouse/` 与 `persistent_offline_202611131613.tar`（配 `.sha256`） |
| ├ `data/` & `runs/` | `$DRIVE_ROOT/{data,runs}/` | `feature_pack.json`、采样 `.jsonl` 分片及各轮训练 Checkpoint |
| ├ `export_bundle/` | `$DRIVE_ROOT/export_bundle/<mode>/` | `tower_a.onnx`、`tower_b.onnx`、`manifest.json`、`golden_io.json` |
| ├ `status/latest_receipt.json` | `$STATUS_DIR/latest_receipt.json` | 唯一机器可读回执与门禁报告 |
| └ `replays.jsonl` | `$DRIVE_ROOT/replays.jsonl` | （待补）真实开局地图录制，用于 Phase 5 Path B 回放对照 |

---

## 2. 已完成基线（Round 0 & Round 1）

- [x] **Round 0 Pilot（Phase 0–4）**：25,000 唯一采样（68 分片 / 44,155 子局，13.02 sims/s），16/16 模拟器对账与四档消融验证通过。
- [x] **Round 1（S0–S5 & ONNX 发布）**：
  - **S0–S3**：锁定 `202611131613` 与特征目录；完成分层放量与主动挖掘，S4 冻结切分共 **301,000 条**（训练 256,065 / 选择 11,293 / 校准 11,223 / 测试 22,419）。
  - **S4**：`m2b_r1` 胜出；三种子集成对胜率 Brier 仅改善 `0.00241 < 0.003`、Top-1 失败率降 `0.796` 个百分点 `< 1`，未触发蒸馏，发布单模型 `seed_43`。
  - **S5 封存评估**：32,000 条分层战绩全局 AUROC **0.9914**、胜率 ECE **0.0085**；1,000 配对上下文（Top-1 21 种子复测）中 779 个可行上下文在 `p_win >= 0.50` 覆盖 744 个、失败 **14/744（1.88%）**；全量面板覆盖 76.4%、失败 4.32%（推荐风险原门禁未全过，桌面采用 `>=0.70` / `0.50–0.70` / `<0.50` 分档回退）。
  - **ONNX 验收**：双塔合计 **8,299,348 字节（7.915 MiB）**，`max_abs_diff = 7.629e-6`，桌面 CPU 512 候选 **p95 53.54 ms**，Java 黄金对齐 **2/2** 通过。交付包见 [models/s4_m2b_r1_seed43/](models/s4_m2b_r1_seed43/README.md)。
- [ ] **Phase 5 · Path B 离线真实地图回放基准**：待客户端提供脱敏 `replays.jsonl` 后在 `env/run_env.py` 执行对照。

---

## 3. 复现、回放与停机红线

- **云端复现与续跑**：按 [notebooks/colab_labyrinth_pipeline.ipynb](notebooks/colab_labyrinth_pipeline.ipynb) Cell 顺序执行；环境依赖与参数口径见 `scripts/` 与 `configs/*.json`。
- **Phase 5 回放入口**：
  ```bash
  PYTHONPATH=src python -c "from pathlib import Path; from labyrinth_nn.catalog.schema import FeatureCatalog; from labyrinth_nn.env.run_env import PathBReplayPolicy, compare_policies, load_replays; c=FeatureCatalog.load(Path('models/s4_m2b_r1_seed43/feature_pack.json')); print(compare_policies(load_replays(Path('replays.jsonl')), path_b=PathBReplayPolicy(target_score=53000), settings=c.score_settings, catalog=c, repeats=5))"
  ```
- **立即停止并告警条件**：① `probe_simulator` 失败或服务反复退出；② 训练出现非有限 loss/NLL 或 `--auto-resume` 无法推进；③ `m2b` 核心切片明显劣于 `m0`；④ ONNX `max_abs_diff >= 1e-4` 或双塔体积 `>= 10,000,000` 字节。
- **Git 仓库卫生**：仅 `models/<release_id>/` 允许入库验收后的轻量化发布包；`*.db`、`*.jsonl`、`*.parquet`、`*.pt`、`*.pth` 及未验收临时 `*.onnx` 严禁入库。

---

## 4. 封存发布门禁与 20 CC 储备

后续启用 20 CC 储备重训或扩层时，沿用以下 Round 1 封存门禁与预算安排：

| 封存面板的发布门禁 | Go 阈值 |
| --- | --- |
| 区分能力 | 全局 AUROC **≥0.950**。普通、机器人 EX、特殊双 EX、其他 EX、3F Boss、5F Boss 依次 **≥0.800 / 0.820 / 0.935 / 0.915 / 0.910 / 0.945**。普通和机器人各自比同面板 Quest-Only **≥0.02**。各桶至少 **200 正例和 200 负例**。 |
| 概率质量 | 重加权三类 Brier **≤0.210**，胜率 Brier **≤0.074**，全局胜率 ECE **≤0.030**，各桶胜率 ECE **≤0.055**，三类准确率 **≥0.845**。报告独立校准前后的 NLL。 |
| 金币未知 | 至少 **2,000 个成对上下文**及 **500 个金币缩放遗物样本**。遮蔽金币后胜率 ECE **≤0.045**，相较同样本金币可见视图的胜率 Brier 增量 **≤0.010**，单列金币缩放遗物切片。 |
| 连续量输出 | 非胜样本伤害 MAE **≤0.185**、胜利余时 MAE **≤0.125**。独立终局 HP 标签单独报告。只有聚合伤害代理时，验收仅涵盖该代理的误差。 |
| 推荐风险 | 至少 **1,000 个新上下文**。预先定义的可行上下文面板上，Top-1 失败率 **≤5%**；预测为胜的 Top-1 失败率 **≤3%**、单侧 95% 置信上界 **≤5%**、覆盖率 **≥80%**。同时报告分子分母、无可行队难例与未过滤流量。 |
| 桌面导出 | 两个 ONNX 合计 **<10,000,000 bytes**。输出有限且双塔串联 `max_abs_diff < 1e-4`。验收真实特征输入、金币已知/未知、站位并列、部分队伍和缓存上下文多队评分。 |
| 桌面延迟 | 目标桌面至少测 **200 个上下文**。热态 CPU p95：Tower A **≤25 ms**，Tower B 64 队一批 **≤30 ms**，512 候选完整推荐 **≤500 ms**（含特征计算和配队）。记录硬件、线程与冷启动。 |

### 20 CC 储备分配

| CC | 触发与用途 |
| ---: | --- |
| **8** | 国服 DB、模拟器、库存/OCR 或楼层变化（如 10 层扩展）引入分布偏移。先复跑成对样例，再定向采购约 **300k–500k 子局**，保留新版本封存面板。 |
| **5** | 分桶 AUROC 低于门槛 **≥0.03**，普通/机器人同上下文失败率恶化 **≥2 个百分点**，或独立 HP 证据确认系统性标签错误。定向扩增或重标。 |
| **4** | 三种子教师达到 S4 蒸馏门槛，单模型效果或延迟仍未达标。训练学生或更窄模型，独立校准。 |
| **3** | 候选已通过开发门禁，进入正式发布确认。使用全新确认面板、目标桌面计时、导出对齐和可用的真实地图回放。 |
