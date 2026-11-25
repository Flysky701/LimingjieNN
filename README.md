# LimingjieNN · 黎明界高分神经网络与全局结算分规划系统

`LimingjieNN` 是面向《公主连结 Re:Dive》（国服）黎明界高分上限场景的离线决策训练、Path B 数学规划原型与模型发布仓库，作为 `LimingjieDesktop` 上层策略模块的后端源。跨仓总里程碑、视觉冗余审查、分支收口与客户端落地节奏统一维护在 `LimingjieDesktop/docs/ROADMAP.md`。

## 已发布模型（`s4_m2b_r1_seed43`）

Round 1 采用 `m2b_r1 / seed_43` 单模型（国服锁库 `202611131613`，5 层 D5），S5 32k 全局 AUROC **0.9914**、胜率 ECE **0.0085**，双塔 ONNX 合计 **7.915 MiB**，桌面 CPU 512 候选推荐 **p95 53.54 ms**，详细指标、张量契约与阈值见 [models/s4_m2b_r1_seed43/README.md](models/s4_m2b_r1_seed43/README.md)。

桌面端可通过 `-Dlimingjie.nn.bundle` 或 `LIMINGJIE_NN_BUNDLE` 指向 `models/s4_m2b_r1_seed43`（`BattleOnnxBundle` 自动发现 `feature_pack.json`），未启用或校验失败时自动回退规则评分。

## 核心架构（Level 1 + Path B）

1. **Level 1 · Two-Tower 晚交互双塔战斗网络 (`V_battle`)**
   - **Tower A (`ContextRosterEncoder`)**：每次进编组运行 1 次，联合编码敌方波次、遗物集合（内置永不掩码 `[NULL_RELIC]` Sink Token）、5 系印记、41 强化节点、金币及缺失态与完整持有池，输出上下文表征与单兵契合分。
   - **Tower B (`SlotAwareTeamHead`)**：5 槽轻量交互头，严格按 `(search_area_width, unit_id)` 升序排位，注入带符号成对站位差与全队协同特征，输出 `(p_win, p_timeout, p_wipe)` 校准概率、`damage_ratio` 分布及胜场 `win_rem_time`。
   - **消融与扩展性**：`m0 → m1 → m2a → m2b` 四档可归因演进；词表边界（`max_areas=16`, `max_difficulties=10`）动态读取 `master_cn.db`，可直接平滑升级至 10 层多队版本。
2. **Path B · 全局官方结算分精确规划器（Python 参考实现）**
   - **`route_dp.py`**：以分数最大公约数构建离散概率 DAG 反向 DP，支持 `t=0` 开局筛图、`T=0` 稳健清票与 `T>0` 冲榜早停门禁。
   - **`shop_dp.py` & `marginal_eval.py`**：支持折扣叠加（`#21` + `40003`）、递增刷新费与持币遗物耦合的期望 DP，以及选角/遗物/事件按官方结算分同尺度边际估值。
   - **`env/run_env.py`**：离线真实地图回放与策略对照沙盒。

## 目录结构

```text
LimingjieNN/
├── configs/                     # 采样、模拟与 m0..m2b 训练配置
├── docs/                        # AGENT_SUPERVISION_RULES.md
├── models/s4_m2b_r1_seed43/     # Round 1 单模型 ONNX 发布包
├── notebooks/                   # Colab 端到端流水线
├── scripts/                     # 离线锁库模拟器与依赖安装脚本
├── src/labyrinth_nn/            # catalog / data / models / train / export / planning / env / active_mining
└── tests/                       # stdlib 单测 + Torch/ORT 对齐测试
```

## 快速校验

```powershell
# 本地 Python 3.11+（仅标准库即可运行单测、目录提取与采样干跑）
$env:PYTHONPATH = (Resolve-Path -LiteralPath ".\src").Path
python -m unittest discover -s tests -v
python -m labyrinth_nn.catalog.build_catalog --db ".\master_cn.db" --out "artifacts/feature_pack.json"
python -m labyrinth_nn.data.sampler --catalog "artifacts/feature_pack.json" --dry-run 2000
```

云端采样、训练与导出请直接按 [notebooks/colab_labyrinth_pipeline.ipynb](notebooks/colab_labyrinth_pipeline.ipynb) 与 [TASK.md](TASK.md) 执行。

## 后续接入 `LimingjieDesktop` 交接清单（同步见 `LimingjieDesktop/docs/ROADMAP.md`）

1. **桌面加载、战斗搜索与 Path B 接线已合流至单一 `main`**：`BattleOnnxBundle`、`ModelGuidedBattleTeamSearcher`（Tower A 缓存 + 三档回退）、EX 映射、Boss 多队规划、`kernel/planning/` 规划器及状态 UI 已整合进 `LimingjieDesktop` `main`（`1.0.18+desktop.17`），并在统一主干通过 ORT Java `1.21.0` `BattleOnnxGoldenTest`（对齐 PR #29 三档前排模式）；后续桌面便携发行包倾向直接内置本发布包。
2. **金币未知路径可先联调，Path B 依赖 M1 视觉与账本补齐**：S5 遮蔽金币后 Win Brier 仅增 `+0.00003`，但完整 `ShopDp` / `MarginalEval` / `RouteDp` 运行时接入需在 `LimingjieDesktop` M1 视觉冗余专项中补齐地图/事件/商店金币 OCR 及局内收支账本。
3. **保留模块隔离与 SHA 锁**：纯 Kotlin 策略 SPI 位于共享层，Android 保持零 ONNX 体积并使用规则引擎；仅在 `desktop:app` Worker 注入双塔 ONNX，关闭 `memoryPatternOptimization` 并复用单个 `OrtEnvironment`。
4. **M2 上线前真机核实三项游戏语义**：`retire()` 扣票与可结算时点、老虎机 `23161/33161` 后续阶段 `condition_value` 是否净扣本金、事件战与 `50002` 免死在结算分项中的计数口径。
5. **Phase 5 离线回放基准**：待桌面端导出脱敏开局地图录制 `replays.jsonl` 后回填 `env/run_env.py` 基准对照；[TASK.md](TASK.md) 预留的 **20 CC** 供后续 DB 变动或 10 层扩展使用。
