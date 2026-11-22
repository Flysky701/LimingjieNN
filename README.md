# LimingjieNN · 黎明界高分神经网络与全局结算分规划系统

`LimingjieNN` 是面向《公主连结 Re:Dive》（国服）黎明界高分上限场景的独立决策训练与规划仓库，定位为 `LimingjieDesktop`（`refactor/event-kernel`）上层策略模块的洁净替代实现。

## 核心架构（Level 1 + Path B 两层协作）

1. **Level 1 · Two-Tower 晚交互双塔战斗网络 (`V_battle`)**
   - **Tower A (`ContextRosterEncoder`)**：每次进编组只运行 1 次，联合编码敌方波次、遗物集合（内置永不掩码的 `[NULL_RELIC]` Sink Token 防 ONNX `NaN`）、5 系印记叠层、41 强化节点、金币及缺失态、以及完整持有角色池，输出全池条件化表征与单兵契合分。
   - **Tower B (`SlotAwareTeamHead`)**：固定 5 槽轻量交互头，严格按 `(search_area_width, unit_id)` 升序排位（覆盖 31 组同站位碰撞角色），注入带符号成对站位差矩阵（含 300/400 半径物理阈值特征）与全队协同标量，输出三头：
     - `(p_win, p_timeout, p_wipe)` 三分类结局（纯软标签交叉熵，禁用冲突的 Focal Loss）；
     - `damage_ratio` 敌方削血比例分布（剔除 `770900501` 难度环境敌与过量伤害）；
     - `win_rem_time` 胜场剩余时间比（仅胜场回传梯度）。
   - **四档可归因消融阶梯**：`m0`（纯 ID 双塔）→ `m1`（+高分全局量/金币高比例缺失/印记耦合噪声）→ `m2a`（+带符号站位差偏置）→ `m2b`（+国服静态面板与 10 个干净原子二值机制标记 + 15% ID Dropout）。
   - **搜索策略**：当前 5 层以「单队 LCB 保守胜率优先 + 胜场剩余时间决胜」为主，辅以基于单兵契合分种子的 `1/2-swap` 局部搜索、失败队签名剔除及精确 3-Set Packing 多队兜底；数据结构与网络词表（`max_areas=16`, `max_difficulties=10`）全面动态读取 `master_cn.db`，可直接平滑升级至后续 10 层多队版本。

2. **Path B · 全局官方结算分精确数学规划器（不使用 MBRL `V_run`）**
   - 统一度量衡为国服官方结算分（`labyrinth_setting` 101–106 / 201–225 + `labyrinth_difficulty_data`）。
   - **`route_dp.py`**：以分数最大公约数构建 300 桶（支持 10 层动态扩桶）离散概率分布 DAG 反向 DP，支持 `t=0` 开局协议全图筛图（绝不累加互斥分支）、`T=0` 日常稳健清票与 `T>0`（如 `52500/53000`）Renewal-Reward 冲榜与两道自动早停门禁。
   - **`shop_dp.py`**：支持节点 `#21 (-25%)` + 救济院 `40003 (-10%)` 加法 `0.65` 折、递增刷新费与 `50003/50009` 持币留存耦合的记忆化期望 DP 与快速 `gold -> pts` 价值曲线。
   - **`marginal_eval.py`**：角色三选一、遗物三选一（保留无印记遗物如 `30001`）与事件（老虎机/石板支持可配置 `deduct_condition_principal`）全部按「直接收集分 + 金币兑换分 + 剩余考题 LCB 过关分」同尺度估值。
   - **`env/run_env.py`**：离线真实开局地图回放与策略灵敏度评测沙盒（仅作基准对照与计分对账，所有开局重刷、翻车与中途撤退严格计入分母）。

## 目录结构

```text
LimingjieNN/
├── configs/
│   ├── sim_profile_highscore_cn.json   # 难度5/高分模拟与自适应复测配置
│   ├── sampling_round0.json            # 6桶分层采样与边界探针配比
│   └── critic_m0..m2b.json             # 4档消融实验配置
├── src/labyrinth_nn/
│   ├── catalog/{schema,build_catalog}.py
│   ├── data/{sample_schema,sim_client,probe_simulator,sampler,dataset}.py
│   ├── models/critic.py
│   ├── train/{losses,metrics,train_critic}.py
│   ├── export/export_onnx.py
│   ├── planning/{scoring,route_dp,shop_dp,marginal_eval}.py
│   └── env/run_env.py
├── scripts/colab_setup_simulator.sh    # Colab 一键部署离线锁库 PCRCalculatorServer
├── notebooks/colab_labyrinth_pipeline.ipynb
└── tests/                              # 全套 stdlib 单测 + Colab Torch/ORT 校验
```

## 快速开始

### 1. 本地 Windows 环境校验（仅需 Python 3.11+ 标准库）

```powershell
cd D:\Project\LimingjieNN
$env:PYTHONPATH = "D:\Project\LimingjieNN\src"

# 运行全套本地单测（含真实 master_cn.db 集成验证与 Path B 规划测试）
python -m unittest discover -s tests -v

# 从国服数据库生成特征目录（数据库文件本身不入库）
python -m labyrinth_nn.catalog.build_catalog `
  --db "D:\path\to\master_cn.db" `
  --out "artifacts/feature_pack.json"

# 采样分布干跑验证
python -m labyrinth_nn.data.sampler `
  --catalog "artifacts/feature_pack.json" `
  --dry-run 2000
```

### 2. Google Colab 端到端流水线

直接上传或在 Colab 打开 [`notebooks/colab_labyrinth_pipeline.ipynb`](notebooks/colab_labyrinth_pipeline.ipynb)，按顺序执行：
1. 挂载 Google Drive，安装 `pip install -e ".[train]"` 并运行全套单元测试（含 PyTorch 与 ONNX Runtime `< 1e-4` 对齐测试）；
2. 运行 `scripts/colab_setup_simulator.sh` 挂载锁定版本的 `master_cn.db`（映射为 `prod<BossVersionCN>.db`）拉起 `PCRCalculatorServer`，并跑 `probe_simulator.py` 完成国服口径对账与 `sims/s` 压测；
3. 运行 `sampler.py` 进行断点续传分片造数（2k 烟测 → 25k Pilot → 150k 放量）；
4. 运行 `train_critic.py --configs m0 m1 m2a m2b` 输出消融对比表；
5. 运行 `export_onnx.py` 导出 `tower_a.onnx`、`tower_b.onnx`、`manifest.json` 与 `golden_io.json`。

## 后续接入 `LimingjieDesktop` 交接清单

在把导出的 ONNX 模型与 Path B 规划逻辑接入 `LimingjieDesktop`（`refactor/event-kernel`）前，需同步完成三件事：
1. **补齐 AlphaCoin 紧凑数字 OCR**：当前生产管线中 `observedAlphaCoin` 未稳定传值，接入前需在商店/事件/地图链路补齐金币 OCR 与局内收支账本。
2. **遵守模块隔离与 SHA 锁**：在共享纯 Kotlin 层定义轻量策略 SPI（Android 保持零 ONNX 体积并默认回退规则引擎），仅在 `desktop:app` Worker 进程组装根注入 `<10MB` ONNX Runtime 实现（关闭 `memoryPatternOptimization` 并复用单 `OrtEnvironment`）。
3. **上线冲榜策略前真机核实 3 条游戏语义**：`retire()` 扣票与可结算时点、老虎机 `23161/33161` 后续阶段 `condition_value` 是否净扣本金（可在 `marginal_eval` 一键切换 `deduct_condition_principal`）、以及事件战与 `50002` 免死在结算分项中的计数口径。
