# `s4_m2b_r1_seed43` 模型发布说明

## 模型概览与溯源

本包为 `LimingjieDesktop` 提供国服黎明界的战斗结局估计。Tower A 编码当前关卡、遗物、库存等上下文，每次编组运行一次。Tower B 复用上下文，对候选队伍批量估计胜率、伤害代理和胜场余时。

| 项目 | 发布值 |
| --- | --- |
| 发布标识 | `s4_m2b_r1_seed43` |
| 导出格式 | `limingjie_twotower_onnx_v1`，ONNX opset 17 |
| 来源 | `m2b_r1 / seed_43`，epoch 21，温度 `T = 0.939903` |
| `checkpoint_sha256` | `f5e4ead9259c7ba80a611cacc124f586aba40c43d4b07fe021476376fbf26b50` |
| S4 冻结上下文切分 | 301,000 条，训练 256,065、验证选择 11,293、验证校准 11,223、测试 22,419 |
| 国服锁定 DB | `202611131613` |
| `db_sha256` | `c54dccbe32fc91a1c79ba70330d5ca63f21516d6e1cd51af278d2a0586d17016` |
| `catalog_sha256` | `e356c66a68c09da1c43ac2c5cf4492777c856850259c65429b952526f0839259` |
| 特征目录覆盖 | 308 角色、1,171 敌人、289 关卡、101 遗物，`max_area=5`、`max_difficulty=5` |

蒸馏在冻结验证选择集上判定。`win_brier_gain = 0.00241 < 0.003`，同上下文 Top-1 失败率下降约 0.796 个百分点，低于 1 个百分点门槛。蒸馏未触发，本次发布单模型。

## 文件清单

MiB 以 1,048,576 字节为单位，表内保留六位小数。`catalog_sha256` 是特征目录内容标识，`feature_pack.json` 的文件 SHA256 见下表。

| 文件 | 字节数 | MiB | 用途 | 文件 SHA256 |
| --- | ---: | ---: | --- | --- |
| `tower_a.onnx` | 6,387,100 | 6.091213 | 编码上下文与持有角色池 | `0c5470cba376c318bb7bdee3314d109b554b2bacd5b0fa6ed1bef45c0b535362` |
| `tower_b.onnx` | 1,912,248 | 1.823662 | 批量评分候选队伍 | `3798374413377f07f3c2e06fbe234a51be908ff1b48b7d6517cbae4c0a787ce9` |
| `manifest.json` | 9,734 | 0.009283 | 模型版本、数据指纹、张量契约及导出验收 | `27bda0804cfdf37d57413ccfb9d34e6b0c1cd3f5812b22d0a294745d08c1168d` |
| `golden_io.json` | 2,244,508 | 2.140530 | 双塔端到端黄金输入与预期输出 | `7848b03b63086d8fb82d7ecfa2dbdd164691c8bc86240c406143d8cb580dd160` |
| `feature_pack.json` | 1,544,895 | 1.473327 | 打包的国服锁库特征目录 | `5a1492aaef32d2588eb5e29a2a9ad740b73118a2965c11484dd841f4e57fded5` |
| `eval_summary.json` | — | — | 封存评测摘要 | 随发布包生成，字节数与校验和详见 `SHA256SUMS.txt` |
| `SHA256SUMS.txt` | — | — | 发布包文件校验清单 | 随发布包生成 |

两个 ONNX 文件合计 **8,299,348 bytes**（**7.915 MiB < 10 MB**）。

## 输入输出张量契约

### Tower A 输入（12 个）

| 张量 | Dtype | 形状 |
| --- | --- | --- |
| `quest_idx` | `tensor(int64)` | `[batch]` |
| `area_idx` | `tensor(int64)` | `[batch]` |
| `difficulty_idx` | `tensor(int64)` | `[batch]` |
| `enemy_ids` | `tensor(int64)` | `[batch, 5]` |
| `enemy_mask` | `tensor(bool)` | `[batch, 5]` |
| `roster_unit_ids` | `tensor(int64)` | `[batch, 64]` |
| `roster_mask` | `tensor(bool)` | `[batch, 64]` |
| `relic_ids` | `tensor(int64)` | `[batch, 32]` |
| `relic_mask` | `tensor(bool)` | `[batch, 32]` |
| `enemy_feats` | `tensor(float)` | `[batch, 5, 11]` |
| `roster_unit_feats` | `tensor(float)` | `[batch, 64, 27]` |
| `global_feats` | `tensor(float)` | `[batch, 41]` |

### Tower A 输出（3 个）

| 张量 | Dtype | 形状 |
| --- | --- | --- |
| `roster_ctx` | `tensor(float)` | `[batch, 64, 128]` |
| `unit_affinity` | `tensor(float)` | `[batch, 64]` |
| `context_summary` | `tensor(float)` | `[batch, 128]` |

### Tower B 输入（6 个）

| 张量 | Dtype | 形状 |
| --- | --- | --- |
| `roster_ctx` | `tensor(float)` | `[batch, 64, 128]` |
| `context_summary` | `tensor(float)` | `[batch, 128]` |
| `team_roster_pos` | `tensor(int64)` | `[batch, 5]` |
| `team_slot_mask` | `tensor(bool)` | `[batch, 5]` |
| `team_pos_matrix` | `tensor(float)` | `[batch, 5, 5, 3]` |
| `team_synergy` | `tensor(float)` | `[batch, 17]` |

### Tower B 输出（4 个）

| 张量 | Dtype | 形状 | 说明 |
| --- | --- | --- | --- |
| `outcome_probs` | `tensor(float)` | `[batch, 3]` | 列顺序为 `[win, timeout, wipe]`，图内已应用温度缩放 `T = 0.9399030804634094`，第一列为 `p_win` |
| `damage_mean` | `tensor(float)` | `[batch, 1]` | |
| `damage_std` | `tensor(float)` | `[batch, 1]` | |
| `win_rem_time` | `tensor(float)` | `[batch, 1]` | |

### 接入约定

- `roster` 前部的有效角色与队内站位均严格按 `(search_area_width, unit_id)` 升序排列。
- 无遗物时保留索引 0 的内置 `[NULL_RELIC]` 有效位，避免遗物注意力全掩码后产生 `NaN`。
- `alpha_rupee = null` 时，`global_feats` 中的金币特征填 `0`，`gold_known` 填 `0.0`。

## S5 封存评测

### 32,000 条分层战绩

单模型全局 AUROC 为 **0.9914**，胜率 ECE（15 分箱）为 **0.0085**。按目标流量重加权的三分类 Brier（`reweighted_brier_3class`）为 **0.0947028373178458**。下表对照同一面板上的 Quest-Only 基线。

| 分桶 | 样本数 | 模型 AUROC | Quest-Only AUROC | 模型 ECE | Quest-Only ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `normal` | 12,000 | 0.9900 | 0.8931 | 0.0029 | 1.04e-10 |
| `robot_ex` | 4,000 | 0.9681 | 0.7658 | 0.0133 | 6.10e-10 |
| `special_dual_ex` | 4,000 | 0.9770 | 0.7372 | 0.0279 | 4.03e-10 |
| `other_ex` | 4,000 | 0.9775 | 0.8523 | 0.0197 | 2.42e-10 |
| `boss_3f_d5` | 4,000 | 0.9768 | 0.5801 | 0.0152 | 1.15e-10 |
| `boss_5f_d5` | 4,000 | 0.9838 | 0.6177 | 0.0346 | 1.48e-10 |

对同一批样本全量遮蔽金币，Win Brier 从 0.03146849 到 0.03150109，变化 **+0.00003**，胜率 ECE 从 0.00854 到 0.00895。`gold_scaling_relic` 切片覆盖 **14,780 行**，遮蔽前后 Win Brier 为 0.03199795 和 0.03202193，ECE 为 0.00918 和 0.00849。

### 1,000 个配对上下文与阈值

对每个上下文比较 8 支候选队，首选队使用 21 个独立种子复测。报告中的可行子集有 **779** 个上下文。以下两组数字均取 `p_win >= 0.50`，失败率以达到阈值的上下文数为分母。

| 模型 | 全量 1,000 个上下文（比较汇总） | 779 个可行上下文（阈值诊断） |
| --- | ---: | ---: |
| S4 `seed_43` 单模型（`s4_single_seed43`） | 33/764（4.32%） | **14/744（1.88%）** |
| S4 三种子集成（`s4_ens_3m`） | 37/772（4.79%） | 17/752（2.26%） |
| S2 三种子集成（`s2_ens_3m`） | 30/754（3.98%） | 18/741（2.43%） |

单模型在可行子集的 Top-1 失败率低于两种集成。原始 S5 报告按全量面板记录覆盖 **76.4%**、覆盖内失败率 **4.32%**、Wilson 单侧 95% 上界 **5.70%**，三项推荐风险门禁未通过。`all_gates_passed` 为 `false`。原始口径见 `s5_comparison_summary.json`，阈值诊断见 `s5_threshold_diag.json`。

| `p_win` 阈值 | 全量覆盖 | 全量 Top-1 失败 | 可行子集覆盖 | 可行子集 Top-1 失败 |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 764/1,000（76.4%） | 33/764（4.32%） | 744/779（95.51%） | 14/744（1.88%） |
| 0.60 | 744/1,000（74.4%） | 22/744（2.96%） | 730/779（93.71%） | 9/730（1.23%） |
| 0.70 | 726/1,000（72.6%） | 14/726（1.93%） | 719/779（92.30%） | 7/719（0.97%） |
| 0.75 | 718/1,000（71.8%） | 13/718（1.81%） | 712/779（91.40%） | 7/712（0.98%） |

`LimingjieDesktop` 按首选候选队的校准胜率执行三档决策。

| 条件 | 桌面决策 |
| --- | --- |
| `p_win >= 0.70` | 稳妥推荐 |
| `0.50 <= p_win < 0.70` | 边缘尝试 |
| `p_win < 0.50` | 判为无稳过队，回退规则引擎 |

## 数值一致性与桌面延迟

`golden_io.json` 包含 6 个单样本黄金用例（`batch = 1`），分别为 `empty_relics`、`underfull_team`、`tied_search_area_width`、`gold_unknown`、`robot_ex`、`boss_5f_d5`。

`manifest.json` 记录的导出校验覆盖 `batch_sizes = [1, 6]`，包含 `batch_6` 批量用例。PyTorch 与 ONNX Runtime（ORT）的双塔串联输出全局 `max_abs_diff = 7.62939453125e-06`，低于 `1e-4` 的容差，数值对齐通过。

桌面 CPU 延迟报告覆盖 200 个上下文，使用 ORT 1.32.0、单算子内线程。

| 指标 | 实测 |
| --- | ---: |
| `cold_load` | 90.57 ms |
| `tower_a` p95 | 1.67 ms |
| `tower_b_64` p95 | 4.06 ms |
| `full_512` p95 | 53.54 ms |

`LimingjieDesktop` 在 `refactor/event-kernel` / `feat/nn-battle-engine` 使用 ORT Java `1.21.0` 验证加载与推理，`BattleOnnxGoldenTest` 实测 2/2 通过。

## 接入 LimingjieDesktop

1. 克隆本仓库，将 `LIMINGJIE_NN_BUNDLE` 或 JVM 参数 `-Dlimingjie.nn.bundle=<LimingjieNN>/models/s4_m2b_r1_seed43` 指向发布目录。`BattleOnnxBundle.catalogPath` 自动发现包内的 `feature_pack.json`，无需另设 `limingjie.nn.catalog`。
2. 在桌面端选择「设置 → 运行环境 → 迷宫战斗价值估计 → 使用本地 ONNX 模型（下局生效）」。未启用或哈希、版本不匹配时，自动回退规则评分。
3. 在 `LimingjieDesktop` 的 `desktop/` 目录复跑黄金对齐测试（将 `<LimingjieNN>` 替换为实际克隆位置）：

```powershell
cd desktop
$env:LIMINGJIE_NN_BUNDLE = '<LimingjieNN>/models/s4_m2b_r1_seed43'
.\gradlew.bat :app:test --tests '*BattleOnnxGoldenTest'
```

## 适用范围与已知局限

- 当前验收范围是国服锁定 DB `202611131613` 的 5 层 D5。数据库、图鉴或楼层变化时，应按新版本重新评估。
- `damage_mean` 和 `damage_std` 对应截断聚合伤害代理（`truncated_aggregate_damage_proxy`）。代理来自聚合伤害，精确终局 HP 或跨队削血计算需要独立的终局 HP 数据。
- `win_rem_time` 是胜利条件下的剩余时间比。高胜率档的实测失败率与覆盖率随阈值变化，使用时以封存范围和上面的回退规则为准。

仓库架构及桌面交接待办见 [项目 README](../../README.md)，各阶段状态见 [TASK.md](../../TASK.md)。
