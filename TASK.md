# TASK · LimingjieNN 云端训练与自动监督任务书

本任务书面向在 **Google Colab / 云端 GPU 环境**执行 `LimingjieNN` 训练流水线，并供**自动监督 Agent** 巡检云端产物、判断阶段门禁、执行断点续跑与异常处理。

> 执行入口与对应参数以仓库内 [notebooks/colab_labyrinth_pipeline.ipynb](notebooks/colab_labyrinth_pipeline.ipynb)、[scripts/colab_setup_simulator.sh](scripts/colab_setup_simulator.sh) 及 `configs/*.json` 为准。

---

## 1. 仓库与云端目录约定

| 项 | 约定路径 / 地址 | 说明 |
| :--- | :--- | :--- |
| **GitHub 私有仓库** | `https://github.com/Flysky701/LimingjieNN`（`main` 分支） | 仅存放代码、配置、脚本、Notebook 与任务书 |
| **Colab 本地工作副本** | `/content/LimingjieNN` 或 `/content/LimingjieNN-drive` | 分别用于私有 GitHub 克隆和 Drive `repo/` 同步 |
| **Google Drive 根目录** | `/content/drive/MyDrive/LimingjieNN/` | 持久化存储 DB、样本、Checkpoint 与导出包 |
| ├ `master_cn.db` | `$DRIVE_ROOT/master_cn.db` | 离线锁定国服主库（`BossVersionCN=202611131613`，见 `configs/sim_profile_highscore_cn.json`）；**严禁入库 Git** |
| ├ `cache/` | `$DRIVE_ROOT/cache/` | 缓存 `PCRCalculator-2.11.0-server-linux.zip`。各端口使用本地 `/tmp/pcr_calc_persistent_<port>` |
| ├ `data/` | `$DRIVE_ROOT/data/` | `feature_pack.json`、`round0_pilot.jsonl` 及 `round0_pilot.part-*.jsonl` |
| ├ `runs/` | `$DRIVE_ROOT/runs/ablation_round0/` | `<mode>/seed_<seed>/{last.pt,best.pt,test_metrics.json}`、`ablation_summary.{json,md}` |
| ├ `export_bundle/` | `$DRIVE_ROOT/export_bundle/<winning_mode>/` | `tower_a.onnx`、`tower_b.onnx`、`manifest.json`、`golden_io.json` |
| └ `replays.jsonl` | `$DRIVE_ROOT/replays.jsonl` | （可选）真实开局地图录制文件，用于 Path B 回放对照 |

建议在 Colab Shell / Notebook 中统一导出环境变量：

```bash
export DRIVE_ROOT="/content/drive/MyDrive/LimingjieNN"
export CACHE_DIR="$DRIVE_ROOT/cache"
export DATA_DIR="$DRIVE_ROOT/data"
export RUNS_DIR="$DRIVE_ROOT/runs"
export EXPORT_DIR="$DRIVE_ROOT/export_bundle"
export CN_DB_PATH="$DRIVE_ROOT/master_cn.db"
export CN_DB_VERSION="202611131613"
```

---

## 2. 阶段进度看板（监督 Agent 状态基准）

未完成的阶段按实际产物和门禁验收。后续放量与训练按第 5 节执行。

- [x] **Round 0 Pilot** · 北京时间 **2026-11-23 03:47** 完成。68 个分片包含 **25,000** 个唯一样本（44,155 个模拟子局）。`probe_round0.json` 记录 **13.02 sims/s**。胜出配置 `m2b` 的双塔 ONNX 合计 **7.91 MiB**，`max_abs_diff = 2.384e-6`。
  - `m2b` 与 `Quest-Only` 的 AUROC 分别为：`normal` **0.713 vs 0.771**（`normal` 上 `Quest-Only` 已超过 `m2b`，且测试集仅 18 个非胜样本）、`robot_ex` **0.769 vs 0.742**（`m2b` 仅领先 `+0.028`，且全猜胜准确率 `84.34%` 高于 `m2b` `83.87%`）。后续按同一关卡、库存与公会的配对候选队伍直接检验同上下文排序。

- [x] **Phase 0 · 环境依赖与全套单测门禁**（Round 0 Colab 实跑已验证；Round 1 / 30 CC 方案见第 5 节 `S0`）
- [x] **Phase 1 · 特征目录生成与模拟器口径对账**（`probe_round0.json` 16/16 通过，`13.02 sims/s`）
- [x] **Phase 2 · Round 0 分层自适应采样（2k 烟测 → 25k Pilot）**（68 分片，25,000 条）
- [x] **Phase 3 · Round 0 四档消融训练（`m0 -> m1 -> m2a -> m2b`）与独立温度校准**（`seed=42` 完成，`m2b` Global AUROC `0.94976`）
- [x] **Phase 4 · Round 0 ONNX 双塔导出与 `< 1e-4` 数值对齐验收**（`7.91 MiB`，`max_abs_diff = 2.384e-6`）
- [ ] **Phase 5 · Path B 离线真实地图回放基准（可选，依赖 `replays.jsonl`）**

---

## 3. 分阶段云端执行指令与验收门禁

### Phase 0 · 环境与单测门禁

```bash
cd /content/LimingjieNN
pip install -q -e ".[train]"
PYTHONPATH=src python -m unittest discover -s tests -v
```

**验收门禁**：
- `tests/` 下全部单元测试通过（不可跳过 `test_export_manifest_replay_inputs_and_runtime_parity`，确保已正确安装 `torch`、`numpy`、`onnx`、`onnxruntime`）。

---

### Phase 1 · 特征目录与模拟器口径对账

1. 拉起无头 `PCRCalculatorServer`（离线锁定 `BossVersionCN=202611131613`）：

```bash
chmod +x scripts/colab_setup_simulator.sh
bash scripts/colab_setup_simulator.sh \
  --cache-dir "$CACHE_DIR" \
  --db-path "$CN_DB_PATH" \
  --db-version "$CN_DB_VERSION" \
  --port 8080
```

2. 提取国服客观特征目录：

```bash
PYTHONPATH=src python -m labyrinth_nn.catalog.build_catalog \
  --db "$CN_DB_PATH" \
  --out "$DATA_DIR/feature_pack.json"
```

3. 运行 15 项国服口径对账与吞吐压测：

```bash
PYTHONPATH=src python -m labyrinth_nn.data.probe_simulator \
  --catalog "$DATA_DIR/feature_pack.json" \
  --profile configs/sim_profile_highscore_cn.json \
  --url http://127.0.0.1:8080 \
  --bench-count 64
```

**验收门禁**：
- `feature_pack.json` 成功生成且包含 `db_version`、`catalog_sha256`、`units`、`enemies`、`quests`、`relics`。
- `probe_simulator` 退出码为 `0`，输出 JSON 中 `"passed": true`，并记录压测吞吐 `benchmark.sims_per_sec`。

---

### Phase 2 · Round 0 分层自适应采样复现（2k 烟测 → 25k Pilot）

1. 先执行分布干跑检查（不请求模拟器）：

```bash
PYTHONPATH=src python -m labyrinth_nn.data.sampler \
  --catalog "$DATA_DIR/feature_pack.json" \
  --dry-run 2000
```

2. 在高速本地 `/tmp` 执行分片断点续采（`2000` 烟测 → `25000` Pilot）。Round 1 走第 5 节 S1。

```bash
mkdir -p "$DATA_DIR"
# 续采前：同步 Drive 上已有的主文件及所有 .part-*.jsonl 分片到本地 /tmp
cp -f "$DATA_DIR"/round0_pilot*.jsonl /tmp/ 2>/dev/null || true

PYTHONPATH=src python -m labyrinth_nn.data.sampler \
  --catalog "$DATA_DIR/feature_pack.json" \
  --profile configs/sim_profile_highscore_cn.json \
  --sampling configs/sampling_round0.json \
  --url http://127.0.0.1:8080 \
  --out /tmp/round0_pilot.jsonl \
  --target-battles 25000 \
  --resume

# 采样结束或中断后：把全部新增/更新分片原子回写到 Drive
cp -f /tmp/round0_pilot*.jsonl "$DATA_DIR/"
```

> **关键分片避坑说明**：
> `sampler.py` 每累计 `chunk_size=5000`（见 `configs/sampling_round0.json`）或在捕获 `SimulationError` / `KeyboardInterrupt` 时，会把样本分片写入 `round0_pilot.jsonl` 及 `round0_pilot.part-00001.jsonl`、`round0_pilot.part-00002.jsonl` ……（见 `_shard_path`）。
> **切勿只同步 `round0_pilot.jsonl` 单文件**！无论恢复续采还是后续训练，都必须带上全部 `round0_pilot*.jsonl` 分片，否则会丢失已采样本并导致重复采样或缺失数据。

**验收门禁**：
- `sampler.py` 退出码为 `0`，返回 JSON 中 `total_battles` 达到目标体量（烟测 `2000`、Pilot `25000`）。
- Drive `$DATA_DIR/` 中主文件与所有 `.part-NNNNN.jsonl` 分片时间戳最新且非空。

---

### Phase 3 · 四档消融训练与独立温度校准（`m0 -> m1 -> m2a -> m2b`）

1. 运行四档消融阶梯（`train_critic.py --samples` 支持 `nargs="+"`，直接传入全部分片）：

```bash
PYTHONPATH=src python -m labyrinth_nn.train.train_critic \
  --configs m0 m1 m2a m2b \
  --seeds 42 \
  --amp \
  --catalog "$DATA_DIR/feature_pack.json" \
  --samples "$DATA_DIR"/round0_pilot.jsonl "$DATA_DIR"/round0_pilot.part-*.jsonl \
  --out-dir "$RUNS_DIR/ablation_round0"
```

2. 对胜出配置执行多种子复验（示例 3 种子集成）：

```bash
PYTHONPATH=src python -m labyrinth_nn.train.train_critic \
  --configs m2b \
  --seeds 42 43 44 \
  --amp \
  --catalog "$DATA_DIR/feature_pack.json" \
  --samples "$DATA_DIR"/round0_pilot.jsonl "$DATA_DIR"/round0_pilot.part-*.jsonl \
  --out-dir "$RUNS_DIR/ablation_round0"
```

3. Colab 断线后单次恢复续训（**严格约束**）：
   `train_critic.py --resume` 仅允许恢复**单个配置 + 单个种子**（`len(configs) * len(seeds) == 1`），且传入的 `--catalog`、`--samples`（全部分片完全一致）及 `--split-seed` 必须与生成 `last.pt` 时一致，否则会因 `split_sha256` 或配置校验报错退出：

```bash
PYTHONPATH=src python -m labyrinth_nn.train.train_critic \
  --config configs/critic_<mode>.json \
  --seeds <seed> \
  --amp \
  --resume "$RUNS_DIR/ablation_round0/<mode>/seed_<seed>/last.pt" \
  --catalog "$DATA_DIR/feature_pack.json" \
  --samples "$DATA_DIR"/round0_pilot.jsonl "$DATA_DIR"/round0_pilot.part-*.jsonl \
  --out-dir "$RUNS_DIR/ablation_round0"
```

**验收门禁**：
- 每个 `<mode>/seed_<seed>/` 下生成 `last.pt`、`best.pt`、`test_metrics.json`。
- 根目录生成 `ablation_summary.md` 与 `ablation_summary.json`。
- 在 `global`、6 个分桶（`normal`、`robot_ex`、`special_dual_ex`、`other_ex`、`boss_3f_d5`、`boss_5f_d5`）和 `gold_unknown` 切片上核验关键指标：`auroc`、`brier_3class`、`brier_win`、`ece_15`、`accuracy_3class`、`loss_damage_mae`、`win_time_mae`、`group_top1_fpr`、`group_top1_pos_fpr`。

---

### Phase 4 · ONNX 导出与数值一致性验收

```bash
WINNING_MODE="m2b"
PYTHONPATH=src python -m labyrinth_nn.export.export_onnx \
  --checkpoint "$RUNS_DIR/ablation_round0/${WINNING_MODE}/seed_42/best.pt" \
  --catalog "$DATA_DIR/feature_pack.json" \
  --out-dir "$EXPORT_DIR/${WINNING_MODE}"
```

**验收门禁**：
- `$EXPORT_DIR/${WINNING_MODE}/` 下完整输出 `tower_a.onnx`、`tower_b.onnx`、`manifest.json`、`golden_io.json`。
- `manifest.json` 中 `validation.max_abs_diff < 1e-4`（覆盖 6 类边界用例：`empty_relics`、`underfull_team`、`tied_search_area_width`、`gold_unknown`、`robot_ex`、`boss_5f_d5` 以及 `batch_6` 批推理）。
- `tower_a.onnx + tower_b.onnx` 合计文件体积 `< 10MB`，满足 `LimingjieDesktop` 接入体量限制。

---

### Phase 5 · Path B 离线真实地图回放基准（可选）

当 `$DRIVE_ROOT/replays.jsonl` 就绪时，可参照 [notebooks/colab_labyrinth_pipeline.ipynb](notebooks/colab_labyrinth_pipeline.ipynb) Cell 7 执行策略对照：

```bash
PYTHONPATH=src python -c "
from pathlib import Path
from labyrinth_nn.catalog.schema import FeatureCatalog
from labyrinth_nn.env.run_env import PathBReplayPolicy, compare_policies, load_replays

catalog = FeatureCatalog.load(Path('$DATA_DIR/feature_pack.json'))
episodes = load_replays(Path('$DRIVE_ROOT/replays.jsonl'))
reports = compare_policies(
    episodes,
    path_b=PathBReplayPolicy(target_score=53000),
    settings=catalog.score_settings,
    catalog=catalog,
    repeats=5,
)
for name, rep in reports.items():
    print(name, rep)
"
```

---

## 4. 自动监督 Agent 巡检与异常处理协议

### 4.1 Drive 产物巡检清单

自动监督 Agent 每次巡检按以下顺序核对 `$DRIVE_ROOT` 文件存在性、修改时间与内容：

1. **前置资源**：核对 `$DRIVE_ROOT/master_cn.db` 与缓存服务端压缩包。各端口启动后，核对本地工作目录及 `/tmp/pcr_calc_persistent_<port>/prod202611131613.db`。
2. **目录阶段**：`$DATA_DIR/feature_pack.json` 是否存在且与 `master_cn.db` 版本一致。
3. **采样阶段**：按轮次核对 manifest、分片行数和 SHA256；对比唯一成功样本数与该阶段目标。
4. **训练阶段**：检查 `$RUNS_DIR/ablation_round0/{m0,m1,m2a,m2b}/seed_*/{last.pt,best.pt,test_metrics.json}` 及 `ablation_summary.md`；读取已完成的 `test_metrics.json` 汇报当前最佳验证 NLL 与测试切片表现。
5. **导出阶段**：检查 `$EXPORT_DIR/*/{tower_a.onnx,tower_b.onnx,manifest.json,golden_io.json}`，核验 `manifest.json` 的 `validation.max_abs_diff` 及文件总大小。

### 4.2 自动恢复规则

- **采样中断恢复**：先将 Drive 上该前缀的 JSONL 全部复制回本地，包括主文件、分片、隔离失败日志和对应的 `.manifest.json`。以相同 `--out`、`--seed`、`--round-id` 及 `--resume` 恢复。完成后回写全部分片和 manifest。训练只传入主文件和 `.part-*.jsonl`。
- **训练中断恢复**：检查对应 `<mode>/seed_<seed>/last.pt` 是否存在；存在时使用 `--config` + 单一 `--seeds` + `--resume .../last.pt` 续训，并确保 `--samples` 传入与中断前完全相同的分片集合。

### 4.3 立即停止并上报条件（禁止盲目循环重试）

出现以下任一异常时，监督 Agent 必须停止当前步骤并向用户告警：
1. `probe_simulator` 检查失败（`"passed": false`）或 `PCRCalculatorServer` 进程反复退出（附带 `/tmp/pcr_calc_server_<port>.log` 末尾日志）。
2. 训练出现非有限数值异常（`non-finite training loss` / `non-finite selection NLL`）或连续未完成且无法通过 `--resume` 推进。
3. 消融结论异常：例如 `m2b` 在 `gold_unknown` 或 `boss_5f_d5` 等核心切片上明显劣于 `m0` 基线。
4. ONNX 导出数值对齐失败（`max_abs_diff >= 1e-4`）或双塔模型合计超过 `10MB`。

### 4.4 Git 仓库卫生护栏

- **严禁提交**任何 `*.db`、`*.sqlite*`、`*.jsonl`、`*.parquet`、`*.pt`、`*.pth`、`*.onnx`、Google OAuth 凭证或 Drive 本地缓存到 `Flysky701/LimingjieNN`（`.gitignore` 已拦截常见后缀，提交前仍须 `git status` 复核）。
- 所有云端训练状态与模型产出统一落在 Google Drive `$DRIVE_ROOT`；仓库仅更新代码、配置与本 `TASK.md` 进度记录。

## 5. 30 CC Optimal Plan

按 Carbon + Astra + Opus 5.5 三方交叉评估的阶段安排执行。核心预算 **21.5 CC**，条件预算 **8.5 CC**，本轮上限 **30 CC**。另外保留 **20 CC**。采样、对账与封存评估使用 CPU Runtime。训练和导出使用 T4 GPU Runtime，阶段之间释放 GPU。

| 阶段 | CC 上限 | 执行与放行条件 |
| --- | ---: | --- |
| **S0 · Pre-Burn** | **1.0** | 核对 68 个 Pilot 分片、25,000 个唯一 ID、锁定 DB 和目录哈希。跑完单测、模拟器 15 项对账、分片恢复和隔离失败烟测、上下文分组、遮蔽金币评估、梯度裁剪与续训、ONNX 回放。用至少 200 个非胜样例审计伤害标签，覆盖机器人、Boss 和其他关卡及截断样例。终局 HP 缺失时将伤害标记为截断聚合伤害代理，并单独限制其策略用途。用代表性负载测 CPU 成本，再做短 T4 定时。 |
| **S1 · Round 1** | **2.5** | `configs/sim_profile_round1_cn.json` + `configs/sampling_round1.json`，新增 **125k → 150k 累计**。覆盖 10 公会和 6 分桶，每上下文 4 支队伍，3% 部分队伍。每队先模拟 2 次，混合胜负、高随机性、接近胜负边界或跨桶确定性随机抽样触发时加到 6 次。核对实际重复率、成功行数和隔离率。并行 CPU 工作者分别指定 `--seed`、`--round-id`、`--out` 和端口，保持可追溯的分片与 manifest。 |
| **S2 · R1 选型与集成** | **3.5** | 在固定的上下文级开发、选择、校准与测试划分上比较 `critic_m1.json`、`critic_m2b.json`、`critic_m2b_r1.json`、`critic_m2b_r1_dmg025.json`。每档先跑 2 个固定种子，胜出配置补到 **3–5 种子**形成离线集成。比较选择 NLL、同上下文 Top-1、金币遮蔽表现与经审计的连续标签。 |
| **S3 · Round 2 主动挖掘** | **5.0** | 新增 **100k → 250k 累计**。建议 40% 普通、25% 机器人、15% Boss、20% 广覆盖探索。收集预测高信心胜但实际失败、模型分歧、近胜负及一至两槽替换队伍。重复档按 50% × 1、35% × 6、15% × 12 分配，约 440k 子局，保留随机复测对照。封存面板排除在挖掘池之外。 |
| **S4 · R2 训练及蒸馏门禁** | **6.5** | 最多 8 次 250k 等效训练，胜出模型与最强对照各 3 种子，另留 2 次学生试验。先验证单模型部署版本。集成教师在开发集上改善 win Brier **≥0.003** 或降低同上下文 Top-1 失败率 **≥1 个百分点**时才启动蒸馏。蒸馏仅使用训练池预测，学生独立校准，并通过双塔 ONNX 体积、数值一致性和实机延迟门禁。 |
| **S5 · 封存评估** | **3.0** | 冻结模型、阈值及推荐规则后，使用 32k 分层样本 × 5 次（普通 12k，其余各 4k）；另取 1,000 个新上下文 × 8 支候选队 × 11 次，对两种策略各自首选队再使用 21 个独立种子确认。总计约 290k 子局。报告自然分布和难例面板，按目标流量重加权，使用上下文聚类配对置信区间。 |
| **C1 · 条件定向补采** | **4.5** | 开发集证实普通负例不足、机器人边界不稳定或伤害标签需修正时，最多追加 **90k → 340k 累计**（最多约 360k 子局）。金币遮蔽评估可复用现有标签。 |
| **C2 · 条件最终拟合** | **4.0** | C1 实际改变数据，或合格教师表明学生仍有明确收益时，最多做 4 次 340k 等效训练。单模型达到开发门禁即停止。 |

### S1 / S2 执行参考命令（CPU 采样与 T4 选型解耦）

```bash
# S1（CPU Runtime）：Round 1 覆盖 10 公会 + 同上下文 4 候选配对 + 坏样本隔离续采
cp -f "$DATA_DIR"/round1_scale*.jsonl "$DATA_DIR"/round1_scale*.json /tmp/ 2>/dev/null || true
PYTHONPATH=src python -m labyrinth_nn.data.sampler \
  --catalog "$DATA_DIR/feature_pack.json" \
  --profile configs/sim_profile_round1_cn.json \
  --sampling configs/sampling_round1.json \
  --url http://127.0.0.1:8080 \
  --out /tmp/round1_scale.jsonl \
  --target-battles 125000 \
  --isolate-errors \
  --resume
cp -f /tmp/round1_scale*.jsonl /tmp/round1_scale*.json "$DATA_DIR/"

# S2（T4 GPU Runtime）：冻结上下文划分上对比 m1 / m2b / m2b_r1 / m2b_r1_dmg025（含 counterfactual gold_unknown）
PYTHONPATH=src python -m labyrinth_nn.train.train_critic \
  --config-files configs/critic_m1.json configs/critic_m2b.json configs/critic_m2b_r1.json configs/critic_m2b_r1_dmg025.json \
  --seeds 42 43 \
  --group-by context \
  --amp \
  --catalog "$DATA_DIR/feature_pack.json" \
  --samples "$DATA_DIR"/round0_pilot.jsonl "$DATA_DIR"/round0_pilot.part-*.jsonl "$DATA_DIR"/round1_scale.jsonl "$DATA_DIR"/round1_scale.part-*.jsonl \
  --out-dir "$RUNS_DIR/ablation_round1"
```

### Go / No-Go

- **立即停止**：模拟器对账失败，DB、目录或分片哈希不符，ID 重复或切分泄漏，出现非有限损失或预测，ONNX 数值对齐失败或超出体积上限。
- **暂停放量**：代表性 CPU 烟测超过 **1.15 CC / 100k 子局**、重复种子无法对账，或核心切片样本量不足。先试标准 CPU，预算按实际子局数和计费折算。S1 的自适应规则生效时以实际重复率测算 2.5 CC 是否足够。
- **S2 转向**：若普通和机器人分桶在开发集对冻结 Round 0 模型的 AUROC 增益都 **<0.01**，且同上下文排序未改善，将下一批预算转向候选队构造和标签质量。

| 封存面板的发布门禁 | Go 阈值 |
| --- | --- |
| 区分能力 | 全局 AUROC **≥0.950**。普通、机器人 EX、特殊双 EX、其他 EX、3F Boss、5F Boss 依次 **≥0.800 / 0.820 / 0.935 / 0.915 / 0.910 / 0.945**。普通和机器人各自比同面板 Quest-Only **≥0.02**。各桶至少 **200 正例和 200 负例**。 |
| 概率质量 | 重加权三类 Brier **≤0.210**，胜率 Brier **≤0.074**，全局胜率 ECE **≤0.030**，各桶胜率 ECE **≤0.055**，三类准确率 **≥0.845**。报告独立校准前后的 NLL。 |
| 金币未知 | 至少 **2,000 个成对上下文**及 **500 个金币缩放遗物样本**。遮蔽金币后胜率 ECE **≤0.045**，相较同样本金币可见视图的胜率 Brier 增量 **≤0.010**，单列金币缩放遗物切片。 |
| 连续量输出 | 非胜样本伤害 MAE **≤0.185**、胜利余时 MAE **≤0.125**。独立终局 HP 标签单独报告。只有聚合伤害代理时，验收仅涵盖该代理的误差。 |
| 推荐风险 | 至少 **1,000 个新上下文**。预先定义的可行上下文面板上，Top-1 失败率 **≤5%**；预测为胜的 Top-1 失败率 **≤3%**、单侧 95% 置信上界 **≤5%**、覆盖率 **≥80%**。同时报告分子分母、无可行队难例与未过滤流量。 |
| 桌面导出 | 两个 ONNX 合计 **<10,000,000 bytes**。输出有限且双塔串联 `max_abs_diff < 1e-4`。验收真实特征输入、金币已知/未知、站位并列、部分队伍和缓存上下文多队评分。 |
| 桌面延迟 | 目标桌面至少测 **200 个上下文**。热态 CPU p95：Tower A **≤25 ms**，Tower B 64 队一批 **≤30 ms**，512 候选完整推荐 **≤500 ms**（含特征计算和配队）。记录硬件、线程与冷启动。 |

### 20 CC 储备

| CC | 触发与用途 |
| ---: | --- |
| **8** | 国服 DB、模拟器、库存/OCR 或楼层变化引入分布偏移。先复跑成对样例，再定向采购约 **300k–500k 子局**，保留新版本封存面板。 |
| **5** | 分桶 AUROC 低于门槛 **≥0.03**，普通/机器人同上下文失败率恶化 **≥2 个百分点**，或独立 HP 证据确认系统性标签错误。定向扩增或重标。 |
| **4** | 三种子教师达到 S4 蒸馏门槛，单模型效果或延迟仍未达标。训练学生或更窄模型，独立校准。 |
| **3** | 候选已通过开发门禁，进入正式发布确认。使用全新确认面板、目标桌面计时、导出对齐和可用的真实地图回放。 |
