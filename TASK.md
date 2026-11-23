# TASK · LimingjieNN 云端训练与自动监督任务书

本任务书面向在 **Google Colab / 云端 GPU 环境**执行 `LimingjieNN` 训练流水线，并供**自动监督 Agent** 巡检云端产物、判断阶段门禁、执行断点续跑与异常处理。

> 执行入口与对应参数以仓库内 [notebooks/colab_labyrinth_pipeline.ipynb](notebooks/colab_labyrinth_pipeline.ipynb)、[scripts/colab_setup_simulator.sh](scripts/colab_setup_simulator.sh) 及 `configs/*.json` 为准。

---

## 1. 仓库与云端目录约定

| 项 | 约定路径 / 地址 | 说明 |
| :--- | :--- | :--- |
| **GitHub 私有仓库** | `https://github.com/Flysky701/LimingjieNN`（`main` 分支） | 仅存放代码、配置、脚本、Notebook 与任务书 |
| **Colab 本地工作副本** | `/content/LimingjieNN` | 高速本地盘克隆或同步自 Drive `repo/` |
| **Google Drive 根目录** | `/content/drive/MyDrive/LimingjieNN/` | 持久化存储 DB、样本、Checkpoint 与导出包 |
| ├ `master_cn.db` | `$DRIVE_ROOT/master_cn.db` | 离线锁定国服主库（`BossVersionCN=202611131613`，见 `configs/sim_profile_highscore_cn.json`）；**严禁入库 Git** |
| ├ `cache/` | `$DRIVE_ROOT/cache/` | 缓存 `PCRCalculator-2.11.0-server-linux.zip` 与 `persistent/prod202611131613.db` |
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

> 说明：在挂载或读取到对应 Google Drive 账号内容前，所有阶段统一保持未验收状态；监督 Agent 检查完 Drive 实际产物后按验收门禁逐项勾选并补充时间戳与关键指标。

- [ ] **Phase 0 · 环境依赖与全套单测门禁**
- [ ] **Phase 1 · 特征目录生成与模拟器口径对账**
- [ ] **Phase 2 · 分层自适应采样造数（2k 烟测 → 25k Pilot → 最高 150k 放量）**
- [ ] **Phase 3 · 四档消融训练（`m0 -> m1 -> m2a -> m2b`）与独立温度校准**
- [ ] **Phase 4 · ONNX 双塔导出与 `< 1e-4` 数值对齐验收**
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

### Phase 2 · 分层自适应造数（2k 烟测 → 25k Pilot → 最高 150k 放量）

1. 先执行分布干跑检查（不请求模拟器）：

```bash
PYTHONPATH=src python -m labyrinth_nn.data.sampler \
  --catalog "$DATA_DIR/feature_pack.json" \
  --dry-run 2000
```

2. 在高速本地 `/tmp` 执行分片断点续采（推荐阶梯：`2000` 烟测 → `25000` Pilot → 最多 `150000` Full）：

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
- `sampler.py` 退出码为 `0`，返回 JSON 中 `total_battles` 达到目标体量（烟测 `2000`、Pilot `25000`、放量上限 `150000`）。
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
  --samples "$DATA_DIR"/round0_pilot*.jsonl \
  --out-dir "$RUNS_DIR/ablation_round0"
```

2. 对胜出配置执行多种子复验（示例 3 种子集成）：

```bash
PYTHONPATH=src python -m labyrinth_nn.train.train_critic \
  --configs m2b \
  --seeds 42 43 44 \
  --amp \
  --catalog "$DATA_DIR/feature_pack.json" \
  --samples "$DATA_DIR"/round0_pilot*.jsonl \
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
  --samples "$DATA_DIR"/round0_pilot*.jsonl \
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

1. **前置资源**：`$DRIVE_ROOT/master_cn.db` 是否存在；`$CACHE_DIR/` 下是否已缓存服务端压缩包与 `persistent/prod202611131613.db`。
2. **目录阶段**：`$DATA_DIR/feature_pack.json` 是否存在且与 `master_cn.db` 版本一致。
3. **采样阶段**：检查 `$DATA_DIR/round0_pilot*.jsonl` 的分片总数、文件大小与最后修改时间；对比已采总样本数与 `target_battles`（2k / 25k / 150k）。
4. **训练阶段**：检查 `$RUNS_DIR/ablation_round0/{m0,m1,m2a,m2b}/seed_*/{last.pt,best.pt,test_metrics.json}` 及 `ablation_summary.md`；读取已完成的 `test_metrics.json` 汇报当前最佳验证 NLL 与测试切片表现。
5. **导出阶段**：检查 `$EXPORT_DIR/*/{tower_a.onnx,tower_b.onnx,manifest.json,golden_io.json}`，核验 `manifest.json` 的 `validation.max_abs_diff` 及文件总大小。

### 4.2 自动恢复规则

- **采样中断恢复**：先将 Drive 上全部分片 `round0_pilot*.jsonl` 复制回 `/tmp/`，再带 `--resume` 重新执行 `sampler.py`；完成后必须把全部新增 `.part-*.jsonl` 同步回 Drive。
- **训练中断恢复**：检查对应 `<mode>/seed_<seed>/last.pt` 是否存在；存在时使用 `--config` + 单一 `--seeds` + `--resume .../last.pt` 续训，并确保 `--samples` 传入与中断前完全相同的分片集合。

### 4.3 立即停止并上报条件（禁止盲目循环重试）

出现以下任一异常时，监督 Agent 必须停止当前步骤并向用户告警：
1. `probe_simulator` 检查失败（`"passed": false`）或 `PCRCalculatorServer` 进程反复退出（附带 `/tmp/pcr_calc_server.log` 末尾日志）。
2. 训练出现非有限数值异常（`non-finite training loss` / `non-finite selection NLL`）或连续未完成且无法通过 `--resume` 推进。
3. 消融结论异常：例如 `m2b` 在 `gold_unknown` 或 `boss_5f_d5` 等核心切片上明显劣于 `m0` 基线。
4. ONNX 导出数值对齐失败（`max_abs_diff >= 1e-4`）或双塔模型合计超过 `10MB`。

### 4.4 Git 仓库卫生护栏

- **严禁提交**任何 `*.db`、`*.sqlite*`、`*.jsonl`、`*.parquet`、`*.pt`、`*.pth`、`*.onnx`、Google OAuth 凭证或 Drive 本地缓存到 `Flysky701/LimingjieNN`（`.gitignore` 已拦截常见后缀，提交前仍须 `git status` 复核）。
- 所有云端训练状态与模型产出统一落在 Google Drive `$DRIVE_ROOT`；仓库仅更新代码、配置与本 `TASK.md` 进度记录。
