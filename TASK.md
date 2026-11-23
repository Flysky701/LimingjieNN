# TASK · LimingjieNN 云端训练与自动监督任务书

本任务书面向在 **Google Colab / 云端 GPU 环境**执行 `LimingjieNN` 训练流水线，并供**自动监督 Agent** 巡检云端产物、判断阶段门禁、执行断点续跑与异常处理。

> 执行入口与对应参数以仓库内 [notebooks/colab_labyrinth_pipeline.ipynb](notebooks/colab_labyrinth_pipeline.ipynb)、[scripts/colab_setup_simulator.sh](scripts/colab_setup_simulator.sh) 及 `configs/*.json` 为准。预算、故障恢复与汇报约束见 [自动监督规则](docs/AGENT_SUPERVISION_RULES.md)。

---

## 1. 仓库与云端目录约定

| 项 | 约定路径 / 地址 | 说明 |
| :--- | :--- | :--- |
| **GitHub 私有仓库** | `https://github.com/Flysky701/LimingjieNN`（`main` 分支） | 仅存放代码、配置、脚本、Notebook 与任务书 |
| **Colab 本地工作副本** | `/content/LimingjieNN-zip`、`/content/LimingjieNN-drive` 或 `/content/LimingjieNN` | 对应 Drive 的 `source.zip`、Drive 的 `repo/` 和私有 GitHub 克隆 |
| **Google Drive 根目录** | `/content/drive/MyDrive/LimingjieNN/` | 持久化存储 DB、样本、Checkpoint 与导出包 |
| ├ `master_cn.db` | `$DRIVE_ROOT/master_cn.db` | 离线锁定国服主库（`BossVersionCN=202611131613`，见 `configs/sim_profile_highscore_cn.json`）；**严禁入库 Git** |
| ├ `cache/` | `$DRIVE_ROOT/cache/` | 缓存 `PCRCalculator-2.11.0-server-linux.zip`。各端口使用本地 `/tmp/pcr_calc_persistent_<port>` |
| ├ `cache/wheelhouse/` | `$CACHE_DIR/wheelhouse/` | 跨运行时复用的 Python ABI 兼容 wheels |
| ├ `cache/persistent_offline_202611131613.tar` | `$CACHE_DIR/persistent_offline_202611131613.tar` | 模拟器持久缓存的未压缩归档，随同名 `.sha256` 校验后解包至本地 SSD |
| ├ `data/` | `$DRIVE_ROOT/data/` | `feature_pack.json`、`round0_pilot.jsonl` 及 `round0_pilot.part-*.jsonl` |
| ├ `runs/` | `$DRIVE_ROOT/runs/ablation_round0/` | `<mode>/seed_<seed>/{last.pt,best.pt,test_metrics.json}`、`ablation_summary.{json,md}` |
| ├ `export_bundle/` | `$DRIVE_ROOT/export_bundle/<winning_mode>/` | `tower_a.onnx`、`tower_b.onnx`、`manifest.json`、`golden_io.json` |
| ├ `status/latest_receipt.json` | `$STATUS_DIR/latest_receipt.json` | 唯一机器可读回执，`status/` 同时保存 S0 门禁报告 |
| └ `replays.jsonl` | `$DRIVE_ROOT/replays.jsonl` | （可选）真实开局地图录制文件，用于 Path B 回放对照 |

建议在 Colab Shell / Notebook 中统一导出环境变量：

```bash
export DRIVE_ROOT="/content/drive/MyDrive/LimingjieNN"
export CACHE_DIR="$DRIVE_ROOT/cache"
export DATA_DIR="$DRIVE_ROOT/data"
export RUNS_DIR="$DRIVE_ROOT/runs"
export EXPORT_DIR="$DRIVE_ROOT/export_bundle"
export STATUS_DIR="$DRIVE_ROOT/status"
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

在当前运行时进入锁定提交的本地工作副本。Notebook Cell 1 自动选择代码来源并导出第 1 节的环境变量。

```bash
export PYTHONPATH="$PWD/src"
export CN_DB_PATH="$DRIVE_ROOT/master_cn.db"
mkdir -p "$STATUS_DIR"
bash scripts/colab_install_deps.sh --repo-dir "$PWD" --wheelhouse "$CACHE_DIR/wheelhouse"
python -m unittest discover -s tests -v

mapfile -t pilot_shards < <(
  find "$DATA_DIR" -maxdepth 1 -type f -name 'round0_pilot*.jsonl' |
  grep -E '/round0_pilot(\.part-[0-9]{5,})?\.jsonl$' | sort
)
python -m labyrinth_nn.data.s0_preburn verify-data \
  --db "$CN_DB_PATH" --catalog "$DATA_DIR/feature_pack.json" \
  --samples "${pilot_shards[@]}" --expected-shards 68 --expected-unique 25000 \
  --min-nonwin-audit 200 --out "$STATUS_DIR/s0_verify_data.json"
```

**验收门禁**：单元测试全部通过且跳过数为 0，包含真实国服库和 ONNX Runtime 对齐测试。审计核对锁库、特征目录、68 个 Pilot 分片、25,000 个唯一 ID 及至少 200 个非胜样例的伤害标签算术。若已有锁定摘要，给 `verify-data` 追加 `--db-sha256` 和 `--catalog-sha256`。打印完整报告供巡检。

---

### Phase 1 · 特征目录与模拟器口径对账

1. 将持久缓存归档打包至 Drive，再一次性解包到本地 SSD。已经有归档时复用原件，锁定的 `master_cn.db` 始终覆盖归档中的同名 DB。

```bash
persistent_tar="$CACHE_DIR/persistent_offline_${CN_DB_VERSION}.tar"
if [[ ! -f "$persistent_tar" ]]; then
  bash scripts/pack_persistent_cache.sh --persistent-dir "$CACHE_DIR/persistent" --out "$persistent_tar"
fi
bash scripts/colab_setup_simulator.sh \
  --cache-dir "$CACHE_DIR" --persistent-tar "$persistent_tar" \
  --db-path "$CN_DB_PATH" --db-version "$CN_DB_VERSION" --port 8080
```

2. 首次缺少特征目录时，从国服锁库提取。进入 S1 前核对目录并冻结该文件。

```bash
if [[ ! -f "$DATA_DIR/feature_pack.json" ]]; then
  python -m labyrinth_nn.catalog.build_catalog --db "$CN_DB_PATH" --out "$DATA_DIR/feature_pack.json"
fi
```

3. 执行 16 项国服口径对账与 64 子局吞吐探针。以下为 Round 0 复现示例，S0 / Round 1 改用 `configs/sim_profile_round1_cn.json` 和 `probe_round1.json`。

```bash
python -m labyrinth_nn.data.probe_simulator \
  --catalog "$DATA_DIR/feature_pack.json" \
  --profile configs/sim_profile_highscore_cn.json \
  --url http://127.0.0.1:8080 --bench-count 64 > "$DATA_DIR/probe_round0.json"
```

**验收门禁**：目录包含 `db_version`、`catalog_sha256`、`units`、`enemies`、`quests`、`relics`。探针退出码为 0，JSON 的 `passed` 为 `true`，16 项全部通过并记录 `benchmark.sims_per_sec`。

---

### Phase 2 · Round 0 分层自适应采样复现（2k 烟测 → 25k Pilot）

先用完整配置做 2,000 条分布干跑。真实烟测先设 `--target-battles 2000`，验收后调到 25,000。先按 Notebook Cell 5 把全部已有分片、manifest 和失败日志恢复至本地，再运行采样命令。Round 1 用第 5 节的 S1 命令。

```bash
python -m labyrinth_nn.data.sampler \
  --catalog "$DATA_DIR/feature_pack.json" \
  --profile configs/sim_profile_highscore_cn.json --sampling configs/sampling_round0.json \
  --dry-run 2000

python -m labyrinth_nn.data.sampler \
  --catalog "$DATA_DIR/feature_pack.json" \
  --profile configs/sim_profile_highscore_cn.json --sampling configs/sampling_round0.json \
  --url http://127.0.0.1:8080 --out /tmp/limingjienn-samples/round0_pilot.jsonl \
  --target-battles 25000 --batch-size 16 --concurrency 1 --timeout-sec 120 \
  --sync-dir "$DATA_DIR" --max-seconds 10800 --progress --isolate-errors --resume
```

**验收门禁**：核对每段落盘的主文件与全部 `.part-NNNNN.jsonl`、manifest 哈希和失败日志。计划内分段可能在达到目标前正常退出，仍需在下一段接力。只有 `total_battles` 达到所选目标时采样阶段才完成。

---

### Phase 3 · 四档消融训练与独立温度校准（`m0 -> m1 -> m2a -> m2b`）

用 Notebook Cell 6 校验完整训练分片和各 manifest SHA256 后运行。续训重复原命令，保持同一训练输入、配置、种子列表、`--split-seed` 和输出目录。胜出配置的多种子复验需要保留原有消融汇总时，使用独立输出目录。

```bash
python -m labyrinth_nn.train.train_critic \
  --configs m0 m1 m2a m2b --seeds 42 43 --split-seed 42 \
  --group-by context --auto-resume --amp --device cuda --pin-memory \
  --catalog "$DATA_DIR/feature_pack.json" \
  --samples "$DATA_DIR"/round0_pilot.jsonl "$DATA_DIR"/round0_pilot.part-*.jsonl \
  --out-dir "$RUNS_DIR/ablation_round0"
```

**验收门禁**：

- 每个 `<mode>/seed_<seed>/` 生成 `last.pt`、`best.pt`、`test_metrics.json`，根目录生成 `ablation_summary.md` 与 `ablation_summary.json`。
- 核验 `global`、六个分桶（`normal`、`robot_ex`、`special_dual_ex`、`other_ex`、`boss_3f_d5`、`boss_5f_d5`）和 `gold_unknown` 切片上的 `auroc`、`brier_3class`、`brier_win`、`ece_15`、`accuracy_3class`、`loss_damage_mae`、`win_time_mae`、`group_top1_fpr`、`group_top1_pos_fpr`。

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

当 `$DRIVE_ROOT/replays.jsonl` 就绪时，可参照 [notebooks/colab_labyrinth_pipeline.ipynb](notebooks/colab_labyrinth_pipeline.ipynb) Cell 8 执行策略对照与封存评估：

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

1. **前置资源**：核对 `$DRIVE_ROOT/master_cn.db`、服务端 ZIP、wheelhouse 与持久缓存 tar 的 SHA256。各端口启动后，核对本地工作目录及 `/tmp/pcr_calc_persistent_<port>/prod202611131613.db`。
2. **目录阶段**：`$DATA_DIR/feature_pack.json` 是否存在且与 `master_cn.db` 版本一致。
3. **采样阶段**：按轮次核对 manifest、分片行数和 SHA256；对比唯一成功样本数与该阶段目标。
4. **训练阶段**：检查 `$RUNS_DIR/ablation_round0/{m0,m1,m2a,m2b}/seed_*/{last.pt,best.pt,test_metrics.json}` 及 `ablation_summary.md`；读取已完成的 `test_metrics.json` 汇报当前最佳验证 NLL 与测试切片表现。
5. **导出阶段**：检查 `$EXPORT_DIR/*/{tower_a.onnx,tower_b.onnx,manifest.json,golden_io.json}`，核验 `manifest.json` 的 `validation.max_abs_diff` 及文件总大小。
6. **状态回执**：检查 `$STATUS_DIR/latest_receipt.json` 及本阶段门禁报告，核对锁定提交、已同步产物、实测花费、余额和阻塞。

### 4.2 自动恢复规则

- **计划内接力**：单段由 `--max-seconds 10800` 控制，新分片通过 `--sync-dir "$DATA_DIR"` 即时原子发布。核对 Drive 上的全部分片及 manifest 后，用 `--stop-only` 停服并释放运行时。计划接力不计入故障恢复次数。
- **采样异常恢复**：超时后先停服，检查模拟器日志和已落盘分片。按 Cell 5 校验和恢复主文件、分片、隔离失败日志及 manifest，仍用相同 `--out`、`--seed`、`--round-id` 和 `--resume`。禁止自动重发超时请求。异常恢复最多三次。
- **训练中断恢复**：使用 `--auto-resume --split-seed 42` 和原有配置、种子列表、完整训练分片及输出目录，按各配置和种子的 `last.pt` 续训。

```bash
bash scripts/colab_setup_simulator.sh --port 8080 --stop-only
python -m labyrinth_nn.data.s0_preburn update-receipt \
  --receipt "$STATUS_DIR/latest_receipt.json" --locked-commit "$LOCKED_COMMIT" \
  --current-stage S0 --stage-name S0 \
  --artifact-key s0_cpu_cost --artifact-value "$STATUS_DIR/s0_cpu_cost.json" \
  --note "已同步 CPU 成本门禁报告"
```

按实际情况填写 `--stage-status`、`--stage-cc`、`--observed-balance-cc`，记录 `--blocker` 或在解除后传 `--clear-blocker`。Notebook Cell 9 更新回执并调用 `runtime.unassign()`，随后核对活跃会话为 0。

### 4.3 立即停止并上报条件（禁止盲目循环重试）

出现以下任一异常时，监督 Agent 必须停止当前步骤并向用户告警：
1. `probe_simulator` 检查失败（`"passed": false`）或 `PCRCalculatorServer` 进程反复退出（附带 `/tmp/pcr_calc_server_<port>.log` 末尾日志）。
2. 训练出现非有限数值异常（`non-finite training loss` / `non-finite selection NLL`）或连续未完成且无法通过 `--auto-resume` 推进。
3. 消融结论异常：例如 `m2b` 在 `gold_unknown` 或 `boss_5f_d5` 等核心切片上明显劣于 `m0` 基线。
4. ONNX 导出数值对齐失败（`max_abs_diff >= 1e-4`）或双塔模型合计达到或超过 `10,000,000` 字节。

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

### S0 收尾参考命令（先 CPU，再切换 T4）

先完成第 3 节的环境、数据审计和 64 子局探针。CPU Runtime 热身 8 行后测至少 200 行，再停服并释放该会话。T4 Runtime 重新安装或复用 wheelhouse 后，对第 3 节收集的同一组 Pilot 分片跑两轮短测。命令中的 0.08 与 1.17 为参考小时费率，执行时使用实测速率。保存并打印完整 JSON，核对预计 S1 不超过 2.5 CC、S2 不超过 3.5 CC，再更新回执。

```bash
# CPU Runtime
python -m labyrinth_nn.data.s0_preburn cpu-cost \
  --catalog "$DATA_DIR/feature_pack.json" --profile configs/sim_profile_round1_cn.json \
  --sampling configs/sampling_round1.json --url http://127.0.0.1:8080 \
  --probe-bench-count 64 --probe-out "$DATA_DIR/probe_round1.json" \
  --warmup-rows 8 --rows 200 --batch-size 8 --concurrency 1 --timeout-sec 120 \
  --sim-log /tmp/pcr_calc_server_8080.log --cpu-cc-per-hour 0.08 \
  --out "$STATUS_DIR/s0_cpu_cost.json"
cat "$STATUS_DIR/s0_cpu_cost.json"
bash scripts/colab_setup_simulator.sh --port 8080 --stop-only

# 切换至 T4 GPU Runtime 后，重跑 Cell 1 并按 Phase 0 重建 pilot_shards 数组。
python -m labyrinth_nn.data.s0_preburn t4-timing \
  --catalog "$DATA_DIR/feature_pack.json" --config configs/critic_m2b_r1.json \
  --samples "${pilot_shards[@]}" --epochs 2 --device cuda --amp --pin-memory \
  --group-by context --t4-cc-per-hour 1.17 --out "$STATUS_DIR/s0_t4_timing.json"
cat "$STATUS_DIR/s0_t4_timing.json"
```

### S1 / S2 执行参考命令（CPU 采样与 T4 选型解耦）

同一时间只保留一个 Colab 运行时。不同采样工作者可分段顺序运行，继续使用各自固定的标识和输出前缀。

```bash
# S1 CPU Runtime。先按 Cell 5 校验并恢复已有分片、manifest 和失败日志。
python -m labyrinth_nn.data.sampler \
  --catalog "$DATA_DIR/feature_pack.json" \
  --profile configs/sim_profile_round1_cn.json --sampling configs/sampling_round1.json \
  --url http://127.0.0.1:8080 --out /tmp/limingjienn-samples/round1_scale.jsonl \
  --target-battles 125000 --batch-size 16 --concurrency 1 --timeout-sec 120 \
  --sync-dir "$DATA_DIR" --max-seconds 10800 --progress --isolate-errors --resume
bash scripts/colab_setup_simulator.sh --port 8080 --stop-only

# S2 T4 GPU Runtime。先用 Cell 6 校验每个 TRAIN_PREFIXES 条目的 manifest 和 SHA256。
python -m labyrinth_nn.train.train_critic \
  --config-files configs/critic_m1.json configs/critic_m2b.json configs/critic_m2b_r1.json configs/critic_m2b_r1_dmg025.json \
  --seeds 42 43 --split-seed 42 --group-by context --auto-resume --amp --device cuda --pin-memory \
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
