# LimingjieNN 云端自动监督规则

本规则适用于后续启用 [TASK.md](../TASK.md) 中预留 **20 CC** 执行云端采样、训练与导出时的监督调度。

1. **任务源与冻结**：以锁定提交的 `TASK.md` 与 `notebooks/colab_labyrinth_pipeline.ipynb` 为准；正式采样启动前确认修复提交号，启动后同一轮冻结代码提交与 `feature_pack.json`。
2. **单运行时与缓存纪律**：同一时间最多保留 1 个 Colab 运行时；严禁对高频 DB / `Pool/` 缓存使用 Drive FUSE 软链接，统一使用 `scripts/pack_persistent_cache.sh` 打包 tar 校验后由 `scripts/colab_setup_simulator.sh` 解包至 `/tmp` 本地 SSD，Python 依赖经 `scripts/colab_install_deps.sh` 复用 `cache/wheelhouse/`。
3. **分段接力与异常恢复**：计划内采样每段 3–4 小时（`--max-seconds 10800 --sync-dir "$DATA_DIR"` 原子落盘），收尾停服并确认活跃会话归 0（不计入异常配额）。遇 `SimulationTimeoutError` **禁止自动重发或二分拆批**，先 `--stop-only` 停服核对分片与日志后再 `--resume`；异常恢复全轮最多 3 次，训练中断使用 `--auto-resume --split-seed 42`。
4. **门禁放行与停机红线**：按 `TASK.md` 的 Go / No-Go 门槛与 4 条立即停机红线执行，严禁盲目重试。
5. **汇报与唯一回执**：通过 `python -m labyrinth_nn.data.s0_preburn update-receipt`（Notebook Cell 9）维护唯一的 `$DRIVE_ROOT/status/latest_receipt.json`，记录锁定提交、阶段状态、实耗/余额、产物路径与阻塞；消息汇报均引用同一份回执。
