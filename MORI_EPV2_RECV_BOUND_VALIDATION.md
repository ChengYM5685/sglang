# MORI EPV2 Receive-View Trimming Validation

验证时间：2026-09-22 至 2026-09-23

## 1. 结论

MORI EPV2 的 receive-view trimming 已完成实现和分层验证。对于已经证明接收布局会按
`(source token, destination rank)` 去重、并将有效接收行紧密排列在前缀中的单节点
backend，AITER 的逻辑输入行数可以安全地从：

```text
next_power_of_two(sum(sender_rows) * router_topk)
```

缩减为：

```text
max(32, next_power_of_two(sum(sender_rows)))
```

物理 MORI arena 不变，本修改只缩短交给 AITER 的逻辑 tensor view。

当前正式开启的范围：

- EPV2 FlyDSL intranode；
- EPV2 HIP intranode；
- `TP=attention-DP=EP`、`MoE TP=1`、`MoE DP=1`；
- TBO off；
- sender metadata 完整、非负，并与当前 local dispatch 行数一致；
- eager 与 CUDA graph capture/replay；
- FP4 和 BF16 transport；
- direct output on/off。

`SGLANG_MORI_RECV_BOUND` 的默认值仍为 `false`，需要显式设置为 `1` 才会启用裁剪。

## 2. 版本

```text
SGLang base: ee9bc41817a563b1f21011c8bd8658bf5bfee620
Receive-bound implementation: 7755f0ef8a60a8a20a8ddc84d758f5e7556c0022
HIP intranode extension: 0cc063f0d664e8cf81a08fce88939b52f45e11e9
Branch: mori-epv2-recv-bound-validation-20260923

MORI:  05fd0b03ba9ddfc271f7005b2f2b993788356e66
AITER: 49c6fdd4558a45b2cd34e713b74d0280a2a61c88
ROCm:  7.2.0
GPU:   8 x AMD Instinct MI355X, gfx950
```

基础镜像：

```text
chengyueming/test:test_epv2_0922
sha256:78f563de6b11e17b23c145b8076d4c6a0f5d2ac177170a11d0cd1d3d26f35531
```

本地候选服务镜像：

```text
yuecheng/sglang-mori-epv2:ee9bc418-recvbound-candidate-20260922
sha256:aaad4b2df889120e75e30e3fe2bd8555ad1378e45d7bb80835bfdaa7154ded07
```

该候选镜像包含 FlyDSL 功能修改，但不包含最终的全局日志去重及 HIP intranode 开放；
正式发布镜像应从本分支重新构建。

## 3. 实现

主要实现文件：

```text
python/sglang/srt/layers/moe/token_dispatcher/moriepv2.py
```

关键行为：

1. 在 `dispatch_a()` 中快照本次 dispatch 对应的 CPU sender metadata，避免
   `dispatch_a()` 与 `dispatch_b()` 分离时读到下一批次或另一个 TBO child 的状态。
2. 仅对已经验证的 intranode FlyDSL/HIP backend 使用去 top-k 上界。
3. 保留 32 行最小逻辑视图和原有物理容量。
4. 对不满足条件的配置返回完整物理视图，并记录明确原因。
5. 日志按 `(reason, logical_rows)` 全局去重，避免每个 MoE layer 重复输出。

回退原因包括：

```text
disabled
tbo_metadata_missing
layout_unverified
sender_mapping_unknown
metadata_missing
metadata_invalid
metadata_mismatch
capacity_unproven
no_saving
```

`capacity_unproven` 仅表示关闭逻辑裁剪，不表示物理 arena 一定足够。

## 4. FP4+AITER 对照结果

“Conservative” 是使用相同 metadata gate、但保留 `sum * topk` 的受控对照。
对于 eager，它不是未修改 `ee9bc418` 的原始行为；原始 eager 通常直接回退完整视图。

| Backend/场景 | Conservative/full 行数 | Candidate 行数 | 基线峰值额外显存 | Candidate 峰值额外显存 | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| FlyDSL EP2 eager，`[56] * 2` | 1024 | 128 | 87.23 MiB | 7.02 MiB | 5.49e-4 |
| FlyDSL EP2 graph，20 replay | 1024 | 128 | 未单独采样 | 7.02 MiB | 5.49e-4 |
| FlyDSL EP4 eager，`[56] * 4` | 2048 | 256 | 204.72 MiB | 27.02 MiB | 1.22e-4 |
| FlyDSL EP4 graph，20 replay | 2048 | 256 | 未单独采样 | 27.02 MiB | 1.22e-4 |
| FlyDSL EP8 eager BF16 transport，`[56] * 8` | 4096 | 512 | 295.52 MiB | 40.05 MiB | 1.22e-4 |
| FlyDSL EP8 graph BF16 transport，20 replay | 4096 | 512 | 未单独采样 | 41.05 MiB | 1.22e-4 |
| HIP EP2 eager FP4，`[56] * 2` | full 16384 | 128 | 1615.37 MiB | 8.77 MiB | 5.49e-4 |
| HIP EP2 graph FP4，20 replay | full 16384 | 128 | 未单独采样 | 8.77 MiB | 5.49e-4 |
| HIP EP4 eager FP4，`[56] * 4` | full 32768 | 256 | 3230.74 MiB | 27.02 MiB | 1.22e-4 |
| HIP EP8 eager FP4，`[56] * 8` | full 65536 | 512 | 6461.48 MiB | 47.05 MiB | 1.22e-4 |
| HIP EP8 graph FP4，20 replay | full 65536 | 512 | 未单独采样 | 55.05 MiB | 1.22e-4 |
| HIP EP8 eager nonuniform | full 65536 | 1024 | 6461.48 MiB | 104.17 MiB | 0 |

EP8 nonuniform sender vector：

```text
[0, 1, 7, 33, 56, 128, 257, 448]
```

HIP EP8 最坏 fan-in 结构验证中，rank 0 实收 448 行，其他 rank 实收 0 行，
512 行逻辑 bound 足够；reverse-source map、dense prefix、hidden、全部 top-k IDs
和 weights 均匹配独立期望。

## 5. TBO

生产配置继续安全回退：

```text
recv_caps=[65536, 65536]
bound_reasons=[tbo_metadata_missing, tbo_metadata_missing]
```

为判断是否有进一步优化空间，测试中为两个 child 分别注入与当前 dispatch 绑定的
sender metadata。两个 child 的 sender sums 分别为 35 和 34，均成功使用 64 行
逻辑视图，FP4 dispatch/combine 数值通过：

```text
recv_caps=[64, 64]
bound_reasons=[trimmed_dedup, trimmed_dedup]
```

结论：TBO 的数据布局允许进一步裁剪，但生产代码还不能开启。需要先将每个 child
的完整 per-rank sender vector 接入调度路径，并验证真实 AITER、graph 和交错执行。

## 6. 已跳过的原有失败

EP4 非均匀 sender 且包含 idle rank：

```text
[0, 1, 7, 56]
```

在裁剪开启和完全关闭的情况下均在 combine 后触发 GPU memory access fault；
direct output on/off 也都能复现。因此它不是 receive-view trimming 引入的回归，
本轮按原有 MORI EP4 限制记录并跳过，不声明该组合受支持。

## 7. DeepSeek-V4 Pro 短服务烟测

```text
配置: 8 x MI355X, TP8/DPA8/EP8, FP4, c32, 300 秒
mem_fraction_static: 0.92
HiCache ratio: 1.0
TBO: off
direct output: on
acceptance: simulated
```

- target verify graph：约 3.00 GB/rank；
- draft verify graph：约 0.21 GB/rank；
- 211 个成功请求，0 个请求错误，2 个 grace-window cancellation；
- 总吞吐 63,704.72 tok/s；
- TTFT P90 2.783 s；
- ITL P90 25.32 ms；
- E2E P90 16.444 s；
- 容器 exit 0，`OOMKilled=false`。

结果目录：

```text
/mnt/m2m_nobackup/yuecheng/mori-epv2-v0520-20260922/results/epv2-recvbound-candidate-short-20260922T133900Z
```

该结果是服务健康烟测，不是正式性能 A/B；未运行真实 GSM8K 准确率。

## 8. 当前支持边界

| 场景 | 状态 |
|---|---|
| FlyDSL intranode，EP2/EP4/EP8，均匀 sender | `DEDUP_TRIM_VERIFIED` |
| FlyDSL EP8 nonuniform/idle sender | `DEDUP_TRIM_VERIFIED` |
| FlyDSL EP2 idle sender | `DEDUP_TRIM_VERIFIED` |
| HIP intranode，EP2/EP4/EP8，FP4+AITER | `DEDUP_TRIM_VERIFIED` |
| HIP EP8 nonuniform/idle sender | `DEDUP_TRIM_VERIFIED` |
| FP4/BF16 transport，eager/graph | `DEDUP_TRIM_VERIFIED` |
| TBO 正常配置 | `FALLBACK_VERIFIED` |
| TBO 注入准确 per-child metadata | `PROBE_PASS`，尚未接入生产 |
| EP4 nonuniform/idle sender | `BASELINE_UNSUPPORTED` |
| HIP internode、多节点 | `NOT_RUN`，保持回退 |
| TP/DPA/EP 非一一映射 | `NOT_RUN`，保持回退 |
| EPV1 normal/AsyncLL | `NOT_RUN`，代码未修改 |
| c256、长稳、真实 GSM8K | `NOT_RUN` |

## 9. 检查

```text
44 passed: test_mori_epv2_fp4_tbo.py + test_aiter_runner.py
TBO fallback real 8-rank test: PASS
TBO per-child trimming probe: PASS
pre-commit: PASS
git diff --check: PASS
FP4 indexer cache checker: FIXED
```

原始结果保存在：

```text
/mnt/m2m_nobackup/yuecheng/mori-recv-bound-validation-20260922/cases/
```

## 10. 下次接手入口

远端仓库与分支：

```text
origin: https://github.com/ChengYM5685/sglang.git
branch: mori-epv2-recv-bound-validation-20260923
```

在当前机器继续：

```bash
cd /mnt/m2m_nobackup/yuecheng/mori-recv-bound-validation-20260922/sglang
git switch mori-epv2-recv-bound-validation-20260923
git pull --ff-only
git status --short --branch
```

开始新工作前应确认：

```bash
git rev-parse HEAD
git -C /mnt/m2m_nobackup/yuecheng/mori-epv2-v0520-20260922/mori rev-parse HEAD
git -C /mnt/m2m_nobackup/yuecheng/mori-epv2-v0520-20260922/aiter rev-parse HEAD
docker image inspect chengyueming/test:test_epv2_0922
```

本机的完整实验材料没有全部提交到 Git；继续验证时应同时保留：

```text
/mnt/m2m_nobackup/yuecheng/mori-recv-bound-validation-20260922/scripts/
/mnt/m2m_nobackup/yuecheng/mori-recv-bound-validation-20260922/cases/
/mnt/m2m_nobackup/yuecheng/mori-recv-bound-validation-20260922/RUN_SUMMARY.md
/home/yuecheng/MORI_EPV2_HANDOFF.md
```

其中：

- `scripts/run_operator_case.sh`：小 hidden 的真实 MORI 布局与 combine 检查；
- `scripts/run_aiter_case.sh`：真实 DSV4 FP4 权重和 AITER fused-MoE 检查；
- `scripts/test_mori_epv2_recv_bound.py`：独立构造 expected routing/output；
- `scripts/measure_epv2_aiter.py`：full/conservative/candidate 显存和数值对照；
- `cases/<case_id>/run.log`：完整运行日志；
- `cases/<case_id>/result.json`：各 rank 的结构、误差和显存结果。

## 11. 复现命令

以下命令从实验目录执行：

```bash
cd /mnt/m2m_nobackup/yuecheng/mori-recv-bound-validation-20260922
```

基础单测：

```bash
docker start AIAT_yuecheng_epv2_recvbound_validation
docker exec AIAT_yuecheng_epv2_recvbound_validation bash -lc \
  'cd /workspace/sglang && python -m pytest -q \
   test/registered/unit/layers/moe/test_mori_epv2_fp4_tbo.py \
   test/registered/unit/layers/moe/test_aiter_runner.py'
```

EP2 FlyDSL 最坏 fan-in：

```bash
WORLD_SIZE=2 \
CASE_ID=ep2_eager_t56_fanin_bf16_struct \
SENDER_ROWS=56 \
ROUTE_PATTERN=fanin \
EXECUTION=eager \
DIRECT_OUTPUT=1 \
scripts/run_operator_case.sh
```

EP4 FlyDSL graph：

```bash
WORLD_SIZE=4 \
CASE_ID=ep4_graph_uniform_bf16_candidate_direct0 \
SENDER_ROWS=56 \
ROUTE_PATTERN=random \
EXECUTION=graph \
GRAPH_REPLAYS=20 \
DIRECT_OUTPUT=0 \
scripts/run_operator_case.sh
```

EP8 BF16 transport + AITER：

```bash
IMAGE=yuecheng/sglang-mori-epv2:ee9bc418-recvbound-candidate-20260922 \
WORLD_SIZE=8 \
CASE_ID=aiter_ep8_eager_t56_bf16_candidate \
SENDER_ROWS=56 \
POLICY=candidate \
EXECUTION=eager \
DIRECT_OUTPUT=1 \
DISPATCH_DTYPE=bf16 \
COMPARE_DIR=/validation/cases/aiter_ep8_eager_t56_bf16_conservative \
scripts/run_aiter_case.sh
```

HIP intranode 正式 gate：

```bash
IMAGE=yuecheng/sglang-mori-epv2:ee9bc418-recvbound-candidate-20260922 \
WORLD_SIZE=8 \
MORI_BACKEND=hip \
CASE_ID=aiter_ep8_hip_eager_t56_candidate \
SENDER_ROWS=56 \
POLICY=candidate \
EXECUTION=eager \
DIRECT_OUTPUT=1 \
COMPARE_DIR=/validation/cases/aiter_ep8_hip_eager_t56_full \
scripts/run_aiter_case.sh
```

注意：候选镜像里的 SGLang Python 源码会被只读 bind mount 的当前工作树覆盖；
因此运行日志中的 `moriepv2_source` 必须是 `/workspace/sglang/.../moriepv2.py`。

## 12. 已知失败与诊断边界

### EP4 nonuniform/idle

以下三个 case 均在 combine 后出现 GPU memory access fault：

```text
cases/ep4_eager_nonuniform_bf16_struct/
cases/ep4_eager_nonuniform_bf16_full_direct0/
cases/ep4_eager_nonuniform_bf16_full_direct1/
```

因为 `RECV_BOUND=0` 的完整视图也失败，所以不能归因于本次裁剪。不要通过增加
receive bound、放宽误差阈值或重复运行来掩盖该问题；后续应在 MORI EP4 combine
路径单独建立最小复现。

### TBO

结果文件：

```text
cases/tbo_fallback_verified/result.json
cases/tbo_trim_probe/result.json
```

生产路径结果：两个 child 均为 65536 行，原因均为
`tbo_metadata_missing`。验证性探针在每个 `dispatch_a()` 前发布对应 child 的完整
sender vector，并只在测试对象上绕过 TBO gate；两个 child 的总行数分别为 35 和
34，均使用 64 行并通过。

这证明 TBO 的布局可以裁剪，但不证明当前 scheduler metadata 生命周期已经满足
生产要求。正式实现前需要：

1. 在 `split_forward_batch()` 中保存每个 child 的 per-rank sender counts；
2. 确保 `_StageExecutor.next()` 为正在执行的 child 发布对应 counts；
3. 验证 unequal split、空 child、交错 dispatch/combine、graph replay；
4. 再移除 `tbo_metadata_missing` gate。

### HIP internode 和其他拓扑

HIP intranode 已开放，但 `cfg.is_internode=True` 仍返回 `layout_unverified`。跨节点
需要验证节点内/节点间两阶段 dispatch 的最终行分配和 sender 上界。`TP/DPA/EP`
不相等时，当前 DP counts 不能直接解释为 EP sender counts，也继续回退。

## 13. 后续优先级

建议从以下顺序继续，避免重跑已经有结论的矩阵：

1. **TBO metadata 接线**：这是当前最明确、探针已证明有收益的下一项；完成后先跑
   `test_mori_epv2_tbo_dispatcher.py` 的正常模式，不再依赖
   `PROBE_TBO_TRIM=1`。
2. **EP4 nonuniform 原生故障**：先在 `RECV_BOUND=0` 下修复 MORI combine，再恢复
   candidate 测试；当前不是 SGLang bound 公式问题。
3. **HIP 服务烟测**：用最终提交重新构建镜像，再跑短 DSV4 服务。算子级已经通过，
   但尚未做完整模型启动验证。
4. **目标负载**：c256 / 448 verify-token 档位、3600 秒长稳、真实 DSpark acceptance
   和 GSM8K 1319 样本。
5. **其他拓扑/多节点**：先建立 sender mapping，不要直接删除现有 fallback gate。
6. **EPV1**：作为独立任务验证 normal、AsyncLL、SDMA 和 dtype 组合，本分支未修改
   EPV1。

不要重复使用 2026-09-22 13:07-13:35 左右的 synthetic identity-combine 失败
作为结论；那些实验绕过了真实 AITER expert path，且部分时间与其他 GPU 容器重叠。
权威结果应以本文件列出的 `result.json` 和最终服务目录为准。
