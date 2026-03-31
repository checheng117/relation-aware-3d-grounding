# 官方 Full-Train 执行计划

## 目标与优先级

| 优先级 | 内容 | 是否本阶段必选 |
|--------|------|----------------|
| **P1** | **B** raw-text relation（entity + full） | **必选** |
| **P1** | **C** structured relation-aware（entity + full） | **必选** |
| **P2** | **A** attribute-only（entity + full） | 可选；`--scope bca` |
| **P2** | **full A** 单独为 shortlist 提供 `np_f_attr` | **`--scope bc` 时自动加入**（1 个额外训练任务） |
| **P3** | **D** hard-negative | 可选；`--scope all`，且应在 B/C 完成后跑 |

## 实际命令（仓库根目录）

### 官方主跑（推荐：先 B/C + shortlist 用 full 属性粗模型）

```bash
# 默认：CUDA 若可用则用 GPU；诊断 YAML 完整 epoch；无 debug_max_batches
python scripts/run_next_phase_pipeline.py \
  --official-full-train \
  --scope bc \
  --seeds 42 \
  --device auto

# 输出目录形如：
# outputs/YYYYMMDD_HHMMSS_full_train_official/
```

**多种子 B/C（统计稳定性，推荐终稿引用）**：

```bash
python scripts/run_next_phase_pipeline.py \
  --official-full-train \
  --scope bc \
  --seeds 42,43,44 \
  --device auto
```

- **示例已完成目录**：`outputs/20260327_135641_full_train_official/`（`main_table_official` 为 mean±std；后处理见该目录 `repro_commands.sh` 末行）
- **单种子对照**：`outputs/20260326_095106_full_train_official/`

### 含 A 的完整三线（无 D）

```bash
python scripts/run_next_phase_pipeline.py \
  --official-full-train \
  --scope bca \
  --seeds 42 \
  --device auto
```

### 含 D 的完整矩阵

```bash
python scripts/run_next_phase_pipeline.py \
  --official-full-train \
  --scope all \
  --seeds 42 \
  --device auto
```

### 多种子（B/C 主指标聚合 mean±std；分层仍以第一粒种子为主）

```bash
python scripts/run_next_phase_pipeline.py \
  --official-full-train \
  --scope bc \
  --seeds 42,43,44 \
  --device auto
```

### 仅复用已有 checkpoint 重新出表/图

```bash
python scripts/run_next_phase_pipeline.py \
  --official-full-train \
  --scope bc \
  --skip-train \
  --seeds 42 \
  --device auto
```

## 底层入口映射

| 步骤 | 脚本 / 模块 |
|------|----------------|
| 训练 B | `scripts/train_baseline.py` + `configs/train/diagnosis/diag_{entity,full}_raw_relation.yaml`（由生成 YAML 覆盖 `checkpoint_dir` / `run_name` / `seed`） |
| 训练 C | `scripts/train_main.py` + `diag_*_rel_structured.yaml` |
| 训练 A | `scripts/train_baseline.py` + `diag_*_baseline.yaml` |
| 训练 D | `scripts/train_main.py` + structured + `loss.hard_negative` |
| 受控评估 | `scripts/eval_all.py` + `configs/dataset/diagnosis_entity_geom.yaml`，`val` |
| 全场景评估 | `eval_all.py` + `diagnosis_full_geom.yaml` |
| Shortlist | `src/rag3d/evaluation/shortlist_bottleneck.py` + `np_f_attr_last.pt` + full val manifest |
| 几何 / 难例 | `stratified_eval` 切片 → `geometry_quality_results.json`、`hard_case_results.json` |
| 释义（便宜） | `scripts/eval_paraphrase_consistency.py` |
| 失败类型图 | `scripts/analyze_hard_cases.py` + `next_phase_plots.py` |

## 输出目录约定

所有产物在 **`outputs/<timestamp>_full_train_official/`**：

- `main_results.json`, `stratified_results.json`, `hard_case_results.json`
- `geometry_quality_results.json`, `geometry_quality_table.csv`
- `shortlist_diagnostics.json`, `shortlist_metrics.csv`, `shortlist_curve.png`, `shortlist_interpretation.md`
- `diagnostics_results.json`, `summary.csv`
- `main_table_official.csv`, `main_table_official.md`（含 **n_seeds**、**notes** 列）
- `main_table.csv`, `main_table.md`（与旧版兼容的简表）
- `relation_stratified_table.*`（有 rel 键时）
- `hard_case_table.csv`, `hard_case_table.md`
- `paraphrase_results.json`（若跑通）
- `repro_commands.sh`, `train_logs/`, `generated_configs/`
- `report_bundle/` + `README.md`

## 与 Smoke 的关系

- **不覆盖** `outputs/20260326_093621_next_phase/`。
- `shortlist_interpretation.md` 会引用该 smoke 目录（若存在）做 **recall/oracle** 对比。

## 最近一次已完成的官方跑

- **目录**：`outputs/20260326_095106_full_train_official/`
- **命令**：`python scripts/run_next_phase_pipeline.py --official-full-train --scope bc --seeds 42 --device auto`
- **书面结论**：`reports/full_train_official_summary.md`
- **图**：若根目录缺 `*.png`，可在该目录下用 `scripts/next_phase_plots.py` 的逻辑对 `diagnostics_results.json` 重绘（本机已补全并复制到 `report_bundle/`）。

## 强制 vs 可选

| 项目 | 强制 |
|------|------|
| B/C full-train + 双体制评估 | ✅（`--scope bc` 起） |
| Shortlist（recall@K + 可选 two-stage） | ✅（需 `np_f_attr`；bc 范围自动训 full A） |
| 难例 JSON + `hard_case_table.csv` | ✅ |
| 几何表（controlled + full 的 C） | ✅ |
| A/D | ❌ 按 scope |
| 3 种子 | ❌ 视算力；`--seeds 42` 为默认可完成证据 |
