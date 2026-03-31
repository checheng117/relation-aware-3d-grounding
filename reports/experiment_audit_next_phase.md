# 下一阶段实验 — 仓库审计（Phase 0）

## 包根与安装

- **Python 包**：`src/rag3d/`（`pyproject.toml` 中 `name = relation-aware-3d-grounding`）
- **安装**：`pip install -e ".[dev,viz]"`（`viz` 含 matplotlib，用于出图）

## 已发现的可运行入口

| 脚本 | 作用 |
|------|------|
| `scripts/prepare_data.py` | `validate` / `build` / `build-nr3d-geom` / `mock-debug` |
| `scripts/train_baseline.py` | 训练 `attribute_only` 或 `raw_text_relation` |
| `scripts/train_main.py` | 训练 `relation_aware`（heuristic / structured parser） |
| `scripts/train_coarse_stage1.py` | 粗排 stage-1（attribute / coarse_geom） |
| `scripts/train_two_stage_rerank.py` | 两阶段 rerank |
| `scripts/eval_all.py` | 多模型 Acc@1/5 + `stratified_results.json` |
| `scripts/eval_stage1_recall_pass.py` | coarse recall@K + 可选 two-stage 行 |
| `scripts/eval_paraphrase_consistency.py` | 模板释义一致性 |
| `scripts/analyze_hard_cases.py` | 难例与标签统计 |
| `scripts/run_diagnosis_pass.py` | 8 格诊断：entity/full × baseline/raw/heuristic/structured |
| `scripts/collect_results.py` | 汇总 `main_results.json` → Markdown/CSV |
| `scripts/aggregate_full_gpu_pass_tables.py` | 表格聚合 |
| `scripts/build_report_bundle_blueprint.py` | 复制固定报告产物 |

## 配置（代表性）

- **数据集**：`configs/dataset/referit3d.yaml`，`diagnosis_entity_geom.yaml`（受控候选），`diagnosis_full_geom.yaml`（全场景）
- **训练**：`configs/train/baseline.yaml`，`raw_relation.yaml`，`main.yaml`；`configs/train/diagnosis/diag_*.yaml`
- **评估**：`configs/eval/default.yaml`，`debug.yaml`，`configs/eval/diagnosis/*.yaml`，`configs/eval/stage1_recall_pass.yaml`
- **模型**：`configs/model/{attribute_only,raw_text_relation,relation_aware}.yaml`

## 已有结果与约定

- **Checkpoint**：`outputs/checkpoints/`，`outputs/checkpoints_diagnosis/`，`outputs/checkpoints_stage1/`（后者可能为空）
- **指标 JSON**：`outputs/metrics/main_results.json`，`stratified_results.json`，`outputs/metrics/diagnosis/*.json`
- **图表**：`outputs/figures/`，`make figures`（见 Makefile）

## 基线与「我们的」方法（命名）

- **A** `attribute_only`（`train_baseline.py`，`baseline_line: attribute_only`）
- **B** `raw_text_relation`
- **C** `relation_aware` + **structured** parser（`parser_mode: structured`）
- **D** 同 C + **hard-negative**（`loss.hard_negative`，见 `configs/train/blueprint_loss_example.yaml`）

## 已实现的实验能力（与需求对照）

| 需求 | 现状 |
|------|------|
| 受控 vs 全场景 | ✅ diagnosis entity / full 双 manifest；`eval_all.py` |
| Acc@1 / Acc@5 | ✅ `eval_all.py` |
| 关系分层 | ✅ `stratified_eval.py`（`acc@1_rel::*`） |
| 难例子集 | ✅ tags：`same_class_clutter`、`anchor_confusion`、`occlusion_heavy`、`parser_failure`、`low_model_margin` |
| 两阶段 / shortlist | ✅ `two_stage_eval.py`、`coarse_recall.py`、`eval_stage1_recall_pass.py` |
| 几何质量切片 | ✅ `augment_meta_geometry_fallback_tags`（`geometry_high_fallback`、`real_box_heavy`、`weak_feature_source`） |
| 释义 | ✅ `eval_paraphrase_consistency.py` + `paraphrase_templates.py` |

## 缺口（本阶段最小补齐）

1. **统一时间戳实验目录**下写入 `main_results.json`、`stratified_results.json`、`hard_case_results.json`、`diagnostics_results.json`、`summary.csv` 的**编排脚本**（原工具多写到 `outputs/metrics` 全局路径）。
2. **主对比表**（A–D × 受控/全场景）的**单次聚合**与 **CSV/Markdown**。
3. **Recall@40** 与 **两阶段 oracle / 条件 rerank** 的轻量函数（在现有 `eval_coarse_stage1_metrics` / `eval_two_stage` 之上封装）。
4. **报告 bundle + repro_commands.sh + 书面总结** 的自动生成。

以上由 `scripts/run_next_phase_pipeline.py` 及 `src/rag3d/evaluation/result_bundle.py`、`shortlist_bottleneck.py` 等补齐。
