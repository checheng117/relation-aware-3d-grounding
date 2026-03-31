# 下一阶段实验协议（固定）

## 数据集与划分

- **主实验**：ReferIt3D 风格 manifest（`prepare_data.py` 生成）。
- **验证划分**：`val`（`train_manifest.jsonl` / `val_manifest.jsonl` 由数据集配置指定）。
- **受控候选**：`configs/dataset/diagnosis_entity_geom.yaml` → `data/processed/diagnosis_entity_geom/val_manifest.jsonl`。
- **全场景候选**：`configs/dataset/diagnosis_full_geom.yaml` → `data/processed/diagnosis_full_geom/val_manifest.jsonl`。

## 评估体制

1. **受控候选集（controlled）**：在 entity-only manifest 上评估；训练使用对应 **entity** 诊断配置（与 `run_diagnosis_pass.py` 一致）。
2. **全场景（full-scene）**：在 full manifest 上评估；训练使用对应 **full** 诊断配置。

## 主指标

- **Acc@1**、**Acc@5**（全体）。
- **关系分层**：`stratified_results.json` 中 `acc@1_rel::<type>`（类型由数据 `relation_type_gold` / `relation_type` 决定）。
- **难例子集**：`acc@1_subset::*`（clutter、anchor、occlusion、parser、low margin 等）。

## 随机种子

- **调试 / smoke**：`seed=42`，训练 `epochs=1`、`debug_max_batches` 限制（见流水线 `--smoke`）。
- **正式报告**：若算力允许，对每条训练线使用 **3 个种子**（如 42/43/44），在 `summary.csv` 中报告 mean±std；默认流水线可先写单种子，通过重复运行并合并扩展。

## 标准输出模式（时间戳目录 `outputs/<UTC_ts>_next_phase/`）

| 文件 | 内容 |
|------|------|
| `main_results.json` | 按模型键聚合的 Acc@1/5/n（可多 regime 嵌套，由 `result_bundle` 约定） |
| `stratified_results.json` | 各模型分层指标 |
| `hard_case_results.json` | 难例子集指标子集 |
| `diagnostics_results.json` | shortlist / 几何 / 释义等诊断 |
| `summary.csv` | 扁平汇总，便于表格与画图 |
| `main_table.csv` / `main_table.md` | 主对比矩阵 |
| `repro_commands.sh` | 记录命令 |

## 模型矩阵（最小）

- **A**：attribute-only baseline  
- **B**：raw-text relation baseline  
- **C**：structured relation-aware（ours-base）  
- **D**：C + hard-negative training（`loss.hard_negative.enabled: true`）

每条线在 **受控** 与 **全场景** 上各训一版 checkpoint，并在 **对应 val manifest** 上评估（训练/评估体制匹配）。

## 不可静默修改的定义

- 不修改 `eval_all.py` 中 Acc@1/5 的计算方式。
- 分层键名沿用 `stratified_eval.stratified_accuracy_from_lists`。
