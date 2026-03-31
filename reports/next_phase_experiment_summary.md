# 下一阶段实验 — 书面总结

## 1. 已执行内容

- **审计与协议**：`reports/experiment_audit_next_phase.md`、`reports/experiment_protocol_next_phase.md`。
- **主流水线**：`scripts/run_next_phase_pipeline.py`（默认 **smoke**：每线 1 epoch、`debug_max_batches=64`、CPU 友好）。
- **本次已跑通的一次完整 smoke 输出目录**：`outputs/20260326_093621_next_phase/`（8 训 + 8 评 + shortlist/释义/难例/报告包）。

复现：

```bash
cd /path/to/relation-aware-3d-grounding
CUDA_VISIBLE_DEVICES= python scripts/run_next_phase_pipeline.py --device cpu
# 正式训练（诊断 YAML 默认 epoch）：加 --full-train
# 仅复用已有权重再出表：--skip-train
```

## 2. 未能自动完成或需人工加强的部分

- **关系类型条形图**：若 val 上几乎没有 `acc@1_rel::*` 键，则不会生成 `relation_stratified_plot.png`（本次 smoke 即如此）；表格 `relation_stratified_table.md` 可能为空。
- **两阶段诊断**：依赖仓库中是否已有 `outputs/checkpoints_stage1/coarse_geom_recall_last.pt` 与 `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`；**本次环境存在**，故 `shortlist_diagnostics.json` 中含 **K=10** 的 oracle / 条件 rerank；若无权重则仅保留 coarse **recall@1/5/10/20/40** 曲线。
- **多种子**：协议要求 3 种子时需重复运行并合并 `summary.csv`（当前脚本为单种子 42）。

## 3. 关键发现（基于本次 smoke 数值，仅作诊断参考）

| 模型 | 受控 Acc@1 | 全场景 Acc@1 |
|------|------------|----------------|
| A attribute-only | ~0.526 | ~0.045 |
| B raw-text relation | ~0.596 | ~0.051 |
| C structured | ~0.596 | ~0.032 |
| D + hard-negative | ~0.506 | ~0.013 |

- **受控候选**下 B/C 略高于 A；**单 epoch smoke 下 D（hardneg）未显示收益**，需 **full-train** 再下结论。
- **全场景** Acc@1 普遍 ~2–5%，与 README 中「全场景难」一致。
- **Shortlist**：coarse（`np_f_attr`）在 **recall@40≈0.89**，但 **recall@10≈0.28**；两阶段 **K=10** 时 **oracle 上界 ≈0.31**，**条件 rerank Acc@1≈0.125** —— 说明 **短表召回（K 较小）与 rerank 质量** 同时吃紧；增大 K 可抬高 oracle，但实际管线需与训练时 K 对齐。

## 4. 证据是否支持课程叙事

- **结构化关系在受控体制是否优于简单基线**：本次 smoke 上 **C≈B > A 部分成立**；需 full-train 与多种子巩固。
- **全场景是否被 shortlist / 几何限制**：**recall@K 随 K 上升明显** → 多路分类/检索瓶颈成立；`geometry_quality_results.json` 与 stratified 中的 fallback 切片可进一步支撑几何因素（见该目录下 `geometry_interpretation.md`）。

## 5. 后续优先三件事

1. 运行 `python scripts/run_next_phase_pipeline.py --full-train --device cuda`（或可用 GPU），刷新主表与 shortlist 诊断。  
2. 按 `configs/eval/stage1_recall_pass.yaml` 训练与 **K 对齐** 的 coarse+rereank，更新 `shortlist_diagnostics.json`。  
3. 对结构化模型补充 **relation 标签覆盖**（或换 split），使 `relation_stratified_plot.png` 有内容，便于报告 RQ2。

## 产物索引（最新一次）

| 路径 | 说明 |
|------|------|
| `outputs/20260326_093621_next_phase/main_table.md` | 主对比表 |
| `outputs/20260326_093621_next_phase/stratified_results.json` | 全部分层指标 |
| `outputs/20260326_093621_next_phase/hard_case_results.json` | 难例/子集 |
| `outputs/20260326_093621_next_phase/shortlist_diagnostics.json` | Shortlist +（可选）两阶段 |
| `outputs/20260326_093621_next_phase/geometry_quality_results.json` | 几何相关切片 |
| `outputs/20260326_093621_next_phase/paraphrase_results.json` | 模板释义一致性 |
| `outputs/20260326_093621_next_phase/report_bundle/` | 报告用拷贝 + `README.md` |
| `outputs/20260326_093621_next_phase/repro_commands.sh` | 已执行命令记录 |
