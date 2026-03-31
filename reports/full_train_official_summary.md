# 官方 Full-Train 实验总结（单种子 + 多种子 B/C）

## 1. 实际完成了什么

### 1a. 单种子官方跑（历史）

- **命令**：`python scripts/run_next_phase_pipeline.py --official-full-train --scope bc --seeds 42 --device auto`
- **输出目录**：`outputs/20260326_095106_full_train_official/`
- **要点**：B/C 各 entity + full；full A 属性粗模型用于 shortlist；`main_table_official` 为 **n_seeds=1**。

### 1b. 多种子官方 B/C（本阶段新增）

- **命令**：`python scripts/run_next_phase_pipeline.py --official-full-train --scope bc --seeds 42,43,44 --device auto`
- **输出目录**：`outputs/20260327_135641_full_train_official/`
- **种子**：42、43、44 **全部成功**（见 `multiseed_run_status.json`）
- **聚合**：`main_table_official.csv/.md` 为 **mean ± std**（`n_seeds=3`）
- **后处理**：`python scripts/postprocess_bc_multiseed_report.py --exp-dir outputs/20260327_135641_full_train_official --prior-official outputs/20260326_095106_full_train_official`
- **正文级打包**：`outputs/20260327_135641_full_train_official/report_bundle_main_text/`（主表、难例/几何表与主图、shortlist 曲线与解读、README）
- **完整训练日志副本**：`outputs/20260327_135641_full_train_official/multiseed_training_console.log`

## 2. 多种子后主表（与单种子对比）

| 设定 | 单种子 `20260326_095106` | 多种子 `20260327_135641`（mean ± std） |
|:---|:---|:---|
| **受控 Acc@1** | B 0.558 **>** C 0.455 | B 0.536±0.078，C **0.541±0.045**（**均值接近，C 略高；区间重叠**） |
| **全场景 Acc@1** | C 0.051 **>** B 0.013 | B **0.045±0.006** **>** C 0.021±0.010（**与单种子方向相反**） |

→ **体制依赖的「一句话结论」在跨种子下不稳定**；终稿应 **并列两种证据** 并强调 **方差与种子敏感性**。详细自检见 `reports/bc_multiseed_claim_check.md`。

## 3. 主要子集与难例（多种子后）

- **same_class_clutter（受控）**：B/C **均值相同**（约 0.455），B **方差更大** → **不再支持**「C 在 clutter 上稳定优于 B」作为主文核心论点。
- **same_class_clutter（全场景）**：**B 高于 C**（约 0.045 vs 0.021）→ 与单种子 clutter 叙述 **冲突**； clutter 相关表述须 **降级或重写**。
- **low_model_margin 等**：见 `hard_case_results.json` 与 `hard_case_interpretation.md`。

## 4. 几何质量

- 仍为 **分层/切片 Acc@1**，**相关而非因果**。多种子后见 `geometry_quality_table.*` 与 `geometry_quality_interpretation.md`。
- **主文可写（弱）**：几何/特征不完整与全场景极低 Acc **并存**，作 **限制讨论** 之一。
- **不宜过强**：单一归因「几何差导致失败」。

## 5. Shortlist 诊断

- `outputs/20260327_135641_full_train_official/shortlist_interpretation.md`：**小 K 检索弱 + 条件 rerank 仍弱** → **混合瓶颈（检索 + rerank）** 结论 **与单种子官方一致**；多种子 B/C **未推翻** 该诊断。
- 粗 recall 的多种子均值表见同文件；与 prior official 数值略有出入属 **不同 checkpoint/种子** 预期内波动。

## 6. B vs C 终稿可写与不可写

**可写（保守）**：

- 官方 **多种子主表** + **单种子结果** 并列，明确 **full 上 B/C 排序可随种子翻转**。
- **受控** 上 B/C **整体接近**，不宜再写「B 稳赢」或「C 稳赢」而无方差说明。
- Shortlist **混合瓶颈** 仍适合主文诊断图。

**不可写（过强）**：

- 「结构化在全场景 **一致** 优于 raw-text」（多种子 **不支持**）。
- 「受控 **稳定** B > C」（多种子 **不支持**）。
- 「same-class clutter **稳定** 支撑 C」（多种子 **不支持**）。

## 7. D 为何仍推迟

- 本阶段目标为 **B/C 证据链与报告包**；**未训 D**（`--scope bc`）。在 B/C 主结论 **种子敏感** 的前提下，**不建议**优先引入 D 扩维。

## 8. 本阶段之后的优先动作

1. **终稿叙事**：以 **多种子表** 为统计主引用，**单种子** 作附录或「观测性」对照；全文统一 **种子与方差** 措辞。
2. **工程**：若需稳定 full-scene 结论，考虑 **更多种子** 或与 **验证协议**（early stopping、checkpoint 选择）对齐后再比 B/C。
3. **Stage-1 K 与 rerank**：shortlist 诊断仍指向 **检索 + rerank**；后续改进实验与 D **独立**，仍待你明确要求再开 D。

---

**执行计划**：`reports/full_train_execution_plan.md`  
**多种子声明核对**：`reports/bc_multiseed_claim_check.md`  
**复现**：`outputs/20260326_095106_full_train_official/repro_commands.sh`（单种子）、`outputs/20260327_135641_full_train_official/repro_commands.sh`（多种子 + postprocess）
