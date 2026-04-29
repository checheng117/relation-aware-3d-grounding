# Plan 001: Recover Project Direction

**Created**: 2026-04-29 14:35
**Status**: READY FOR EXECUTION
**Priority**: P0
**Estimated Effort**: 1–2 sessions (no GPU training)

---

## 0. Situation Assessment

### Where We Are

项目 "Latent Conditioned Relation Scoring for 3D Visual Grounding" 处于以下状态：

| 维度 | 状态 |
|------|------|
| 核心方法 | ✅ 已实现并通过 pilot 验证（ViewpointConditionedRelationScorer） |
| 主要 claim（C-001: +2.1%） | ✅ Supported，Phase 4/4.5 controlled evidence |
| 辅助 claim（C-006: CF +0.12pp） | ⚠️ Weak (pilot only) |
| 探索性 claim（C-007: K=4 MoE） | ⚠️ Weak (pilot only) |
| Course-line 报告 | ✅ Markdown 完稿（`writing/course-line/report.md`） |
| Paper-line 论文 | 🔶 LaTeX 初稿存在（`writing/paper-line/main_draft.tex`），缺 full results |
| Full training (E-011~E-014) | ❌ PENDING — 需要 headless GPU 环境 |
| Multi-seed validation (E-015) | ❌ PENDING |
| `phase7_full/` 输出 | 空目录 — 从未成功完成 |

### The Critical Decision Point

项目面临一个二选一的分叉：

- **路径 A**: 立即在远程 GPU 上运行 full training → 获得强有力的 full-epoch evidence
- **路径 B**: 使用当前 pilot evidence 完成论文 → 诚实标注 pilot 限制

**本计划选择**: 先执行路径 B 的准备工作（不依赖 GPU 的任务），同时准备路径 A 的 launch 脚本。这确保无论 GPU 是否可用，项目都有进展。

---

## 1. Most Important Next Task

**任务**: 验证现有 evidence 完整性 + 构建可提交的论文证据框架

**理由**:
1. Full training 依赖外部 GPU（当前 blocker），无法立即行动
2. 论文/报告是最终交付物，但当前 paper-line 草稿存在多处 placeholder
3. 现有 evidence（Level A + Level B）足以支撑主要 claim，但需要系统性验证其完整性
4. 不验证 evidence 完整性就直接写论文，会有引用错误数字的风险

---

## 2. Task Breakdown

### Task A: Evidence Verification（验证任务）

**目标**: 确认所有被引用的数字都能追溯到原始 output 文件

| 步骤 | 操作 | 需检查的文件 | 输出 |
|------|------|-------------|------|
| A1 | 验证 baseline 30.83% | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json` | 确认或标记差异 |
| A2 | 验证 Phase 4 E0=28.60%, E1=30.74%, E3=30.65% | `outputs/phase4_ablation/*.json` | 确认或标记差异 |
| A3 | 验证 Phase 5 pilot E0=34.30%, E1-CF=34.42% | `outputs/phase5_counterfactual/pilot_E0_bs8_safe/latent-conditioned_results.json`, `outputs/phase5_counterfactual/pilot_E1_CF_safe/latent-conditioned+cf_results.json` | 确认或标记差异 |
| A4 | 验证 Phase 6 pilot K1=34.10%, K4=34.19% | `outputs/phase6_latent_modes/pilot_E0_K1_safe/pilot_e0_k1_safe/training_history.json`, `outputs/phase6_latent_modes/pilot_E1_K4_safe/pilot_e1_k4_safe/training_history.json` | 确认或标记差异 |
| A5 | 验证 hard subset 数字（21.96%, 16.07%, 11.90%） | `reports/cover3d_phase1_baseline_subset_results.md` | 确认或标记差异 |
| A6 | 验证 coverage 33.95% | `reports/cover3d_coverage_diagnostics/coverage_summary.json` | 确认或标记差异 |

**输出文件**: `.agent/30_verify/001_evidence_audit.md`

**规则**:
- 仅读取文件，不修改任何 output
- 如发现数字不一致，记录在 audit 中但不自行"修正"
- 不发明数字、不推测缺失的值

---

### Task B: Paper Structure Audit（写作审计任务）

**目标**: 对比当前论文草稿与 evidence map，找出 gap

| 步骤 | 操作 | 需检查的文件 |
|------|------|-------------|
| B1 | 审查 course-line report 的数字引用是否与 Task A 验证结果一致 | `writing/course-line/report.md`, `writing/course-line/report.tex` |
| B2 | 审查 paper-line draft 的 placeholder 列表 | `writing/paper-line/main_draft.tex`, `writing/paper-line/DRAFT_STATUS.md` |
| B3 | 检查 CLAIM_BOUNDARY.md 中的 "Recommended Wording" 是否已体现在论文中 | `course-line/CLAIM_BOUNDARY.md` vs 论文文本 |
| B4 | 识别论文中尚未填充的 table/figure placeholder | 两份论文稿 |

**输出文件**: `.agent/30_verify/001_paper_gap_analysis.md`

**规则**:
- 不编辑论文文件
- 列出每个 gap 的位置（section + line number）
- 为每个 gap 标注 "可用当前 evidence 填充" vs "需要 full training 数据"

---

### Task C: Full Training Launch Script Audit（代码审计任务）

**目标**: 确认 full training 脚本可用、配置正确，为 GPU 可用时做好准备

| 步骤 | 操作 | 需检查的文件 |
|------|------|-------------|
| C1 | 检查 `scripts/run_phase7_full.sh` 的完整性和正确性 | `scripts/run_phase7_full.sh` |
| C2 | 检查 `scripts/train_cover3d_counterfactual.py` 的 full-training config | `scripts/train_cover3d_counterfactual.py` |
| C3 | 检查 `scripts/train_cover3d_latent_modes.py` 的 full-training config | `scripts/train_cover3d_latent_modes.py` |
| C4 | 检查 `configs/cover3d_referit_wrapper.yaml` 基础配置 | `configs/cover3d_referit_wrapper.yaml` |
| C5 | 检查 data path 是否在当前环境可访问 | `data/` 目录结构 |
| C6 | 检查 `environment.yml` / `pyproject.toml` 依赖是否可安装 | `environment.yml`, `pyproject.toml` |

**输出文件**: `.agent/30_verify/001_training_readiness.md`

**规则**:
- 不修改任何脚本或配置
- 不启动 GPU 训练
- 仅运行轻量级检查（如 `python -c "import rag3d"`, dry-run 类检查）
- 记录发现的任何 blocker

---

### Task D: Evidence-to-Paper Mapping（写作准备任务）

**目标**: 创建精确的 evidence → paper section 映射表，供后续写作使用

| 步骤 | 操作 |
|------|------|
| D1 | 将 `CLAIM_LEDGER.md` 的每个 claim 映射到论文中应出现的 section |
| D2 | 将 `FINAL_PROJECT_EVIDENCE_MAP.md` 的每个 evidence 映射到论文 table/figure |
| D3 | 为每个 main table 行指定数据来源文件路径 |
| D4 | 标注哪些 section 可以用 pilot evidence 完成（标记 "pilot label required"） |
| D5 | 标注哪些 section 必须等 full training 结果 |

**输出文件**: `.agent/20_exec/001_evidence_paper_map.md`

**规则**:
- 不编辑论文或 evidence 文件
- 使用 `CLAIM_BOUNDARY.md` 的措辞规范
- 所有 pilot evidence 必须标注 "pilot" 

---

## 3. Execution Order

```
Phase 1 (可并行):
  Task A: Evidence Verification  ←  最高优先级
  Task C: Training Script Audit  ←  独立于 Task A

Phase 2 (依赖 Phase 1):
  Task B: Paper Structure Audit  ←  需要 Task A 的验证数字
  Task D: Evidence-to-Paper Map  ←  需要 Task A + Task B 的结果

Phase 3 (依赖 Phase 2, 本计划不执行):
  → 后续计划: 002_fill_paper_gaps.md (写作任务)
  → 后续计划: 003_launch_full_training.md (实验任务)
```

---

## 4. Files to Inspect (Complete List)

### Must Read (Evidence Sources)
- `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`
- `outputs/phase4_ablation/*.json`（所有 E0/E1/E2/E3 结果）
- `outputs/phase5_counterfactual/pilot_E0_bs8_safe/latent-conditioned_results.json`
- `outputs/phase5_counterfactual/pilot_E1_CF_safe/latent-conditioned+cf_results.json`
- `outputs/phase5_counterfactual/pilot_E2_RHN_safe/random-hard-neg_results.json`
- `outputs/phase6_latent_modes/pilot_E0_K1_safe/pilot_e0_k1_safe/training_history.json`
- `outputs/phase6_latent_modes/pilot_E1_K4_safe/pilot_e1_k4_safe/training_history.json`
- `reports/cover3d_phase1_baseline_subset_results.md`
- `reports/cover3d_coverage_diagnostics/coverage_summary.json`

### Must Read (Paper Drafts)
- `writing/course-line/report.md`
- `writing/course-line/report.tex`
- `writing/paper-line/main_draft.tex`

### Must Read (Normative Documents)
- `course-line/CLAIM_BOUNDARY.md`
- `course-line/FINAL_PROJECT_EVIDENCE_MAP.md`
- `.agent/00_state/CLAIM_LEDGER.md`

### Must Read (Training Readiness)
- `scripts/run_phase7_full.sh`
- `scripts/train_cover3d_counterfactual.py`
- `scripts/train_cover3d_latent_modes.py`
- `configs/cover3d_referit_wrapper.yaml`
- `environment.yml`
- `pyproject.toml`

### Files to Create
- `.agent/30_verify/001_evidence_audit.md`
- `.agent/30_verify/001_paper_gap_analysis.md`
- `.agent/30_verify/001_training_readiness.md`
- `.agent/20_exec/001_evidence_paper_map.md`

### Files NOT to Edit
- Any file under `src/`, `scripts/`, `outputs/`, `data/`
- Any file under `writing/` (read-only in this plan)
- `.agent/00_state/*`（本计划完成后由 verifier 决定是否更新）

---

## 5. Definition of Done

本计划 **完成** 的条件：

- [ ] **A-DONE**: `.agent/30_verify/001_evidence_audit.md` 已创建，包含所有 6 个数字验证的 ✅/❌ 状态
- [ ] **B-DONE**: `.agent/30_verify/001_paper_gap_analysis.md` 已创建，列出论文中所有 gap 及其可填充性
- [ ] **C-DONE**: `.agent/30_verify/001_training_readiness.md` 已创建，列出 full training 的所有 blocker（如有）
- [ ] **D-DONE**: `.agent/20_exec/001_evidence_paper_map.md` 已创建，包含 claim → section 的完整映射
- [ ] **所有验证文件中没有发明的数字** — 每个数字都有文件路径引用
- [ ] **TODO.md 中 P0 第三项标记为完成**

---

## 6. Verifier Focus

执行此计划后，Verifier 应重点检查：

### 数字一致性
- [ ] `001_evidence_audit.md` 中每个数字是否与其引用的 JSON/MD 文件一致
- [ ] 是否有任何数字来源是 "从记忆中" 而非文件

### Claim 合规性
- [ ] `001_evidence_paper_map.md` 中是否有任何 "Do Not Claim" 级别的 claim 被映射到 main text
- [ ] 所有 pilot evidence 是否标注了 "pilot" 标签
- [ ] 是否违反了 `CLAIM_BOUNDARY.md` 中的任何禁止 claim

### 完整性
- [ ] 是否所有 `CLAIM_LEDGER.md` 中的 claim 都在 evidence-paper map 中有对应
- [ ] 是否所有 `EXPERIMENT_LEDGER.md` 中 COMPLETE 状态的实验都被验证了

### 训练就绪性
- [ ] `001_training_readiness.md` 是否识别了所有已知的 blocker
- [ ] 脚本路径是否真实存在
- [ ] 数据路径是否可访问

### 禁止行为
- [ ] 计划执行者是否修改了任何 source code
- [ ] 计划执行者是否修改了任何 output 文件
- [ ] 计划执行者是否发明了任何实验结果

---

## 7. Research Direction Preservation

本计划严格保持当前研究方向不变：

| 维度 | 当前方向 | 本计划影响 |
|------|---------|-----------|
| 主要 claim | Latent conditioned architecture +2.1% | 不改变，仅验证 |
| 方法 | ViewpointConditionedRelationScorer | 不改变 |
| 辅助 claim | CF +0.12pp (pilot) | 不改变，明确标注 pilot |
| 探索方向 | MoE K=4 | 不改变，保留为 future work |
| 禁止 claim | CF>RHN, Multi-anchor solved, SOTA | 继续禁止 |
| 论文定位 | Diagnostic/method paper for 3D grounding | 不改变 |

---

## 8. Risk Mitigation

| 风险 | 缓解措施 |
|------|---------|
| Evidence audit 发现数字不一致 | 记录差异但不修改数据；在后续计划中决定使用哪个数字 |
| Training script 存在 bug | 记录但不修复；在 003 计划中单独处理 |
| Paper draft 与 evidence 严重脱节 | 创建详细 gap list；在 002 计划中按优先级填充 |
| GPU 环境长期不可用 | Task D 确保 pilot-only 论文仍可完成 |

---

## Appendix: Downstream Plans (Preview)

基于本计划的结果，预计产生以下后续计划：

| Plan ID | 名称 | 触发条件 |
|---------|------|---------|
| 002 | Fill Paper Gaps with Current Evidence | 本计划完成且 evidence audit 通过 |
| 003 | Launch Full Training on Remote GPU | GPU 环境可用 |
| 004 | Multi-seed Validation | Full training 完成 |
| 005 | Final Paper Polish & Submission | 002 + (003 或 决定使用 pilot) 完成 |
