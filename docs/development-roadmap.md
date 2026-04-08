# 开发路线：分 4 个 Sprint

## Sprint 1：先把"评测底座"做扎实

这是你现在最该做的。

要落地的东西是：

### 统一 sample schema
每个样本统一成：
```
scene_id, utterance, target_id, candidate_objects, relation_tags, difficulty_tags, parser_cache_key
```
对象级统一成：
```
object_id, class_name, center, size, sampled_points/point_indices, visibility_proxy, embedding_cache_key
```
这一步对应你蓝图里的 "sample schema 统一" 和 "object schema 统一"。

### 把 evaluator 拆成三层
- **overall**：Acc@1 / Acc@5
- **stratified**：按 relation type、same-class clutter、occlusion-heavy 分层
- **diagnostic**：anchor confusion、target margin、entropy、failure taxonomy

这一步是全项目的核心，因为后面每个模型都必须走这一套统一评测。

### 先做 hard-case tagging
你蓝图里已经明确说不要只看 overall Acc@1，要看 same-class clutter、relation-heavy、paraphrase stability、anchor confusion。

所以第一周就该把 tag 管道搭出来，哪怕一开始部分 tag 是 heuristic 规则，也比后面补强得多。

这一步完成的验收标准很简单：
任意一个现有 checkpoint 跑完后，除了 overall 表，还能自动吐出 relation-stratified 表、hard-case 表、失败案例 JSON。

## Sprint 2：把 baseline 重新整理成"论文级对比组"

你蓝图里规定的最低基线非常合理，不要再乱加。就保留三条主线：

- attribute-only listener
- raw-text relation model
- structured parser + soft anchor + relation scorer（你的主方法）

你现在仓库里其实已经有 attribute baseline、raw-text relation、relation-aware structured scorer 这些模型线索，所以这里不一定是从零写，更像是统一接口、统一训练脚本、统一评测出口。README 也写了当前仓库本来就是围绕这些模型线展开的。

这一阶段最重要的不是追新高，而是把三条线都跑进同一个 evaluator，并且每条线都能输出：

- overall metrics
- relation subset metrics
- same-class clutter metrics
- error casebook

这样你最终 report 里的对比表就自然出来了。

## Sprint 3：把"升级点"做成真正的贡献点

这一步才是蓝图里的主创新区，但只做三项就够，不要发散。

### A. Structured parser + cache
InternVL 或其他 VLM 只负责轻量结构化解析，不负责直接预测目标。输出固定 JSON：
```
target_head, target_modifiers, anchor_head, relation_types, parser_confidence, paraphrase_set
```
而且必须加 cache 和 fallback；蓝图已经提醒过 parser 噪声会污染主模型，所以要做缓存、规则回退、heuristic ablation。

### B. Soft anchor selector
把"找 anchor"和"判 relation"拆开，这是你整个项目最像样的结构化贡献之一。输出至少要有：
```
anchor_distribution, anchor_entropy, top_anchor_id
```
因为之后的 diagnostics、可视化、稳定性分析都依赖它。

### C. Relation scorer + hard-negative aware training
这里不要把创新做太散。就聚焦在：

- 候选对象 i 与软 anchor 分布的关系打分
- 同类多实例 hard negative
- 邻近 anchor sharing hard negative

这样你的结论就能很稳地落在："收益主要来自 relation-heavy 与 same-class clutter，而不是刷 easy cases"。

这一阶段的验收标准是：
在 Sr3D / relation-heavy 子集上，你的方法至少要比 raw-text relation 更有解释性，即使 overall 提升不算夸张，也要在 hard-case 上有明确收益。 这正是蓝图里的主张和假设。

## Sprint 4：把项目从"代码"升级成"作品"

这是你最后真正决定分数和简历观感的一步。

必须产出四类东西：

### case-study visualizer
输出 scene-level 图，标出 target、top anchors、anchor 概率、relation rationale。
因为蓝图明确要求 scene-level 可视化、anchor 概率图、可直接截图放主页的 qualitative figure。

### failure taxonomy
至少分成：
- anchor confusion
- same-class confusion
- parser failure
- relation mismatch
- low-confidence / high-entropy ambiguous case

你蓝图里已经把 diagnostics + failure taxonomy 单独列成 issue 了，这个绝对要做。

### paraphrase stability evaluation
这会让项目一下子从"普通 benchmark 改模"变成"结构化推理稳定性研究"。
直接比较原句和改写句的：
- target consistency
- anchor distribution drift
- relation tag stability

### README / report / homepage 三件套
蓝图已经给了标准答案：
- 一张总览图
- 两张最强 qualitative 图
- 一个 failure taxonomy 图
- 一个 metrics 主表

这些不是附加项，而是项目完成态的一部分。

## 你现在最该开的 Issue 顺序

直接按这个顺序开，不要改：

1. 数据读取与 sample schema 统一
2. evaluator + stratified metrics
3. attribute-only baseline
4. raw-text relation baseline
5. parser cache + structured JSON schema
6. soft anchor selector
7. relation scorer
8. hard negative mining / sample tagging
9. paraphrase stability evaluator
10. diagnostics + failure taxonomy + visualization
11. README / report figures / homepage packaging

这套顺序的好处是：前 4 个 issue 完成时，你已经有一个"能交作业的最小系统"；前 8 个完成时，你已经有"能讲方法贡献的研究项目"；11 个全做完时，它才真正成为"能放 GitHub 首页和简历"的作品。

## 我不建议你接下来做的事

有三件事先别做。

**第一，不要继续大规模 LR/seed sweep。**
因为 README 已经告诉你这条线收益递减，而且 seed robustness 没站稳。

**第二，不要把它扩成另一个 EmbodiedSceneAgent。**
蓝图反复强调它是 perception bridge，不是 planner，不抢 execution/planning 的角色。

**第三，不要让 VLM 直接替代几何推理主干。**
VLM 在这里应当只承担轻量结构化解析和可选 paraphrase，主干仍然是 object-centric 3D reasoning。

## 最后给你一个最实际的判断

如果你问"接下来最优的一步是什么"，答案其实非常明确：

先别训练新模型，先重构 evaluator 和 sample/object schema。

因为一旦这层打通，后面的 baseline、parser、anchor、relation、diagnostics 都是往上加模块；但如果这层不先做，你后面每加一个新设计，都会重新陷入"结果散、图表乱、案例难讲、报告不好写"的状态。这个判断和你蓝图里的推荐执行顺序完全一致。