# Focused shortlist hard negatives (three dangerous types)

This phase **does not** enable broad negative mining or extra load-weighting curricula by default. Coarse training uses **only** the three loss terms that map cleanly onto the requested negative classes, all implemented in `src/rag3d/relation_reasoner/losses.py` via `compute_batch_training_loss`.

## 1. Same-class negatives

- **Mechanism:** `loss.hard_negative` → **`same_class_hinge_loss`** (despite the key name, this is **same-class** pushing, not generic hard-negative mining).
- **Identification:** For each sample, objects with **`normalize_class_name(class_name)` equal to the gold object’s class** are treated as hard negatives; the hinge encourages the gold logit to exceed the **max same-class** competitor by `margin`.
- **Mixing:** Added on top of **standard cross-entropy** on the full candidate set (not a replacement).
- **Frequency:** Every batch where the loss block is **enabled**; strength via `lambda_hinge` and `margin`.

## 2. Near-anchor but wrong-target negatives

- **Mechanism:** `loss.spatial_nearby_hinge` → **`spatial_nearby_hinge_loss`**.
- **Identification:** Uses **3D center distances** from the gold object; considers up to **`max_neighbors`** nearest **other** valid objects as “nearby wrong” competitors.
- **Mixing:** Same as above — additive to CE, gated by `lambda` / `margin`.
- **Frequency:** Every batch when enabled and geometry / centers are available on `samples_ref`.

## 3. Current coarse top-ranked false positives

- **Mechanism:** `loss.ranking_margin` → **`hardest_negative_margin_loss`**.
- **Identification:** Among all **non-gold** valid objects, uses the **highest logit** negative (the **hardest** mistake under the current scorer) and applies a margin hinge vs the gold logit — i.e. the **current top false positive** pressure.
- **Mixing:** Additive CE + hinge; `lambda` controls weight.
- **Frequency:** Every batch when enabled.

## What is **not** turned on in the focused longtrain YAML

- **`candidate_load_weight`** — useful for **clutter reweighting**, but it is **not** one of the three negative *types* above; leaving it **off** keeps this phase from conflating “focused hard negatives” with **global reweighting**.

## Why this is safer than a broad hard-negative expansion

Broad mining (extra categories, aggressive sampling, or multi-source distractors) often **dominates the loss** with **noisy** or **mislabeled** neighbors and can **destabilize** coarse logits that must stay **consistent** with the frozen reranker’s training distribution. The three terms above are **local**, **geometry- and class-grounded**, and **directly tied** to failure modes of shortlist construction (confusers in class, in space, and at the **current** decision boundary).

## Training selection under this coarse objective

When `val_two_stage_selection.enabled: true`, the **saved best coarse checkpoint** is chosen by **natural two-stage val Acc@1** with a **fixed reference reranker**, not by Recall@K alone — see `reports/checkpoint_selection_note.md`.
