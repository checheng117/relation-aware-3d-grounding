# Clean Baseline Repair Log

Date: 2026-04-20

## Repairs

| Repair | Files | Reason | Verification |
| --- | --- | --- | --- |
| Added clean sorted-vocab configs | `repro/referit3d_baseline/configs/clean_sorted_vocab_baseline.yaml`, `repro/referit3d_baseline/configs/clean_sorted_vocab_smoke.yaml` | Existing learned-class config had stale/missing BERT cache path and wrong class count risk. | Full train completed; smoke train/resume completed. |
| Rebuilt full official DistilBERT cache | `data/text_features/full_official_nr3d/` | The directory existed but was empty; Apr9/Apr14 checkpoint metadata referenced it. | `prepare_bert_features.py --validate-only` returned no issues. |
| Saved sorted class vocabulary into checkpoints | `repro/referit3d_baseline/scripts/train.py` | Fresh exports must not depend on reconstructing a hidden class mapping. | `class_vocabulary.json` and checkpoint `class_to_idx` exist, size=516. |
| Added resume support | `repro/referit3d_baseline/scripts/train.py` | Required smoke/resume/recoverability verification. | Resume from smoke best checkpoint ran epoch 2/2 successfully. |
| Fixed smoke checkpoint save edge case | `repro/referit3d_baseline/scripts/train.py` | A max_batches smoke can have val Acc@1=0.0; previous best=0.0 initialization skipped `best_model.pt`. | Rerun saved `outputs/20260420_clean_sorted_vocab_baseline/smoke/best_model.pt`. |
| Exported entropy/probability/tie diagnostics | `repro/referit3d_baseline/scripts/evaluate.py` | P3 readiness needs more than top1/top2 raw margin. | Readiness checker validated val/test exports. |
| Evaluator uses checkpoint vocabulary | `repro/referit3d_baseline/scripts/evaluate.py` | Evaluation/export must use the same run vocabulary as training. | Logs show "Using class vocabulary stored in checkpoint (516 classes)". |
| Added export consistency checker | `scripts/check_clean_baseline_readiness.py` | Needed auditable proof that logits export and eval metrics agree. | PASS on clean val/test exports. |
| Added real-logits P3 entry | `scripts/run_cover3d_real_logits_p3_entry.py` | Needed a real base-logits Base/Dense-no-cal/Dense-calibrated runnable entry. | PASS on clean test export. |
| Updated wrapper config | `configs/cover3d_referit_wrapper.yaml` | Formal method entry must not point to Apr9 checkpoint. | Config points to clean checkpoint/logits/class vocabulary. |

## Commands And Outcomes

| Command | Outcome |
| --- | --- |
| `python scripts/prepare_bert_features.py --manifest-dir data/processed/scene_disjoint/official_scene_disjoint --output-dir data/text_features/full_official_nr3d --model distilbert-base-uncased --batch-size 128 --device cuda` | Train split generated; first session stopped before val/test. |
| Custom `prepare_split_features` for val/test | Generated val/test DistilBERT features. |
| `python scripts/prepare_bert_features.py --output-dir data/text_features/full_official_nr3d --validate-only` | PASS, shapes train=(33829,768), val=(3419,768), test=(4255,768), issues=[]. |
| `python -m py_compile ...` | PASS. |
| `python repro/referit3d_baseline/scripts/train.py --config ...clean_sorted_vocab_smoke.yaml --device cuda` | First run exposed missing best checkpoint when val Acc@1=0.0; fixed and reran successfully. |
| `python repro/referit3d_baseline/scripts/train.py --config ...clean_sorted_vocab_smoke.yaml --device cuda --resume-checkpoint outputs/20260420_clean_sorted_vocab_baseline/smoke/best_model.pt` | PASS, resumed at epoch 2/2. |
| `python repro/referit3d_baseline/scripts/train.py --config ...clean_sorted_vocab_baseline.yaml --device cuda` | PASS, 30 epochs, best val Acc@1=33.20%. |
| `python repro/referit3d_baseline/scripts/evaluate.py --checkpoint outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt --split val --device cuda --output-dir outputs/20260420_clean_sorted_vocab_baseline/formal_logits --export-logits` | PASS, val Acc@1=33.20%, Acc@5=92.89%. |
| Same evaluate command with `--split test` | First attempt hit transient manifest/Pydantic load error without persistent data defect; rerun PASS, test Acc@1=30.83%, Acc@5=91.87%. |
| `python scripts/check_clean_baseline_readiness.py ... val ...` | PASS, 3,419 rows. |
| `python scripts/check_clean_baseline_readiness.py ... test ...` | PASS, 4,255 rows. |
| `git diff --check` | PASS. |

## Notes

The real-logits P3 entry originally exposed intermittent Python 3.13 / NumPy `.npz` read instability when loading many geometry files in-process. The entry now isolates geometry loading per scene in a subprocess. This change only affects the readiness entry and does not alter training or evaluation.
