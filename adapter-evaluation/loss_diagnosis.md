# Loss Curve Diagnosis — AfyaPlus LoRA Adapter

**Verdict: HEALTHY**

## Summary numbers
| Metric | First step | Last step | Change |
|---|---|---|---|
| Train loss | 2.9478 | 1.4488 | -50.9% |
| Eval loss | 2.9480 | 1.4749 | -50.0% |

- Final gap (eval_loss - train_loss): **0.0261** (overfitting threshold used: 0.3)
- Validation loss still decreasing at final step: **True**
- Best checkpoint by eval_loss was the final checkpoint (step 15): **True**
- Total optimiser steps: 15 over 3 epochs
  (effective batch size 16 = per-device batch 4 x grad-accum 4; 80 train examples / 16 ≈ 5 steps/epoch)

## Reading the curve
Both curves fall smoothly and almost in lockstep from epoch 0.2 through epoch 3.0:
train loss drops from 2.95 to 1.45, and validation loss drops
from 2.95 to 1.47. At no point does validation loss plateau while
train loss keeps falling, and it never turns upward — the classic overfitting signature.
The gap between the two curves stays under 0.026 at the final step, well inside
the 0.3 threshold used for this domain (AfyaPlus is a safety-sensitive, low-data
domain, so a tight train/eval gap matters more than raw loss magnitude).

## Diagnosis: HEALTHY
The run is **healthy**:
- Loss fell by 51% (train) and 50% (eval) over 3 epochs —
  the model is clearly learning the AfyaPlus instruction format and domain content, not
  underfitting (it did not plateau early or stay near its starting loss).
- Train and eval loss track each other closely with no divergence at the end of training —
  the model is not memorising the 80 training examples at the expense of generalising to
  the 10 held-out validation examples, so it is not overfitting either.
- `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"` selected the final
  checkpoint (step 15) as the best, consistent with a curve that was
  still improving when training stopped rather than one that had already started degrading.

## Caveats
- The dataset is small (80 train / 10 val examples), so only 15 optimiser steps were run.
  Both curves are smooth, but with only 10 validation examples the eval_loss estimate has
  higher variance than a larger held-out set would give — treat "healthy" as a directional
  read, not a statistically tight one.
- Because validation loss was still falling at the last step, more epochs (or a larger
  dataset) could plausibly extract further gains before overfitting sets in; this was not
  tested since the run was capped at 3 epochs per the assignment's compute budget.
