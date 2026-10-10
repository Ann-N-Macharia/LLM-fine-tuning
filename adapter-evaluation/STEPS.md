# How the Loss Curve Analysis Was Produced

This explains, step by step, how the files in this `adapter-evaluation/` folder were
generated from the completed training run, so the process is reproducible.

## Files in this folder
| File | What it is |
|---|---|
| `loss_curve.png` | Plot of training loss vs validation loss across the 3 training epochs |
| `loss_curve_data.csv` | The raw step/epoch/loss numbers behind the plot |
| `loss_diagnosis.md` | Written verdict (healthy / overfit / underfit) with the reasoning and numbers |
| `STEPS.md` | This file |

## Step 1 — Locate the training record
After `fine_tune.py` finished running (on the vast.ai GPU instance) and the adapter was
downloaded locally, Hugging Face's `Trainer` had already written a `trainer_state.json`
file into the adapter output directory: `afyaplus-llama-adapter/trainer_state.json`.
This file is created automatically by `transformers.Trainer` — no extra code was needed
during training to produce it, only `eval_strategy="steps"` and `eval_steps` set in
`TrainingArguments` (see `fine_tune.py`) so that validation loss was actually recorded
alongside training loss.

## Step 2 — Parse the log history
`trainer_state.json` contains a `log_history` list: one dict per logged event. Training
steps carry a `loss` key; evaluation steps (which happen every `eval_steps` steps) carry
an `eval_loss` key instead. [monitor_training_loss.py](../monitor_training_loss.py) loads
the file and splits `log_history` into two pandas DataFrames — one for train-loss rows,
one for eval-loss rows — keyed by `step` and `epoch`.

## Step 3 — Plot both curves together
Using `matplotlib`, the script plots `train_loss` and `eval_loss` against `epoch` on the
same axes and saves the figure as `loss_curve.png` (150 DPI, saved not just shown, so it
works headlessly on a machine with no display). Plotting both curves on one chart is what
makes over/underfitting visible: divergence between the two lines is the signal to look
for, not the absolute loss value.

## Step 4 — Save the raw numbers
The merged train/eval table is also written to `loss_curve_data.csv` so the exact figures
behind the plot are available for the report and for re-plotting if needed, without
re-parsing the JSON.

## Step 5 — Compute the diagnosis
The script derives a small set of signals from the curve and turns them into a verdict:
1. **Final train/eval gap** (`eval_loss - train_loss` at the last step) — a large,
   growing gap is the signature of overfitting.
2. **Whether validation loss was still falling** at the last logged step, or had turned
   upward — an upward turn while train loss keeps falling is overfitting even if the
   absolute gap is still small.
3. **Whether loss dropped substantially from its starting value** — if train and/or eval
   loss stayed near their epoch-0 starting point, that indicates underfitting (the model
   never learned the task).
4. **Whether the best checkpoint by `eval_loss` was the last one saved** — pulled directly
   from `trainer_state.json`'s `best_model_checkpoint` field, this corroborates whether
   training was stopped at (rather than past) the best point.

These checks classify the run as `HEALTHY`, `OVERFIT`, or `UNDERFIT` and are written out,
with the supporting numbers, as `loss_diagnosis.md`.

## Step 6 — Run it
```bash
# from the fine_tuning/ project root, with the venv activated
python monitor_training_loss.py
```
This prints the two loss tables to the console and writes the three files above into
`adapter-evaluation/`. Re-running it after any future training run overwrites these
files with the new run's numbers.

## Result for this run
**Verdict: HEALTHY.** Train loss fell from 2.95 → 1.45 and validation loss fell from
2.95 → 1.47 over 3 epochs (15 optimiser steps, effective batch size 16 on 80 training
examples). The two curves track each other closely throughout with a final gap of only
~0.03, and validation loss never plateaus or turns upward — see
[loss_diagnosis.md](loss_diagnosis.md) for the full write-up and caveats (small
dataset, only 15 steps, so treat the read as directional).
