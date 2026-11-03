# V4 checkpoint resume changes

## What changed

- `11_train_experiment.py` now saves an atomic `*.resume.pt` file after every completed epoch. It contains model weights, optimizer state, learning-rate scheduler state, next epoch, early-stopping counter, best validation MAE, training history, and RNG states.
- The ordinary `*.pt` file remains a plain best-model `state_dict`, so `10_eval_canonical.py` continues to load it without changes.
- Completed folds (`*.pt` plus `*.json`) are skipped unless `--force` is passed.
- If training is interrupted, rerun the same training command without `--force`; the trainer resumes at the next epoch from the latest full state.
- Legacy interrupted runs that have only `*.pt` but no `*.resume.pt` are warm-started from their best saved weights. Their exact epoch and optimizer cannot be reconstructed, so that first continuation starts a fresh optimizer and creates full resume state going forward.
- The pipeline defaults `RUN_OPTUNA` to `0`, because the provided log already shows the 30-trial Optuna search completed and then failed at the missing config entry. This avoids repeating the expensive search. Set `RUN_OPTUNA=1` explicitly if you intend to run a new search.
- The pipeline does not pass `--force` during normal training.

## Resume commands

Run from the project directory, with the same Python environment used previously:

```bash
python 11_train_experiment.py --experiment v4_resnet50_scratch --folds 0,1,2,3,4 --seeds 2
```

Use the same experiment, fold, seed count, and epoch limit as the interrupted run. Existing completed folds will be skipped and folds with `*.resume.pt` will resume.

To run the full pipeline:

```bash
chmod +x run_v4_pipeline.sh
./run_v4_pipeline.sh
```

The QC review pause remains in the pipeline. Type `YES` only after checking the QC CSV. The default pipeline skips Optuna and proceeds with the predefined experiments, evaluation, and downstream analysis.

## Important safety notes

- Do not delete `v4/checkpoints/*.resume.pt` if you want epoch-level resumption.
- Avoid `--force` unless you intentionally want to start a fold from scratch. Back up checkpoints before using it.
- A Ctrl+C during an epoch resumes from the last fully completed epoch, not partway through the current epoch.
- This patch was syntax-checked, but not run through full MRI training in this environment. Test on one fold with a short `--epochs` value before relying on it for a long run.
