# V4 Brain-Age Improvements Patch

This is an **overlay patch** for the existing `v4 approach` project, not a replacement for the full project/data archive.

## What it changes

1. **Longer training:** `MAX_EPOCHS=300`, early-stopping patience `30`, and ReduceLROnPlateau with a less aggressive patience/cooldown. Training can still stop early when validation MAE plateaus.
2. **Deeper transfer learning:** new `v4_sfcn_deep_transfer` starts from the existing UK Biobank-pretrained SFCN, keeps the pretrained feature extractor and transferred age-bin head, and adds three trainable 3D residual convolution blocks.
3. **Higher-capacity architecture comparisons:** new `v4_resnet34_scratch` and `v4_resnet50_scratch`. The supplied project archive has pretrained SFCN weights but no matching pretrained ResNet weights, so these two models train from scratch. Do not describe them as transfer learning.
4. **Fairer baseline settings:** the retrained `v4_sfcn_baseline` now uses V3 m1's layer-wise learning-rate convention (`lr_head=3e-4`, `lr_backbone=3e-4`, `llrd=0.5`). The original V4 baseline had different values despite being described as V3 m1-equivalent.
5. **Long-vs-short training comparison:** before retraining, `run_v4_pipeline.sh` copies existing baseline checkpoints to the `v4_sfcn_baseline_40ep` tag when available. It then backs up existing checkpoints and retrains the registered experiments with the extended schedule.
6. **QC decisions now matter consistently:** optional `qc_manual_review` values are respected by training, Optuna, and canonical evaluation:
   - `CONFIRMED-FAILURE` excludes the scan even if the automatic flag was zero.
   - `LOOKS-OK` overrides an automatic QC flag after visual review.
   - blank/unrecognized values preserve the automatic `qc_flag`.
   QC is based on image quality, never on prediction error.
7. **More complete training batches:** `drop_last=False` avoids silently discarding the final partial batch.

## Files in this patch

Copy these files into the root of your existing `v4 approach` directory, overwriting the matching files:

- `config_v4.py`
- `core_v4.py`
- `deep_sfcn.py` (new)
- `11_train_experiment.py`
- `12_optuna_search.py`
- `10_eval_canonical.py`
- `run_v4_pipeline.sh`

The patch assumes the existing V4 directory still contains `config_v3.py`, `core_v3.py`, `resnet3d.py`, the UK Biobank pretrained SFCN checkpoint, the manifest/fold files, and the preprocessed NPZ data.

## Before running

1. Keep the existing project/zip as a backup.
2. Finish the manual MRI QC review first. Update the `qc_manual_review` column in the configured QC CSV using only image-quality judgments.
3. Make sure the V3 manifest and subject-level folds are already present. This patch intentionally does not regenerate the frozen folds.
4. From the V4 directory, run:

   ```bash
   python -m py_compile config_v4.py core_v4.py deep_sfcn.py 11_train_experiment.py 12_optuna_search.py 10_eval_canonical.py
   bash -n run_v4_pipeline.sh
   ```

5. Start the workflow:

   ```bash
   chmod +x run_v4_pipeline.sh
   ./run_v4_pipeline.sh
   ```

The runner asks for confirmation after creating QC image grids. Do not type `YES` until the visual review and CSV edits are complete.

## Experimental interpretation

- Compare the existing/short baseline against `v4_sfcn_baseline` retrained with the longer schedule and, if copied, `v4_sfcn_baseline_40ep`.
- Compare `v4_sfcn_deep_transfer` against `v4_sfcn_baseline` to test whether added depth helps while retaining transfer learning.
- Compare ResNet-34/50 against the pretrained SFCN variants. Their greater capacity can also increase overfitting; training MAE alone is not evidence of improvement.
- The canonical pipeline should be used for all model comparisons, with both QC-passed and all-eligible populations kept distinct. Prioritize held-out MAE/RMSE/R² and fold/seed consistency.
- The existing z-score ablation remains important because `v3 problems.txt` flags a possible intensity-convention mismatch. Do not choose input normalization from the zero-shot probe alone; compare trained models on identical subject-level folds.
- The `v3 problems.txt` file reports only 2–4 conversion events in prior longitudinal outputs. Better scan-level MAE does **not** automatically validate survival or clinical claims; keep those analyses exploratory unless event counts and uncertainty support stronger conclusions.

## Important limitations

No code can guarantee a lower MAE. A larger network and longer training are hypotheses to test, not guaranteed improvements. The dataset is only about 1,100 subjects, so validation performance and subject-level leakage controls matter more than raw model size. The extra residual blocks are initialized near identity so that adding them does not immediately disrupt the pretrained representation, but they still need empirical validation.
