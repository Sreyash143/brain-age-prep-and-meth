# Pre-registration (v3) -- written 2026-10-08 BEFORE training v3 models

NOTE: v2 test results were already seen. v3 design choices were therefore made on cross-validation /
validation data only; state this honestly in the paper.

1. Primary accuracy endpoint: out-of-fold MAE on CN scans (cross-fitted bias correction), with subject-cluster bootstrap CI.
2. Primary reliability endpoint: pooled within-session sigma_SEM and MDC, subject-cluster bootstrap CI (1000 resamples).
3. Fixed model set (do not change after seeing survival results): m0 (v2-style), m1 (v3 primary), m2 (m1 + consistency loss). Ensemble = seeds within fold.
4. Baseline window: first k=3 sessions, all CDR 0, >=1 later session. Minimum window span: ____ years (primary) / sensitivity ____.
5. Primary clinical endpoint rule: >=30 conversion events -> survival primary; otherwise continuous (CDR-SB/MMSE slope) primary.
6. Primary comparison: paired out-of-bag C-index difference, shrunk-slope vs raw-slope Cox model (95% bootstrap CI).
7. Covariates in floor model: baseline age, sex, baseline BAG.
8. Exclusion rules (image QC only, never error-based): ____
9. Known limitations to state: few converters, interval censoring, single cohort, affine-only registration.
