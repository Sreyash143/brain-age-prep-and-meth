"""
02_zero_shot_probe.py   (Step 3.5 -- ~15-30 min on the GPU, NO training)

Question: does the UK-Biobank SFCN "see" your preprocessed volumes in the intensity convention it was trained on?
The UKB model expects x / mean(x) (brain-extracted, background 0, mean over the whole volume).
Your v2 tensors are brain-only z-scores (negative inside dark tissue, mean 0). With blocks 1-4 frozen in v2, the network
could not adapt to that mismatch. This probe runs the *pretrained, untouched* network on CN scans of fold 0 under
several conventions and reports how well it ranks and predicts age. No labels are used for training.

Pick the convention with the highest Pearson r / lowest corrected MAE, then set MEANNORM_CV in config_v3.py.
(Fold 0 is consumed by this single low-capacity decision; state it in the methods.)
"""
import argparse, json
import numpy as np, pandas as pd, torch
from scipy.stats import pearsonr, linregress
import config_v3 as C
import core_v3 as K

MODES = [("zscore", 0.0), ("meannorm", 0.25), ("meannorm", 0.30), ("meannorm", 0.35), ("meannorm", 0.40), ("meannorm", 0.50)]


@torch.no_grad()
def run(model, store, df, mode, cv, centers, dev, bs=4):
    preds = []
    for i in range(0, len(df), bs):
        x = torch.stack([store.get(p) for p in df.file_path.iloc[i:i + bs]]).squeeze(1).to(dev)
        x = K.to_input(x, mode, cv)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lp = model(x)
        preds.extend(K.expected_age(lp, centers).cpu().numpy().tolist())
    return np.array(preds)


def main(n_scans):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    man = pd.read_csv(C.MANIFEST_CSV)
    df = man[(man.fold == 0) & (man.CDRTOT == 0) & man.age.between(45, 80)]
    df = df.groupby("subject_id").head(1)                                   # one scan per subject
    df = df.sample(n=min(n_scans, len(df)), random_state=C.SEED).reset_index(drop=True)
    print("probe scans:", len(df), "| age range", df.age.min(), "-", df.age.max())

    store = K.ScanStore()
    store.add(df.file_path)
    model = K.load_pretrained_sfcn(C.SFCN_CKPT, K.OLD_N_BINS, K.OLD_MIN, K.OLD_STEP, device=dev).eval()
    centers = (K.OLD_MIN + K.OLD_STEP / 2 + torch.arange(K.OLD_N_BINS, dtype=torch.float32)).to(dev)

    rows = []
    for mode, cv in MODES:
        p = run(model, store, df, mode, cv, centers, dev)
        a = df.age.to_numpy()
        r = pearsonr(a, p)[0]
        s = linregress(a, p)
        corr = (p - s.intercept) / s.slope
        rows.append(dict(mode=mode, cv=cv, raw_MAE=np.abs(p - a).mean(), pearson_r=r, slope=s.slope,
                         MAE_after_linear_fit=np.abs(corr - a).mean()))
        print(rows[-1])
    out = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    print("\n", out.round(3).to_string(index=False))
    out.to_csv(C.OUT / "zero_shot_probe.csv", index=False)
    best = out.iloc[0]
    print(f"\nBest by Pearson r: mode={best['mode']} cv={best['cv']}.")
    print("If 'zscore' wins, set input_mode='zscore' for m1/m2 in config_v3.VARIANTS; "
          "otherwise set MEANNORM_CV to the winning cv.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scans", type=int, default=150)
    main(ap.parse_args().n_scans)
