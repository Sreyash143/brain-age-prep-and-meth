"""
core_v3.py -- model, losses, augmentation and data utilities shared by the v3 scripts.

The SFCN here reproduces the layout of the official UK Biobank release
(feature_extractor.conv_0..conv_5 = Conv3d + BatchNorm3d (+MaxPool) + ReLU; classifier = AvgPool(5,6,5) -> Dropout -> 1x1 Conv)
so the pretrained checkpoint loads with strict=True.  If strict loading fails, the error lists the mismatched keys --
fix the layer names here instead of silently training from scratch.
"""
import math, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset

# ----------------------------------------------------------------- model
OLD_N_BINS, OLD_MIN, OLD_STEP = 40, 42.0, 1.0       # UK Biobank pretraining bins: 42..82, 1-year


def _block(cin, cout, pool, k, p):
    mods = [nn.Conv3d(cin, cout, k, padding=p), nn.BatchNorm3d(cout)]
    if pool:
        mods.append(nn.MaxPool3d(2, stride=2))
    mods.append(nn.ReLU())
    return nn.Sequential(*mods)


class SFCN(nn.Module):
    CH = (32, 64, 128, 256, 256, 64)

    def __init__(self, n_bins=OLD_N_BINS, dropout=0.5):
        super().__init__()
        fe, cin = nn.Sequential(), 1
        for i, c in enumerate(self.CH):
            last = i == len(self.CH) - 1
            fe.add_module(f"conv_{i}", _block(cin, c, pool=not last, k=1 if last else 3, p=0 if last else 1))
            cin = c
        self.feature_extractor = fe
        cl = nn.Sequential()
        cl.add_module("average_pool", nn.AvgPool3d((5, 6, 5)))
        cl.add_module("dropout", nn.Dropout(dropout))
        cl.add_module("conv_6", nn.Conv3d(cin, n_bins, 1))
        self.classifier = cl

    def forward(self, x):
        x = self.classifier(self.feature_extractor(x))
        return F.log_softmax(x.flatten(1), dim=1)          # (B, n_bins) log-probabilities


def load_pretrained_sfcn(ckpt, n_bins, bin_min, bin_step=1.0, device="cpu"):
    """Load UKB weights, then expand the 40-bin head to the OASIS age range by copying the
    nearest pretrained bin's weights (bins outside 42-82 start from the edge bin, bias lowered)."""
    sd = torch.load(str(ckpt), map_location="cpu")
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    m = SFCN(OLD_N_BINS)
    try:
        m.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError("Pretrained checkpoint keys do not match core_v3.SFCN. Print sd.keys() and adapt "
                           "layer names. Original error:\n" + str(e))
    old = m.classifier.conv_6
    new = nn.Conv3d(old.in_channels, n_bins, 1)
    old_c = OLD_MIN + OLD_STEP / 2 + np.arange(OLD_N_BINS) * OLD_STEP
    new_c = bin_min + bin_step / 2 + np.arange(n_bins) * bin_step
    with torch.no_grad():
        for j, c in enumerate(new_c):
            i = int(np.clip(round((c - old_c[0]) / OLD_STEP), 0, OLD_N_BINS - 1))
            new.weight[j] = old.weight[i]
            new.bias[j] = old.bias[i] - (1.0 if (c < old_c[0] or c > old_c[-1]) else 0.0)
    m.classifier.conv_6 = new
    return m.to(device)


def bin_centers(bins, device="cpu"):
    return torch.tensor(bins["bin_min"] + bins["bin_step"] / 2 + np.arange(bins["n_bins"]) * bins["bin_step"],
                        dtype=torch.float32, device=device)


def expected_age(log_probs, centers):
    return (log_probs.float().exp() * centers).sum(1)


def soft_labels(ages, bins, device):
    """Gaussian soft labels over 1-year bins (CDF differences, as in the UKB reference code)."""
    edges = torch.tensor(bins["bin_min"] + np.arange(bins["n_bins"] + 1) * bins["bin_step"],
                         dtype=torch.float32, device=device)
    cdf = 0.5 * (1 + torch.erf((edges[None, :] - ages[:, None].float()) / (bins["sigma"] * math.sqrt(2))))
    p = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(1e-8)
    return p / p.sum(1, keepdim=True)


def kl_loss(log_probs, y_soft):
    return F.kl_div(log_probs.float(), y_soft, reduction="batchmean")


# ----------------------------------------------------------------- freezing / LLRD
BLOCKS = [f"feature_extractor.conv_{i}" for i in range(6)] + ["classifier.conv_6"]


def configure_trainable(model, unfreeze, lr_head, lr_backbone, llrd):
    """Returns AdamW param groups and the list of trainable block names.
    unfreeze='last': conv_5 + head train (v2-style).  unfreeze='all': every block trains with
    lr = lr_head * llrd**depth (depth 0 = head, 1 = conv_5, ... 6 = conv_0)."""
    groups, trainable = [], []
    for depth, name in enumerate(reversed(BLOCKS)):
        mod = model.get_submodule(name)
        if unfreeze == "last" and depth > 1:
            for p in mod.parameters():
                p.requires_grad = False
            continue
        lr = lr_head if depth == 0 else (lr_backbone if unfreeze == "last" else lr_head * (llrd ** depth))
        groups.append(dict(params=list(mod.parameters()), lr=lr, initial_lr=lr))
        trainable.append(name)
    return groups, trainable


def set_train_mode(model, trainable, bn_mode="train"):
    """Frozen blocks stay in eval mode (BatchNorm running stats must not drift with batch size 4-8)."""
    model.train()
    for name in BLOCKS:
        if name not in trainable:
            model.get_submodule(name).eval()
        elif bn_mode == "freeze":
            for m in model.get_submodule(name).modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()


# ----------------------------------------------------------------- input conventions & GPU augmentation
def to_input(z, mode="zscore", cv=0.35):
    """z: brain-only z-scored volume (background == 0).
    'meannorm' approximates the UKB convention x / mean(x over the whole volume) from a z-scored image:
        x = mu*(1 + CV*z),  mean(x over volume) = f*mu  (f = brain fraction)  =>  x/mean = (1 + CV*z)/f
    CV (brain-intensity coefficient of variation) is unknown -> tune it with 02_zero_shot_probe.py."""
    if mode == "zscore":
        return z
    mask = z != 0
    f = mask.float().flatten(1).mean(1).clamp_min(1e-3).view(-1, 1, 1, 1, 1)
    return torch.where(mask, ((1 + cv * z) / f).clamp_min(0), torch.zeros_like(z))


def gpu_augment(x, rot_deg=4.0, scale=0.04, shift_vox=3.0, p_aff=0.8, gain=0.10, noise=0.02):
    """Batch augmentation on GPU: small rotation/scale/shift (no flips), global gain, in-brain Gaussian noise."""
    B, dev = x.shape[0], x.device
    ang = (torch.rand(B, 3, device=dev) * 2 - 1) * math.radians(rot_deg)
    cx, cy, cz = torch.cos(ang).unbind(1)
    sx, sy, sz = torch.sin(ang).unbind(1)
    o, z0 = torch.ones(B, device=dev), torch.zeros(B, device=dev)
    Rx = torch.stack([o, z0, z0, z0, cx, -sx, z0, sx, cx], 1).view(B, 3, 3)
    Ry = torch.stack([cy, z0, sy, z0, o, z0, -sy, z0, cy], 1).view(B, 3, 3)
    Rz = torch.stack([cz, -sz, z0, sz, cz, z0, z0, z0, o], 1).view(B, 3, 3)
    s = 1 + (torch.rand(B, 1, 1, device=dev) * 2 - 1) * scale
    A = (Rz @ Ry @ Rx) * s
    size = torch.tensor(list(x.shape[2:])[::-1], device=dev, dtype=x.dtype)          # (W,H,D) order for affine_grid
    t = ((torch.rand(B, 3, device=dev) * 2 - 1) * shift_vox) * 2 / size
    theta = torch.cat([A, t.unsqueeze(2)], 2)
    ident = torch.eye(3, 4, device=dev).unsqueeze(0).expand(B, -1, -1)
    theta = torch.where(torch.rand(B, 1, 1, device=dev) < p_aff, theta, ident)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False),
                      mode="bilinear", padding_mode="zeros", align_corners=False)
    x = x * (1 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 - 1) * gain)
    return x + noise * torch.randn_like(x) * (x > 0)


def shift_volume(x, shift):
    return x if shift == (0, 0, 0) else torch.roll(x, shifts=shift, dims=(2, 3, 4))


# ----------------------------------------------------------------- data
class ScanStore:
    """Loads each npz once into RAM as float16 (~10 MB/scan) and is shared by every fold/variant in one process."""
    def __init__(self):
        self.arr = {}

    def add(self, paths, desc="Caching scans"):
        from tqdm import tqdm
        for p in tqdm([p for p in dict.fromkeys(paths) if p not in self.arr], desc=desc):
            with np.load(p) as z:
                self.arr[p] = z["data"].astype(np.float16)

    def get(self, p):
        return torch.from_numpy(self.arr[p].astype(np.float32)).unsqueeze(0)


class ScanDS(Dataset):
    def __init__(self, store, df):
        self.store, self.paths = store, df.file_path.tolist()
        self.age = df.age.to_numpy(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.store.get(self.paths[i]), torch.tensor(self.age[i])


class PairDS(Dataset):
    """Same-session repeat pairs (TRAINING subjects only) for the consistency loss."""
    def __init__(self, store, df):
        self.store, self.items = store, []
        for _, g in df.sort_values(["mri_label", "run"]).groupby("mri_label"):
            if len(g) >= 2:
                self.items.append((g.file_path.iloc[0], g.file_path.iloc[1], float(g.age.iloc[0])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, b, age = self.items[i]
        return torch.stack([self.store.get(a), self.store.get(b)]), torch.tensor(age, dtype=torch.float32)


def tail_weights(ages, width=5.0):
    """Inverse-sqrt-frequency weights per 5-year age bin (counters regression to the mean at the age extremes)."""
    b = np.floor(np.asarray(ages) / width).astype(int)
    cnt = {k: (b == k).sum() for k in np.unique(b)}
    w = np.array([1.0 / math.sqrt(cnt[k]) for k in b])
    return w / w.mean()


def load_bins(path):
    return json.load(open(path))
