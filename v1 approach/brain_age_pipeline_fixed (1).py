#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
import torch; print(torch.__version__); print(torch.cuda.is_available())


# In[ ]:


# -*- coding: utf-8 -*-
"""
Fixed preprocessing pipeline for SFCN brain-age model.

Folder structure expected:

T1/
├── OAS30001_MR_d0129/
│   ├── anat2/
│   │   ├── sub-OAS30001_ses-d0129_run-1_T1w.json
│   │   └── sub-OAS30001_ses-d0129_run-1_T1w.nii.gz
│   └── anat3/
│       └── ...
│
├── OAS30001_MR_d0757/
│   └── anat2/
│       └── ...
│
└── ...

The batch preprocessing searches recursively through all
subdirectories inside T1.
"""

import os
import glob
import ants
import time
import antspynet
import numpy as np
import torchio as tio
import nibabel as nib


# =========================================================
# MNI152 brain template cache
# =========================================================

_MNI_BRAIN_CACHE = "mni152_brain_1mm.nii.gz"


def get_mni_brain_template():
    """
    Returns a skull-stripped 1mm MNI152 template.

    The template is generated once and cached to disk so that
    brain extraction does not need to be repeated for every scan.
    """

    if os.path.exists(_MNI_BRAIN_CACHE):
        return ants.image_read(_MNI_BRAIN_CACHE)

    print("Creating skull-stripped MNI152 template...")

    mni_head = ants.image_read(
        antspynet.get_antsxnet_data("mni152")
    )

    # brain_extraction() returns a probability image.
    mni_mask_prob = antspynet.brain_extraction(
        mni_head,
        modality="t1"
    )

    # Convert probability mask to binary mask.
    mni_mask_bin = ants.threshold_image(
        mni_mask_prob,
        0.5,
        1.0,
        1,
        0
    )

    mni_brain = ants.mask_image(
        mni_head,
        mni_mask_bin
    )

    ants.image_write(
        mni_brain,
        _MNI_BRAIN_CACHE
    )

    print(
        f"Saved MNI brain template to: "
        f"{_MNI_BRAIN_CACHE}"
    )

    return mni_brain


# =========================================================
# Single MRI preprocessing
# =========================================================

def run_preprocessing(
    input_nii_path: str,
    output_tensor_path: str
):
    """
    Executes the 6-stage preprocessing pipeline:

    1. Reorientation & FOV Cropping
    2. N4 Bias Field Correction
    3. Skull-Stripping
    4. Affine Registration to MNI152
    5. Voxel Resampling & Tensor Formatting
       (1mm isotropic, 160x192x160)
    6. Brain-masked Z-Score Normalization
    """

    print(
        f"\nProcessing: "
        f"{os.path.basename(input_nii_path)}"
    )

    # -------------------------------------------------------------
    # Step 1: Reorientation & FOV Cropping
    # -------------------------------------------------------------

    img = ants.image_read(
        input_nii_path
    )

    print(
        f"  Input intensity range: "
        f"[{img.min():.2f}, {img.max():.2f}]"
    )

    img = ants.reorient_image2(
        img,
        orientation="RAS"
    )

    img_cropped = ants.crop_image(
        img
    )


    # -------------------------------------------------------------
    # Step 2: B1/N4 Bias Field Correction
    # -------------------------------------------------------------

    img_n4 = ants.n4_bias_field_correction(
        img_cropped,
        shrink_factor=2
    )


    # -------------------------------------------------------------
    # Step 3: Brain Extraction / Skull Stripping
    # -------------------------------------------------------------

    mask_prob = antspynet.brain_extraction(
        img_n4,
        modality="t1"
    )

    # Convert probability mask to binary mask.
    mask_bin = ants.threshold_image(
        mask_prob,
        0.5,
        1.0,
        1,
        0
    )

    brain = ants.mask_image(
        img_n4,
        mask_bin
    )

    voxel_count = int(
        mask_bin.sum()
    )

    print(
        f"  Brain mask voxel count: "
        f"{voxel_count}"
    )

    if voxel_count < 100_000:

        raise RuntimeError(
            f"Brain extraction found almost no brain tissue "
            f"({voxel_count} voxels) "
            f"in {os.path.basename(input_nii_path)}. "
            f"This may indicate that the input is not a raw "
            f"T1w MRI scan."
        )


    # -------------------------------------------------------------
    # Step 4: Spatial Registration to 1mm MNI152 Template
    # -------------------------------------------------------------

    mni_brain = get_mni_brain_template()

    print(
        "  Registering brain to MNI152..."
    )

    reg = ants.registration(
        fixed=mni_brain,
        moving=brain,
        type_of_transform="Affine"
    )

    warped_brain = reg[
        "warpedmovout"
    ]


    # -------------------------------------------------------------
    # Step 5: Crop/Pad to exact SFCN matrix
    # -------------------------------------------------------------

    temp_nii = "_temp_reg.nii.gz"

    ants.image_write(
        warped_brain,
        temp_nii
    )

    subject = tio.Subject(
        mri=tio.ScalarImage(
            temp_nii
        )
    )

    crop_pad = tio.CropOrPad(
        (160, 192, 160)
    )

    processed_subject = crop_pad(
        subject
    )

    data = (
        processed_subject
        .mri
        .data
        .squeeze()
        .numpy()
    )

    final_affine = (
        processed_subject
        .mri
        .affine
    )

    # Remove temporary registered image.
    if os.path.exists(temp_nii):
        os.remove(temp_nii)


    # -------------------------------------------------------------
    # Step 6: Brain-Masked Z-Score Normalization
    # -------------------------------------------------------------

    brain_mask = data > 0

    standardized_brain = np.zeros_like(
        data,
        dtype=np.float32
    )

    if np.any(brain_mask):

        mu = np.mean(
            data[brain_mask]
        )

        sigma = np.std(
            data[brain_mask]
        )

        standardized_brain[brain_mask] = (
            data[brain_mask] - mu
        ) / (sigma + 1e-8)


    # -------------------------------------------------------------
    # Save NPZ
    # -------------------------------------------------------------

    output_directory = os.path.dirname(
        output_tensor_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    np.savez_compressed(
        output_tensor_path,
        data=standardized_brain
    )


    # -------------------------------------------------------------
    # Save NIfTI for Slicer
    # -------------------------------------------------------------

    slicer_nii_path = (
        output_tensor_path
        .replace(
            ".npz",
            ".nii.gz"
        )
    )

    nii_img = nib.Nifti1Image(
        standardized_brain,
        final_affine
    )

    nib.save(
        nii_img,
        slicer_nii_path
    )


    print(
        f"  Saved training tensor: "
        f"{output_tensor_path}"
    )

    print(
        f"  Saved Slicer volume: "
        f"{slicer_nii_path}"
    )


# =========================================================
# FIND ALL MRI FILES RECURSIVELY
# =========================================================

def find_t1_files(raw_dir: str):
    """
    Recursively searches for all NIfTI MRI files inside raw_dir.

    Example input structure:

    T1/
        OAS30001_MR_d0129/
            anat2/
                sub-OAS30001_ses-d0129_run-1_T1w.nii.gz

    The complete file paths are returned.
    """

    nii_files = []

    # Find .nii files
    nii_files.extend(
        glob.glob(
            os.path.join(
                raw_dir,
                "**",
                "*.nii"
            ),
            recursive=True
        )
    )

    # Find .nii.gz files
    nii_files.extend(
        glob.glob(
            os.path.join(
                raw_dir,
                "**",
                "*.nii.gz"
            ),
            recursive=True
        )
    )

    # Remove unwanted files
    nii_files = [
        f
        for f in nii_files
        if not f.endswith(".json")
        and "_preproc" not in os.path.basename(f)
    ]

    return sorted(nii_files)


# =========================================================
# BATCH PREPROCESSING
# =========================================================

def batch_preprocess_t1(
    raw_dir: str,
    output_base_dir: str
):
    """
    Preprocesses every T1w MRI found anywhere inside raw_dir.

    The original folder structure is preserved.

    Example:

    INPUT:

    T1/
    └── OAS30001_MR_d0129/
        └── anat2/
            └── sub-OAS30001_ses-d0129_run-1_T1w.nii.gz


    OUTPUT:

    T1 preprocessed/
    └── OAS30001_MR_d0129/
        └── anat2/
            ├── sub-OAS30001_ses-d0129_run-1_T1w_preproc.npz
            └── sub-OAS30001_ses-d0129_run-1_T1w_preproc.nii.gz
    """

    # ---------------------------------------------------------
    # Make sure output directory exists
    # ---------------------------------------------------------

    os.makedirs(
        output_base_dir,
        exist_ok=True
    )


    # ---------------------------------------------------------
    # Find all MRI files recursively
    # ---------------------------------------------------------

    raw_files = find_t1_files(
        raw_dir
    )


    print(
        "\n=============================================="
    )

    print(
        "T1 BATCH PREPROCESSING"
    )

    print(
        "=============================================="
    )

    print(
        f"Input directory:"
    )

    print(
        f"  {os.path.abspath(raw_dir)}"
    )

    print(
        f"\nOutput directory:"
    )

    print(
        f"  {os.path.abspath(output_base_dir)}"
    )

    print(
        f"\nFound {len(raw_files)} MRI file(s)"
    )

    print(
        "==============================================\n"
    )


    # ---------------------------------------------------------
    # No files found
    # ---------------------------------------------------------

    if len(raw_files) == 0:

        print(
            "ERROR: No NIfTI MRI files were found."
        )

        print(
            f"Searched recursively inside:"
        )

        print(
            os.path.abspath(raw_dir)
        )

        return []


    # ---------------------------------------------------------
    # Process every MRI
    # ---------------------------------------------------------

    failures = []

    for index, input_file in enumerate(
        raw_files,
        start=1
    ):

        filename = os.path.basename(
            input_file
        )


        # -----------------------------------------------------
        # Calculate the path relative to T1
        # -----------------------------------------------------

        relative_path = os.path.relpath(
            input_file,
            raw_dir
        )


        # -----------------------------------------------------
        # Remove MRI extension
        # -----------------------------------------------------

        if relative_path.endswith(
            ".nii.gz"
        ):

            relative_without_extension = (
                relative_path[:-7]
            )

        elif relative_path.endswith(
            ".nii"
        ):

            relative_without_extension = (
                relative_path[:-4]
            )

        else:

            print(
                f"Skipping unsupported file: "
                f"{input_file}"
            )

            continue


        # -----------------------------------------------------
        # Construct output path
        #
        # This preserves the original folder structure.
        # -----------------------------------------------------

        output_relative_path = (
            relative_without_extension
            + "_preproc.npz"
        )


        output_file = os.path.join(
            output_base_dir,
            output_relative_path
        )


        # -----------------------------------------------------
        # Create corresponding output folder
        # -----------------------------------------------------

        output_folder = os.path.dirname(
            output_file
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )


        # -----------------------------------------------------
        # Skip already processed files
        # -----------------------------------------------------

        if os.path.exists(
            output_file
        ):

            print(
                f"\n[{index}/{len(raw_files)}] "
                f"SKIPPING"
            )

            print(
                f"  Already exists:"
            )

            print(
                f"  {output_file}"
            )

            continue


        # -----------------------------------------------------
        # Display current file
        # -----------------------------------------------------

        print(
            "\n=============================================="
        )

        print(
            f"[{index}/{len(raw_files)}]"
        )

        print(
            "Input:"
        )

        print(
            f"  {input_file}"
        )

        print(
            "Output:"
        )

        print(
            f"  {output_file}"
        )

        print(
            "=============================================="
        )


        # -----------------------------------------------------
        # Run preprocessing
        # -----------------------------------------------------

        try:

            # Start timer for this MRI
            start_time = time.perf_counter()

            run_preprocessing(
                input_file,
                output_file
            )

            # Calculate elapsed time
            elapsed_time = time.perf_counter() - start_time

            print(
                f"\n  Preprocessing time: "
                f"{elapsed_time:.2f} seconds"
            )

            print(
                f"  Preprocessing time: "
                f"{elapsed_time / 60:.2f} minutes"
            )

        except Exception as e:

            print(
                "\nFAILED"
            )

            print(
                f"File:"
            )

            print(
                f"  {input_file}"
            )

            print(
                f"Error:"
            )

            print(
                f"  {e}"
            )

            failures.append(
                input_file
            )


    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print(
        "\n\n=============================================="
    )

    print(
        "BATCH PREPROCESSING COMPLETE"
    )

    print(
        "=============================================="
    )

    print(
        f"Total MRI files : {len(raw_files)}"
    )

    print(
        f"Successful      : "
        f"{len(raw_files) - len(failures)}"
    )

    print(
        f"Failed          : {len(failures)}"
    )

    print(
        f"Output location : "
        f"{os.path.abspath(output_base_dir)}"
    )

    print(
        "=============================================="
    )


    # ---------------------------------------------------------
    # Display failures
    # ---------------------------------------------------------

    if failures:

        print(
            "\nFailed files:"
        )

        for f in failures:

            print(
                f"  {f}"
            )

    else:

        print(
            "\nNo preprocessing failures."
        )


    return failures


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print(
        "Starting batch preprocessing on T1 folder..."
    )

    batch_preprocess_t1(
        raw_dir="/mnt/d/brain age prep and meth/T1",
        output_base_dir="/mnt/d/brain age prep and meth/T1 preprocessed"
    )


# ## CELL 4 — Run batch preprocessing
# This cell executes the existing 6-step preprocessing pipeline over the full `T1` directory.
# It skips files that already have a `_preproc.npz` output, so it is safe to rerun after interruption.
# 

# In[ ]:


# =========================================================
# CELL 4 — VERIFY COMPLETED HPC PREPROCESSING
# =========================================================
# IMPORTANT:
# The 6-step preprocessing was already completed on the HPC lab
# for the full dataset. DO NOT rerun preprocessing here.
#
# This cell only verifies that the expected preprocessed NPZ files
# are available to the training pipeline.

import os
import glob
import numpy as np

# FIXED: Linux path for WSL/HPC compatibility
PREPROC_DIR = "/mnt/d/brain age prep and meth/T1 preprocessed"

preproc_files = sorted(
    glob.glob(
        os.path.join(PREPROC_DIR, "**", "*_preproc.npz"),
        recursive=True
    )
)

print("=" * 70)
print("CELL 4: HPC PREPROCESSING VERIFICATION")
print("=" * 70)
print(f"Preprocessed NPZ files found : {len(preproc_files)}")
print(f"Expected completed files     : 3340")
print(f"Directory                    : {PREPROC_DIR}")

if len(preproc_files) == 3340:
    print("STATUS: 3340 preprocessed files detected.")
elif len(preproc_files) > 0:
    print(f"STATUS: {len(preproc_files)} files detected.")
    print("Check the HPC-to-local transfer/path before training.")
else:
    raise FileNotFoundError(
        f"No *_preproc.npz files found under {PREPROC_DIR}"
    )

# Only inspect a small sample here; full QC is Cell 5.
sample = preproc_files[:min(5, len(preproc_files))]
for p in sample:
    with np.load(p) as z:
        data = z["data"]
    print(f"{os.path.basename(p)} -> shape={data.shape}, dtype={data.dtype}")

print("\nNo preprocessing was executed in this cell.")


# ## CELL 5 — Preprocessing QC
# Checks the actual saved tensors before model training: shape, finite values, non-zero brain voxels, and normalization.
# 

# In[ ]:


# =========================================================
# CELL 5 — PREPROCESSING QC
# =========================================================
import os
import glob
import numpy as np

preproc_files = sorted(
    glob.glob(os.path.join(PREPROC_DIR, "**", "*_preproc.npz"), recursive=True)
)

print("=" * 60)
print("CELL 5: PREPROCESSING QC")
print("=" * 60)
print(f"Preprocessed files: {len(preproc_files)}")

shape_counts = {}
bad_files = []
voxel_means = []
voxel_stds = []
brain_voxels = []

for path in preproc_files:
    try:
        with np.load(path) as z:
            data = z["data"]

        shape_counts[data.shape] = shape_counts.get(data.shape, 0) + 1

        if not np.isfinite(data).all():
            bad_files.append((path, "NaN/Inf"))
            continue

        mask = data != 0
        n_brain = int(mask.sum())
        brain_voxels.append(n_brain)

        if n_brain == 0:
            bad_files.append((path, "empty brain"))
            continue

        voxel_means.append(float(data[mask].mean()))
        voxel_stds.append(float(data[mask].std()))

    except Exception as e:
        bad_files.append((path, str(e)))

print("Shapes:")
for shape, count in sorted(shape_counts.items(), key=lambda x: str(x[0])):
    print(f"  {shape}: {count}")

if brain_voxels:
    print(f"Brain voxels: min={min(brain_voxels):,}, max={max(brain_voxels):,}")
    print(f"Normalized mean: {np.mean(voxel_means):.4f}")
    print(f"Normalized std : {np.mean(voxel_stds):.4f}")

print(f"Bad files: {len(bad_files)}")
for item in bad_files[:10]:
    print(" ", item)

expected_shape = (160, 192, 160)
valid_shape_count = shape_counts.get(expected_shape, 0)

print(f"\nExpected shape {expected_shape}: {valid_shape_count}/{len(preproc_files)}")

if bad_files:
    print("WARNING: remove/fix bad files before training.")
elif valid_shape_count != len(preproc_files):
    print("WARNING: not every file has the expected 160x192x160 shape.")
else:
    print("QC PASS: all saved tensors have the expected shape and finite values.")


# ## CELL 6 — SFCN architecture
# Defines the same 3D SFCN architecture used by the training script and instantiates one model to verify the forward pass.
# 

# In[ ]:


# =========================================================
# CELL 6 — SFCN ARCHITECTURE (HPC STABILIZED)
# =========================================================
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # Replaced BatchNorm3d with InstanceNorm3d to prevent gradient instability at small batch sizes
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if max_pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SFCN(nn.Module):
    def __init__(self, output_dim=1, dropout_p=0.5):
        super().__init__()
        self.block1 = ConvBlock(1, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.block5 = ConvBlock(256, 256)
        self.block6 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1, bias=False),
            nn.InstanceNorm3d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


model_check = SFCN().to(device)
n_params = sum(p.numel() for p in model_check.parameters())

print("=" * 60)
print("CELL 6: SFCN ARCHITECTURE (HPC STABILIZED)")
print("=" * 60)
print(f"Device          : {device}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
print(f"Parameters      : {n_params:,}")

# Forward-pass sanity check
with torch.no_grad():
    dummy = torch.zeros((1, 1, 160, 192, 160), device=device)
    out = model_check(dummy)

print(f"Input shape     : {tuple(dummy.shape)}")
print(f"Output shape    : {tuple(out.shape)}")
print("Forward pass    : PASS")

del model_check, dummy, out
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ## CELL 7 — Dataset
# Defines the dataset loader and tests one real preprocessed sample.
# 

# In[ ]:


# =========================================================
# CELL 7 — DATASET (HPC OPTIMIZED WITH AUGMENTATION)
# =========================================================
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio

class BrainDataset(Dataset):
    def __init__(self, df, is_train=False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        # 3D Augmentation: Critical to prevent overfitting on 3K scans
        self.transform = tio.Compose([
            tio.RandomAffine(scales=(0.95, 1.05), degrees=5, p=0.5),
            tio.RandomNoise(std=0.05, p=0.3),
        ]) if is_train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read directly from disk to avoid multiprocessing memory duplication
        with np.load(self.df.iloc[idx]["file_path"]) as z:
            data = z["data"].astype(np.float32)

        # Add channel dimension -> (1, 160, 192, 160)
        tensor = torch.from_numpy(data).unsqueeze(0)

        # Apply augmentations (only on training scans)
        if self.transform:
            tensor = self.transform(tensor)

        age = torch.tensor(float(self.df.iloc[idx]["age"]), dtype=torch.float32)
        return tensor, age

print("=" * 60)
print("CELL 7: DATASET WITH AUGMENTATIONS")
print("=" * 60)
print("BrainDataset defined.")


# ## CELL 8 — Manifest + metadata
# Builds one row per preprocessed scan, joins scan metadata and CDR/MMSE data, and keeps chronological age from CDR when available.
# 

# In[ ]:


# =========================================================
# CELL 8 — MANIFEST + METADATA (LINUX PATHS)
# =========================================================
import os
import re
import glob
import zipfile
import pandas as pd

# UPDATED TO LINUX PATHS FOR WSL/HPC compatibility
SCAN_META_CSV = "/mnt/d/brain age prep and meth/okarun143_8_17_2026_12_36_37.csv"
CDR_CSV = "/mnt/d/brain age prep and meth/OASIS3_UDSb4_cdr_final.csv"

def parse_preproc_filename(path):
    fname = os.path.basename(path)
    m = re.search(r"sub-(OAS\d+)_sess?-d(\d+)(?:_run-(\d+))?", fname)
    if not m:
        raise ValueError(f"Could not parse subject/session/run from filename: {fname}")

    subject_id, day_str, run = m.group(1), m.group(2), m.group(3)
    run = run or "01"

    return {
        "subject_id": subject_id,
        "day": int(day_str),
        "run": run,
        "mri_label": f"{subject_id}_MR_d{day_str}",
        "file_path": path,
    }

def build_manifest(preproc_dir, scan_meta_csv, cdr_csv):
    files = sorted(
        glob.glob(os.path.join(preproc_dir, "**", "*_preproc.npz"), recursive=True)
    )
    if not files:
        raise FileNotFoundError(f"No preprocessed .npz files found under {preproc_dir}")

    valid_files = [f for f in files if zipfile.is_zipfile(f)]
    corrupted = [f for f in files if not zipfile.is_zipfile(f)]

    if corrupted:
        print(f"Skipping corrupted NPZ files: {len(corrupted)}")
        for f in corrupted[:10]:
            print(" ", f)

    rows = []
    parse_failures = []

    for f in valid_files:
        try:
            rows.append(parse_preproc_filename(f))
        except ValueError as e:
            parse_failures.append(str(e))

    if parse_failures:
        print(f"Filename parsing failures: {len(parse_failures)}")
        for msg in parse_failures[:10]:
            print(" ", msg)

    manifest = pd.DataFrame(rows)

    scan_meta = (
        pd.read_csv(scan_meta_csv)[["Label", "Age", "M/F"]]
        .rename(columns={"Label": "mri_label", "Age": "age_meta", "M/F": "sex"})
    )

    cdr = (
        pd.read_csv(cdr_csv)[["MRI Label", "age at visit", "CDRTOT", "CDRSUM", "MMSE"]]
        .rename(columns={"MRI Label": "mri_label", "age at visit": "age_cdr"})
    )

    # Prevent duplicate metadata columns
    scan_meta = scan_meta.drop_duplicates("mri_label")
    cdr = cdr.drop_duplicates("mri_label")

    manifest = manifest.merge(scan_meta, on="mri_label", how="left")
    manifest = manifest.merge(cdr, on="mri_label", how="left")

    manifest["age"] = pd.to_numeric(
        manifest["age_cdr"], errors="coerce"
    ).fillna(
        pd.to_numeric(manifest["age_meta"], errors="coerce")
    )

    manifest = manifest.drop(columns=["age_meta", "age_cdr"])
    manifest = manifest.dropna(subset=["age"]).reset_index(drop=True)

    return manifest

manifest = build_manifest(PREPROC_DIR, SCAN_META_CSV, CDR_CSV)

print("=" * 60)
print("CELL 8: MANIFEST + METADATA")
print("=" * 60)
print(f"Scans       : {len(manifest)}")
print(f"Subjects    : {manifest['subject_id'].nunique()}")
print(f"Age range   : {manifest['age'].min():.2f} - {manifest['age'].max():.2f}")
print(f"CDR 0.0     : {(manifest['CDRTOT'] == 0.0).sum()}")
print(f"Columns     : {list(manifest.columns)}")

display(manifest.head())


# ## CELL 9 — Train / Validation / Test split
# Splits at the **subject level**, so scans from the same person cannot appear in multiple splits.
# 

# In[ ]:


# =========================================================
# CELL 9 — SUBJECT-LEVEL TRAIN / VAL / TEST SPLIT
# =========================================================
from sklearn.model_selection import train_test_split

def subject_level_split(manifest, test_size=0.15, val_size=0.15, random_state=42):
    subjects = manifest["subject_id"].unique()

    train_subj, test_subj = train_test_split(
        subjects, test_size=test_size, random_state=random_state
    )

    train_subj, val_subj = train_test_split(
        train_subj,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )

    train_df = manifest[manifest["subject_id"].isin(train_subj)].reset_index(drop=True)
    val_df = manifest[manifest["subject_id"].isin(val_subj)].reset_index(drop=True)
    test_df = manifest[manifest["subject_id"].isin(test_subj)].reset_index(drop=True)

    assert not (set(train_df["subject_id"]) & set(val_df["subject_id"]))
    assert not (set(train_df["subject_id"]) & set(test_df["subject_id"]))
    assert not (set(val_df["subject_id"]) & set(test_df["subject_id"]))

    return train_df, val_df, test_df


train_df, val_df, test_df = subject_level_split(manifest)

# Training/validation are restricted to cognitively normal scans.
train_cn = train_df[train_df["CDRTOT"] == 0.0].reset_index(drop=True)
val_cn = val_df[val_df["CDRTOT"] == 0.0].reset_index(drop=True)

print("=" * 60)
print("CELL 9: TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)
print(f"Train subjects : {train_df['subject_id'].nunique()} | scans: {len(train_df)}")
print(f"Val subjects   : {val_df['subject_id'].nunique()} | scans: {len(val_df)}")
print(f"Test subjects  : {test_df['subject_id'].nunique()} | scans: {len(test_df)}")
print(f"Train CN       : {len(train_cn)}")
print(f"Val CN         : {len(val_cn)}")
print(f"Test CN        : {(test_df['CDRTOT'] == 0.0).sum()}")

if len(train_cn) == 0 or len(val_cn) == 0:
    raise RuntimeError("No CDR 0.0 scans available for training/validation.")


# ## CELL 10 — DataLoaders
# Creates loaders with `batch_size=2` because the MRI tensor is 160×192×160.
# 

# In[ ]:


# =========================================================
# CELL 10 — DATALOADERS (HPC OPTIMIZED - WSL STABLE)
# =========================================================
import os
import torch
import numpy as np
import torchio as tio
from torch.utils.data import DataLoader, Dataset

class BrainDataset(Dataset):
    def __init__(self, df, is_train=False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

        self.transform = tio.Compose([
            tio.RandomAffine(scales=(0.95, 1.05), degrees=5, p=0.5),
            tio.RandomNoise(std=0.05, p=0.3),
        ]) if is_train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        with np.load(self.df.iloc[idx]["file_path"]) as z:
            data = z["data"].astype(np.float32)

        tensor = torch.from_numpy(data).unsqueeze(0)

        if self.transform:
            tensor = self.transform(tensor)

        age = torch.tensor(float(self.df.iloc[idx]["age"]), dtype=torch.float32)
        return tensor, age

# ---------------------------------------------------------
# Loader Configuration
# ---------------------------------------------------------
BATCH_SIZE = 4  
NUM_WORKERS = 0  # Forces WSL to use the main thread, preventing random kernel crashes

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": torch.cuda.is_available(),
    "persistent_workers": False, # Must be False when NUM_WORKERS is 0
    "prefetch_factor": None,     # Must be None when NUM_WORKERS is 0
}

train_loader = DataLoader(BrainDataset(train_cn, is_train=True), shuffle=True, **loader_kwargs)
val_loader = DataLoader(BrainDataset(val_cn, is_train=False), shuffle=False, **loader_kwargs)
test_loader = DataLoader(BrainDataset(test_df, is_train=False), shuffle=False, **loader_kwargs)

print("=" * 70)
print("CELL 10: DATALOADERS — HPC/LAB CONFIGURATION")
print("=" * 70)
print(f"CPU workers    : {NUM_WORKERS}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"Train batches  : {len(train_loader)}")

x0, y0 = next(iter(train_loader))
print(f"Sample input   : {tuple(x0.shape)}")
print(f"Sample age     : {float(y0[0].item()):.2f}")

del x0, y0
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ## CELL 11 — Train SFCN
# Trains an ensemble of 3 SFCN models with L1 loss, Adam, early stopping, and checkpointing.
# 

# In[ ]:


# =========================================================
# CELL 11 — TRAIN SFCN ENSEMBLE
# =========================================================
import os
import time
import numpy as np
import torch
import torch.nn as nn

N_MODELS = 3
MAX_EPOCHS_PER_MODEL = 40
PATIENCE = 6
LR = 1e-4

# Corrected Linux Paths
RESULTS_DIR = "/mnt/d/brain age prep and meth/results_notebook"
CHECKPOINT_DIR = "/mnt/d/brain age prep and meth/checkpoints_notebook"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

training_start = time.time()

USE_AMP = torch.cuda.is_available()
amp_dtype = torch.float16

print("=" * 70)
print("CELL 11: SFCN ENSEMBLE TRAINING")
print("=" * 70)
print(f"Device        : {device}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory    : {props.total_memory / 1024**3:.2f} GB")
print(f"Models        : {N_MODELS}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"CPU workers   : {NUM_WORKERS}")
print(f"AMP enabled   : {USE_AMP}")

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    n = 0

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP) if train else None
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            try:
                if USE_AMP:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                        preds = model(inputs).view_as(targets)
                        loss = criterion(preds, targets)
                else:
                    preds = model(inputs).view_as(targets)
                    loss = criterion(preds, targets)

                if train:
                    if USE_AMP:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("CUDA OOM: skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

            bs = inputs.size(0)
            total_loss += float(loss.detach().item()) * bs
            total_abs_error += float(torch.abs(preds.detach().float() - targets).sum().item())
            n += bs

    if n == 0:
        return float("nan"), float("nan")

    return total_loss / n, total_abs_error / n

history = {}

def train_one_model(model_idx):
    best_path = os.path.join(CHECKPOINT_DIR, f"model_{model_idx}_best.pt")

    torch.manual_seed(model_idx)
    np.random.seed(model_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_idx)

    model = SFCN().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ADDED: Learning Rate Scheduler for fine-tuned convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    local_history = {"train_mae": [], "val_mae": []}

    best_val = float("inf")
    patience_counter = 0

    print("\n" + "=" * 70)
    print(f"MODEL {model_idx + 1}/{N_MODELS}")
    print("=" * 70)

    for epoch in range(MAX_EPOCHS_PER_MODEL):
        t0 = time.time()

        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        # Step the scheduler based on validation MAE
        scheduler.step(val_mae)

        local_history["train_mae"].append(train_mae)
        local_history["val_mae"].append(val_mae)

        print(
            f"Model {model_idx + 1} | Epoch {epoch + 1:02d}/{MAX_EPOCHS_PER_MODEL} | "
            f"Train MAE={train_mae:.2f} | Val MAE={val_mae:.2f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f} | {time.time() - t0:.1f}s"
        )

        if val_mae < best_val:
            best_val = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} (best Val MAE={best_val:.2f})")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    history[model_idx] = local_history
    print(f"Model {model_idx + 1} complete | Best validation MAE={best_val:.3f} years")

    return model

models = []
for i in range(N_MODELS):
    models.append(train_one_model(i))

print("\n" + "=" * 70)
print("CELL 11: ALL MODELS TRAINED")
print("=" * 70)
print(f"Models trained: {len(models)}/{N_MODELS}")

for i, h in history.items():
    if h["val_mae"]:
        print(f"Model {i + 1} best Val MAE: {min(h['val_mae']):.3f} years")


# ## CELL 12 — Validation results
# Evaluates each trained model and the ensemble on the validation set.
# 

# In[ ]:


# =========================================================
# CELL 12 — VALIDATION RESULTS
# =========================================================
import numpy as np
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def predict_df(model, df):
    # Updated to ensure is_train=False matches BrainDataset signature
    loader = DataLoader(
        BrainDataset(df, is_train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    preds = []
    actual = []

    model.eval()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        p = model(inputs).detach().cpu().numpy().reshape(-1)
        preds.extend(p.tolist())
        actual.extend(targets.numpy().reshape(-1).tolist())

    return np.asarray(actual, dtype=float), np.asarray(preds, dtype=float)

val_model_preds = []

for i, model in enumerate(models):
    y_val, p_val = predict_df(model, val_cn)
    val_model_preds.append(p_val)

    mae = float(np.mean(np.abs(p_val - y_val)))
    rmse = float(np.sqrt(np.mean((p_val - y_val) ** 2)))

    print(f"Model {i + 1}: MAE={mae:.3f} | RMSE={rmse:.3f}")

val_model_preds = np.vstack(val_model_preds)
val_ensemble_pred = val_model_preds.mean(axis=0)

val_mae = float(np.mean(np.abs(val_ensemble_pred - y_val)))
val_rmse = float(np.sqrt(np.mean((val_ensemble_pred - y_val) ** 2)))

print("-" * 60)
print(f"Ensemble validation MAE : {val_mae:.3f} years")
print(f"Ensemble validation RMSE: {val_rmse:.3f} years")


# ## CELL 13 — Test predictions
# Generates ensemble predictions for the held-out test subjects. No test labels are used to train the model or fit the bias correction.
# 

# In[ ]:


# =========================================================
# CELL 13 — TEST PREDICTIONS
# =========================================================
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def ensemble_predict(models, df):
    loader = DataLoader(
        BrainDataset(df, is_train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    all_model_preds = []

    for model in models:
        model.eval()
        preds = []

        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            preds.extend(model(inputs).detach().cpu().numpy().reshape(-1).tolist())

        all_model_preds.append(np.asarray(preds, dtype=float))

    stacked = np.vstack(all_model_preds)

    out = df.copy()
    out["predicted_age"] = stacked.mean(axis=0)
    out["predicted_age_sd"] = stacked.std(axis=0)

    return out, stacked

test_predictions, test_model_preds = ensemble_predict(models, test_df)

test_predictions.to_csv(
    os.path.join(RESULTS_DIR, "test_predictions_raw.csv"),
    index=False
)

print("=" * 60)
print("CELL 13: TEST PREDICTIONS")
print("=" * 60)
print(f"Test scans       : {len(test_predictions)}")
print(f"Models in ensemble: {len(models)}")
print(f"Prediction SD mean: {test_predictions['predicted_age_sd'].mean():.3f}")
display(test_predictions.head())


# ## CELL 14 — Brain Age + BAG
# Fits the age-prediction bias correction on **training CDR 0.0 scans only**, then applies it to the held-out test predictions.
# 

# In[ ]:


# =========================================================
# CELL 14 — BRAIN AGE + BAG
# =========================================================
from scipy.stats import linregress
import numpy as np
import pandas as pd

# Generate training predictions for fitting the correction.
train_predictions, _ = ensemble_predict(models, train_cn)

x = train_predictions["age"].to_numpy(dtype=float)
y = train_predictions["predicted_age"].to_numpy(dtype=float)

slope, intercept, r_value, p_value, std_err = linregress(x, y)

if abs(slope) < 1e-12:
    raise RuntimeError("Bias-correction slope is too close to zero.")

test_predictions["brain_age"] = (
    test_predictions["predicted_age"] - intercept
) / slope

test_predictions["BAG"] = (
    test_predictions["brain_age"] - test_predictions["age"]
)

test_predictions.to_csv(
    os.path.join(RESULTS_DIR, "brainage_predictions.csv"),
    index=False
)

print("=" * 60)
print("CELL 14: BRAIN AGE + BAG")
print("=" * 60)
print(f"Bias correction slope  : {float(slope):.4f}")
print(f"Bias correction intercept: {float(intercept):.4f}")
print(f"Training r             : {float(r_value):.4f}")
print(f"Training n             : {len(train_predictions)}")
print(f"Mean test BAG          : {test_predictions['BAG'].mean():.3f} years")
print(f"SD test BAG            : {test_predictions['BAG'].std(ddof=1):.3f} years")
display(test_predictions[["subject_id", "mri_label", "age", "predicted_age", "brain_age", "BAG"]].head())


# ## CELL 15 — Confusion matrix
# Because SFCN is a regression model, this is a **derived age-group classification analysis**, not the model's training objective.
# 
# Age groups:
# - Young: `< 60`
# - Middle-aged: `60–74`
# - Older: `>= 75`
# 

# In[ ]:


# =========================================================
# CELL 15 — DERIVED AGE-GROUP CONFUSION MATRIX
# =========================================================
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

AGE_BINS = [-np.inf, 60, 75, np.inf]
AGE_LABELS = ["Young (<60)", "Middle (60-74)", "Older (>=75)"]

cm_df = test_predictions.copy()

cm_df["true_group"] = pd.cut(
    cm_df["age"],
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=False,
)

cm_df["pred_group"] = pd.cut(
    cm_df["brain_age"],
    bins=AGE_BINS,
    labels=AGE_LABELS,
    right=False,
)

valid = cm_df.dropna(subset=["true_group", "pred_group"]).copy()

y_true_group = valid["true_group"].astype(str)
y_pred_group = valid["pred_group"].astype(str)

cm = confusion_matrix(
    y_true_group,
    y_pred_group,
    labels=AGE_LABELS
)

cm_table = pd.DataFrame(
    cm,
    index=[f"Actual: {x}" for x in AGE_LABELS],
    columns=[f"Predicted: {x}" for x in AGE_LABELS],
)

accuracy = float(accuracy_score(y_true_group, y_pred_group))

# Fixed typo from original logic where 'recall_recall_fscore_support' was called
precision, recall, f1, support = precision_recall_fscore_support(
    y_true_group,
    y_pred_group,
    labels=AGE_LABELS,
    zero_division=0,
)

print("=" * 60)
print("CELL 15: DERIVED AGE-GROUP CONFUSION MATRIX")
print("=" * 60)
display(cm_table)

print(f"Derived classification accuracy : {accuracy:.3f}")
print(f"Weighted precision              : {precision_recall_fscore_support(y_true_group, y_pred_group, labels=AGE_LABELS, average='weighted', zero_division=0)[0]:.3f}")
print(f"Weighted recall                 : {precision_recall_fscore_support(y_true_group, y_pred_group, labels=AGE_LABELS, average='weighted', zero_division=0)[1]:.3f}")
print(f"Weighted F1                     : {f1.mean():.3f}  [macro F1]")

for label, p, r, f, s in zip(AGE_LABELS, precision, recall, f1, support):
    print(f"{label:16s} | Precision={p:.3f} Recall={r:.3f} F1={f:.3f} Support={s}")


# ## CELL 16 — Measurement noise
# Uses same-day repeat scans with the same `mri_label` and computes the repeat-difference noise floor:
# `σ_SEM = SD(differences) / √2`
# and
# `MDC95 = 1.96 × √2 × σ_SEM`.
# 

# In[ ]:


# =========================================================
# CELL 16 — SAME-DAY REPEAT MEASUREMENT NOISE
# =========================================================
import numpy as np
import pandas as pd

def build_repeat_pairs(pred_df):
    rows = []

    for mri_label, group in pred_df.groupby("mri_label"):
        group = group.sort_values("run")

        if len(group) < 2:
            continue

        r1, r2 = group.iloc[0], group.iloc[1]

        rows.append({
            "subject_id": r1["subject_id"],
            "mri_label": mri_label,
            "BAG_run1": float(r1["BAG"]),
            "BAG_run2": float(r2["BAG"]),
            "CDRTOT": r1["CDRTOT"],
        })

    return pd.DataFrame(rows)


# For the noise analysis, use the held-out test predictions.
repeat_pairs = build_repeat_pairs(test_predictions)

print("=" * 60)
print("CELL 16: MEASUREMENT NOISE")
print("=" * 60)
print(f"Same-day repeat pairs: {len(repeat_pairs)}")

if len(repeat_pairs) >= 2:
    differences = (
        repeat_pairs["BAG_run1"].to_numpy(dtype=float)
        - repeat_pairs["BAG_run2"].to_numpy(dtype=float)
    )

    sd_difference = float(np.std(differences, ddof=1))
    sigma_sem = float(sd_difference / np.sqrt(2))
    mdc95 = float(1.96 * np.sqrt(2) * sigma_sem)

    print(f"SD of repeat differences : {sd_difference:.3f} years")
    print(f"Sigma_SEM                : {sigma_sem:.3f} years")
    print(f"MDC (95%)                : {mdc95:.3f} years")

    repeat_pairs.to_csv(
        os.path.join(RESULTS_DIR, "repeat_pairs.csv"),
        index=False
    )

else:
    sigma_sem = np.nan
    mdc95 = np.nan
    print("Not enough repeat pairs to estimate a stable noise floor.")


# ## CELL 17 — Final statistics
# Prints the main regression, BAG, ensemble, derived classification, and measurement-noise results in one place.
# 

# In[ ]:


# =========================================================
# CELL 17 — FINAL STATISTICS
# =========================================================
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Primary test performance: cognitively normal test scans only.
test_cn_pred = test_predictions[test_predictions["CDRTOT"] == 0.0].copy()

if len(test_cn_pred) > 0:
    y_true = test_cn_pred["age"].to_numpy(dtype=float)
    y_pred = test_cn_pred["brain_age"].to_numpy(dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    if len(y_true) >= 2:
        pearson_r = float(pearsonr(y_true, y_pred)[0])
    else:
        pearson_r = np.nan
else:
    mae = rmse = r2 = pearson_r = np.nan

raw_mae = float(
    mean_absolute_error(
        test_predictions["age"].to_numpy(dtype=float),
        test_predictions["predicted_age"].to_numpy(dtype=float),
    )
)

bag_mean = float(test_predictions["BAG"].mean())
bag_sd = float(test_predictions["BAG"].std(ddof=1))
ensemble_sd_mean = float(test_predictions["predicted_age_sd"].mean())

print("\n" + "=" * 70)
print("CELL 17: FINAL BRAIN-AGE RESULTS")
print("=" * 70)

print("\nDATA")
print(f"Total scans in manifest       : {len(manifest)}")
print(f"Total subjects                : {manifest['subject_id'].nunique()}")
print(f"Train / Val / Test scans      : {len(train_df)} / {len(val_df)} / {len(test_df)}")
print(f"Test CN scans                 : {len(test_cn_pred)}")

print("\nMODEL")
print(f"Models trained                : {len(models)}/{N_MODELS}")
print(f"Batch size                    : {BATCH_SIZE}")
print(f"Learning rate                 : {LR}")
print(f"Max epochs/model              : {MAX_EPOCHS_PER_MODEL}")
print(f"Early-stopping patience       : {PATIENCE}")

print("\nPRIMARY TEST REGRESSION — CDR 0.0")
print(f"MAE                           : {mae:.3f} years")
print(f"RMSE                          : {rmse:.3f} years")
print(f"Pearson r                     : {pearson_r:.3f}")
print(f"R²                            : {r2:.3f}")

print("\nRAW ENSEMBLE TEST ERROR")
print(f"MAE before bias correction    : {raw_mae:.3f} years")
print(f"Mean ensemble prediction SD   : {ensemble_sd_mean:.3f} years")

print("\nBRAIN AGE / BAG — TEST")
print(f"Mean BAG                      : {bag_mean:.3f} years")
print(f"BAG SD                        : {bag_sd:.3f} years")

print("\nDERIVED AGE-GROUP CLASSIFICATION")
print(f"Accuracy                      : {accuracy:.3f}")
print(f"Macro F1                      : {float(np.mean(f1)):.3f}")

print("\nMEASUREMENT NOISE")
if np.isfinite(sigma_sem):
    print(f"Repeat pairs                  : {len(repeat_pairs)}")
    print(f"Sigma_SEM                     : {sigma_sem:.3f} years")
    print(f"MDC (95%)                     : {mdc95:.3f} years")
else:
    print("Noise floor                   : Not estimable (<2 repeat pairs)")

print("\nResults directory:")
print(RESULTS_DIR)

# Save a compact final summary.
summary = {
    "manifest_scans": int(len(manifest)),
    "subjects": int(manifest["subject_id"].nunique()),
    "train_scans": int(len(train_df)),
    "val_scans": int(len(val_df)),
    "test_scans": int(len(test_df)),
    "test_cn_scans": int(len(test_cn_pred)),
    "models_trained": int(len(models)),
    "mae_years": mae,
    "rmse_years": rmse,
    "pearson_r": pearson_r,
    "r2": r2,
    "raw_mae_years": raw_mae,
    "mean_bag_years": bag_mean,
    "bag_sd_years": bag_sd,
    "ensemble_prediction_sd_years": ensemble_sd_mean,
    "age_group_accuracy": accuracy,
    "age_group_macro_f1": float(np.mean(f1)),
    "repeat_pairs": int(len(repeat_pairs)),
    "sigma_sem_years": None if not np.isfinite(sigma_sem) else float(sigma_sem),
    "mdc95_years": None if not np.isfinite(mdc95) else float(mdc95),
}

import json
with open(os.path.join(RESULTS_DIR, "final_statistics.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved: final_statistics.json")

