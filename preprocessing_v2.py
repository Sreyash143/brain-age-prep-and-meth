# -*- coding: utf-8 -*-
"""
Preprocessing pipeline v2 — Revision 2 patch.

WHAT CHANGED vs. your original `run_preprocessing` (preprocessing.ipynb, Cell 3):
  Steps 1-5 (reorientation/crop, N4 bias correction, skull-stripping, 12-DOF
  affine to MNI152, resample+crop to 160x192x160) are copied over UNCHANGED.
  They already produce exactly the input the pretrained checkpoint expects
  (a linearly-registered, skull-stripped 160x192x160 MNI152 1mm volume).

  Step 6 is REPLACED. Your original code did brain-masked Z-score
  normalization: (x - mu) / sigma. The UKBiobank_deep_pretrain checkpoint
  (ha-ha-ha-han/UKBiobank_deep_pretrain, examples.ipynb) was trained on
  data normalized as:

      data = data / data.mean()

  i.e. plain mean-scaling over the whole volume, NOT z-scoring. This is
  confirmed directly from their example inference code. If you fine-tune
  the pretrained weights on z-scored input, the backbone sees an input
  distribution it never saw during pretraining and you lose most of the
  transfer-learning benefit (frozen BatchNorm3d running stats in particular
  will be badly mismatched).

WHY YOU MUST RE-RUN PREPROCESSING (not just patch the saved .npz files):
  Your original pipeline only ever saved the *already z-scored* array
  (mu and sigma were computed and then discarded, never written to disk).
  z = (x - mu) / sigma is not invertible back to `x / mu_x` without the
  original mu and sigma, so the existing 3340 _preproc.npz files cannot be
  losslessly converted in place. Re-run `batch_preprocess_t1_v2` on the raw
  T1 files into a NEW output directory. This repeats the (expensive) ANTs
  registration steps, but that's the safe option on your hardware.

  This version also persists `brain_mean` / `brain_std` per scan (in the
  .npz, not just as console output) so any future renormalization choice
  can be made without re-running ANTs again.
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
# MNI152 brain template cache (UNCHANGED)
# =========================================================

_MNI_BRAIN_CACHE = "mni152_brain_1mm.nii.gz"


def get_mni_brain_template():
    """Returns a skull-stripped 1mm MNI152 template (cached to disk)."""
    if os.path.exists(_MNI_BRAIN_CACHE):
        return ants.image_read(_MNI_BRAIN_CACHE)

    print("Creating skull-stripped MNI152 template...")
    mni_head = ants.image_read(antspynet.get_antsxnet_data("mni152"))
    mni_mask_prob = antspynet.brain_extraction(mni_head, modality="t1")
    mni_mask_bin = ants.threshold_image(mni_mask_prob, 0.5, 1.0, 1, 0)
    mni_brain = ants.mask_image(mni_head, mni_mask_bin)
    ants.image_write(mni_brain, _MNI_BRAIN_CACHE)
    print(f"Saved MNI brain template to: {_MNI_BRAIN_CACHE}")
    return mni_brain


# =========================================================
# Single MRI preprocessing (Steps 1-5 unchanged, Step 6 replaced)
# =========================================================

def run_preprocessing_v2(input_nii_path: str, output_tensor_path: str):
    """
    6-stage preprocessing pipeline:
      1. Reorientation & FOV Cropping                  [UNCHANGED]
      2. N4 Bias Field Correction                       [UNCHANGED]
      3. Skull-Stripping                                [UNCHANGED]
      4. Affine Registration to MNI152                  [UNCHANGED]
      5. Voxel Resampling & Tensor Formatting            [UNCHANGED]
      6. Mean-Scaling Normalization (data / data.mean()) [CHANGED — was Z-score]
    """
    print(f"\nProcessing: {os.path.basename(input_nii_path)}")

    # --- Step 1: Reorientation & FOV Cropping ---
    img = ants.image_read(input_nii_path)
    print(f"  Input intensity range: [{img.min():.2f}, {img.max():.2f}]")
    img = ants.reorient_image2(img, orientation="RAS")
    img_cropped = ants.crop_image(img)

    # --- Step 2: N4 Bias Field Correction ---
    img_n4 = ants.n4_bias_field_correction(img_cropped, shrink_factor=2)

    # --- Step 3: Brain Extraction / Skull Stripping ---
    mask_prob = antspynet.brain_extraction(img_n4, modality="t1")
    mask_bin = ants.threshold_image(mask_prob, 0.5, 1.0, 1, 0)
    brain = ants.mask_image(img_n4, mask_bin)

    voxel_count = int(mask_bin.sum())
    print(f"  Brain mask voxel count: {voxel_count}")
    if voxel_count < 100_000:
        raise RuntimeError(
            f"Brain extraction found almost no brain tissue ({voxel_count} voxels) "
            f"in {os.path.basename(input_nii_path)}. This may indicate that the "
            f"input is not a raw T1w MRI scan."
        )

    # --- Step 4: Spatial Registration to 1mm MNI152 Template ---
    mni_brain = get_mni_brain_template()
    print("  Registering brain to MNI152...")
    reg = ants.registration(fixed=mni_brain, moving=brain, type_of_transform="Affine")
    warped_brain = reg["warpedmovout"]

    # --- Step 5: Crop/Pad to exact SFCN matrix (160x192x160) ---
    temp_nii = "_temp_reg.nii.gz"
    ants.image_write(warped_brain, temp_nii)

    subject = tio.Subject(mri=tio.ScalarImage(temp_nii))
    crop_pad = tio.CropOrPad((160, 192, 160))
    processed_subject = crop_pad(subject)

    data = processed_subject.mri.data.squeeze().numpy()
    final_affine = processed_subject.mri.affine

    if os.path.exists(temp_nii):
        os.remove(temp_nii)

    # --- Step 6: Mean-Scaling Normalization [CHANGED] ---
    # Matches the UKBiobank_deep_pretrain checkpoint's own preprocessing
    # (examples.ipynb: `data = data / data.mean()`), computed here over the
    # brain-masked region only (background is exactly zero after skull-strip
    # + crop, so including it would bias the mean toward zero-padding).
    brain_mask = data > 0
    normalized = np.zeros_like(data, dtype=np.float32)

    brain_mean = float("nan")
    brain_std = float("nan")
    if np.any(brain_mask):
        brain_mean = float(np.mean(data[brain_mask]))
        brain_std = float(np.std(data[brain_mask]))
        if brain_mean > 1e-8:
            normalized[brain_mask] = data[brain_mask] / brain_mean
        else:
            raise RuntimeError(
                f"Brain-masked mean is ~0 for {os.path.basename(input_nii_path)}; "
                f"cannot mean-normalize safely."
            )

    # --- Save NPZ (now also stores brain_mean/brain_std for provenance) ---
    output_directory = os.path.dirname(output_tensor_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    np.savez_compressed(
        output_tensor_path,
        data=normalized,
        brain_mean=brain_mean,
        brain_std=brain_std,
        normalization="mean_scaling",  # explicit tag so downstream code never has to guess
    )

    # --- Save NIfTI for Slicer QC ---
    slicer_nii_path = output_tensor_path.replace(".npz", ".nii.gz")
    nii_img = nib.Nifti1Image(normalized, final_affine)
    nib.save(nii_img, slicer_nii_path)

    print(f"  Brain-masked mean (pre-norm): {brain_mean:.4f}")
    print(f"  Saved training tensor: {output_tensor_path}")
    print(f"  Saved Slicer volume: {slicer_nii_path}")


# =========================================================
# FIND ALL MRI FILES RECURSIVELY (UNCHANGED)
# =========================================================

def find_t1_files(raw_dir: str):
    nii_files = []
    nii_files.extend(glob.glob(os.path.join(raw_dir, "**", "*.nii"), recursive=True))
    nii_files.extend(glob.glob(os.path.join(raw_dir, "**", "*.nii.gz"), recursive=True))
    nii_files = [
        f for f in nii_files
        if not f.endswith(".json") and "_preproc" not in os.path.basename(f)
    ]
    return sorted(nii_files)


# =========================================================
# BATCH PREPROCESSING — writes to a NEW output dir (v2 suffix)
# =========================================================

def batch_preprocess_t1_v2(raw_dir: str, output_base_dir: str):
    """
    Same skip-if-done behavior as your original batch_preprocess_t1, but
    pointed at a new output_base_dir so the (correctly) mean-normalized
    files never collide with the old z-scored ones.
    """
    nii_files = find_t1_files(raw_dir)
    print(f"Found {len(nii_files)} raw T1 files under {raw_dir}")

    n_done, n_skipped, n_failed = 0, 0, 0
    for i, nii_path in enumerate(nii_files):
        rel = os.path.relpath(nii_path, raw_dir)
        out_path = os.path.join(
            output_base_dir, os.path.dirname(rel),
            os.path.basename(nii_path).replace(".nii.gz", "_preproc.npz").replace(".nii", "_preproc.npz"),
        )

        if os.path.exists(out_path):
            n_skipped += 1
            continue

        try:
            t0 = time.time()
            run_preprocessing_v2(nii_path, out_path)
            print(f"  [{i+1}/{len(nii_files)}] done in {time.time()-t0:.1f}s")
            n_done += 1
        except Exception as e:
            print(f"  [{i+1}/{len(nii_files)}] FAILED: {e}")
            n_failed += 1

    print("=" * 60)
    print(f"Preprocessing v2 complete: {n_done} done, {n_skipped} skipped, {n_failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    RAW_DIR = "/mnt/d/brain age prep and meth/T1"
    OUTPUT_DIR = "/mnt/d/brain age prep and meth/T1 preprocessed v2"  # NEW dir — do not overwrite v1
    batch_preprocess_t1_v2(RAW_DIR, OUTPUT_DIR)
