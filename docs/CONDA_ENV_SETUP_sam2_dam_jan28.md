# Conda Environment Setup: `sam2_dam_jan28`

## Environment Summary

- **Environment Name:** `sam2_dam_jan28`
- **Python Version:** 3.10.13
- **Created:** January 28, 2026
- **Location:** `<CONDA_ENVS>/sam2_dam_jan28`

## System Configuration

- **GPUs:** 4x NVIDIA RTX A6000 (49 GB each)
- **GPU Driver:** 565.57.01
- **CUDA Toolkit (driver):** 12.7
- **PyTorch CUDA Build:** cu124 (ships CUDA 12.4 runtime + libraries)
- **Compute Capability:** 8.6

## Creation Steps

```bash
cd <REPO_ROOT>
source "$(conda info --base)"/etc/profile.d/conda.sh
conda create -y -n sam2_dam_jan28 python=3.10.13
conda activate sam2_dam_jan28
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -e ".[dev]"
```

## Installed Highlights

- **PyTorch Stack:** `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`, `torchaudio==2.5.1+cu124`
- **Core libs:** `numpy==2.2.6`, `pillow==12.0.0`, `tqdm==4.67.1`
- **SAM2 extras:** `hydra-core==1.3.2`, `iopath==0.1.10`, `fvcore==0.1.5.post20221221`, `pandas==2.3.3`, `scikit-image==0.25.2`, `tensorboard==2.20.0`, `pycocotools==2.0.11`, `tensordict==0.11.0`, `opencv-python==4.13.0.90`, `submitit==1.5.4`
- **Dev tooling:** `black==24.2.0`, `usort==1.0.2`, `ufmt==2.0.0b2`

## Activation

```bash
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate sam2_dam_jan28
```

## Verification

```bash
python -c "import torch; print('PyTorch:', torch.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('Device count:', torch.cuda.device_count()); \
print('Device 0:', torch.cuda.get_device_name(0))"
```

Expected output:
```
PyTorch: 2.5.1+cu124
CUDA available: True
Device count: 4
Device 0: NVIDIA RTX A6000
```

## Project-Specific Configuration

This environment is set up for the SAM2 training project with DAM (Distractor-Aware Memory) and Adaptive Gating features:
- **Reference documents:** 
  - `update_quality_weighted_RAM_selection.md`
  - `train_guide.md`
  - `README.md`
- **Key features enabled:** 
  - Two-stage memory refinement
  - DAM (RAM/DRM) memory bank structure
  - Quality-Weighted RAM Selection
  - Adaptive Gating (`gate_query`, `gate_ram`, `gate_drm`)
