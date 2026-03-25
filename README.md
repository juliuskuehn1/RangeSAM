# RangeSAM

RangeSAM is a research codebase for studying how visual foundation models, in particular SAM 2 style backbones, transfer to range-view LiDAR semantic segmentation. The repository contains model variants, preprocessing utilities, training and evaluation scripts, checkpoint download helpers, and dataset conversion tools for range-view experiments.

This work was presented at the WACV 2026 Foundational Models Beyond the Visual Spectrum (FoMoV) Workshop.

Paper:
https://openaccess.thecvf.com/content/WACV2026W/FoMoV/html/Kuhn_RangeSAM_On_the_Potential_of_Visual_Foundation_Models_for_Range-View_WACVW_2026_paper.html

## What Is In This Repository

- Range-view segmentation models and experiments, including RangeSAM variants and SAM-based architectures.
- Training scripts for SemanticKITTI-focused experiments, plus additional scripts for nuScenes and Cityscapes variants.
- Preprocessing tools for range projection and offline cache generation.
- A vendored SAM 2 codebase under `sam2/` and reference material under `sam2-files/`.
- Utilities for checkpoint download, visualization, verification, and evaluation.

## Recommended Environment

The repository is primarily set up for Linux with Python 3.10 and a CUDA-capable GPU.

Recommended baseline:

- Linux
- Python 3.10
- NVIDIA GPU with recent CUDA drivers
- `git`, `build-essential`, and `cmake`
- `nvcc` only if you want to build the optional SAM 2 CUDA extension locally

Notes:

- The project can still be installed when the SAM 2 CUDA extension cannot be compiled.
- If you do not have a local CUDA toolkit, disable the extension build explicitly during installation.
- Several scripts are research variants and assume specific dataset locations or experiment settings. Installation is stable; runtime configuration still requires choosing the script that matches your experiment.

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd pointcloud-segmentation
```

### 2. Create and activate a Python environment

Using `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Using Conda:

```bash
conda create -n rangesam python=3.10 -y
conda activate rangesam
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch

Install a PyTorch build that matches your system. The repository currently targets CUDA 12.1 wheels in `requirements.txt`.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Example for CPU-only:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Install the repository in editable mode

If you have a local CUDA toolkit and want the optional SAM 2 extension build:

```bash
SAM2_BUILD_ALLOW_ERRORS=1 pip install -e .
```

If you do not have `nvcc` or do not want to build the extension:

```bash
SAM2_BUILD_CUDA=0 pip install -e .
```

Why both `requirements.txt` and `pip install -e .` are used:

- `requirements.txt` installs the broader experiment dependencies used by this repository.
- `pip install -e .` registers the local code and the vendored `sam2` package so imports resolve correctly.

### 6. Verify the installation

```bash
python -c "import torch; import sam2; print('torch', torch.__version__)"
```

If you also want to verify that the project modules import:

```bash
python -c "from SAM2UNet import SAM2UNet; print('RangeSAM imports OK')"
```

## Checkpoints

**Important:** Checkpoints are not included in this repository due to GitHub's file size limits. You must download them before training or inference.

The repository provides a helper script to download SAM 2.1 checkpoints. Create a checkpoint directory and run the download script from inside it:

```bash
mkdir -p checkpoints
cd checkpoints
../download_ckpts.sh
cd ..
```

This will download and place files such as:

- `checkpoints/sam2.1_hiera_tiny.pt`
- `checkpoints/sam2.1_hiera_small.pt`
- `checkpoints/sam2.1_hiera_base_plus.pt`
- `checkpoints/sam2.1_hiera_large.pt`

This is required because several training and inference scripts assume checkpoint paths under `checkpoints/`. Without running the download script, training and evaluation will fail.

## Dataset Setup

### SemanticKITTI

Many of the training scripts assume a repository-local dataset directory named `dataset/` with a SemanticKITTI-like structure.

Typical expectation:

```text
dataset/
  sequences/
    00/
    01/
    ...
    21/
```

If your dataset lives elsewhere, either:

- create a symlink named `dataset` pointing to the real location, or
- edit the relevant training script arguments and hard-coded dataset roots

Some scripts directly construct datasets with `root="dataset"`, so keeping that path available is the simplest setup.

### Optional offline cache generation

If you want to precompute projected range-view tensors for faster training, use:

```bash
python precompute_semantickitti.py \
  --data-root /path/to/SemanticKITTI/sequences \
  --sequences 00 01 02 03 04 05 06 07 09 10 \
  --out-root /path/to/semk_cache \
  --n-workers 8 \
  --batch-size 4
```

To validate a generated cache:

```bash
python verify_semk_cache.py /path/to/semk_cache/seq_00
```

### nuScenes conversion

For nuScenes-based experiments, the repository includes a conversion helper that exports a SemanticKITTI-like layout:

```bash
python utils/nuscenes2kitti.py \
  --nuscenes_dir /path/to/nuscenes \
  --output_dir /path/to/nuScenes_converted
```

Additional packages may be required for nuScenes workflows, for example:

```bash
pip install nuscenes-devkit pyquaternion click pillow
```

### Cityscapes and other experiments

The repository also contains Cityscapes and other experiment-specific scripts. Those are not fully normalized behind a single CLI, so inspect the script you intend to run before launching training.

## Optional Dependencies

Some parts of the repository need extra packages beyond the base install.

### DeepSpeed training

If you plan to use `train_dpp_deepspeed.py` or the DeepSpeed config, install DeepSpeed separately:

```bash
pip install deepspeed
```

### Notebook and interactive tooling

If you want to use the included notebooks or additional demo tooling from the SAM 2 codebase:

```bash
pip install -e ".[notebooks,interactive-demo]"
```

## Docker Installation

The repository includes a Dockerfile based on NVIDIA CUDA 12.6.1.

Build the image:

```bash
docker build -t rangesam:latest .
```

Run it with GPU access and mount your dataset:

```bash
docker run --rm -it \
  --gpus all \
  -v /path/to/your/dataset:/workspace/dataset \
  -v $(pwd):/workspace \
  rangesam:latest
```

The Docker image installs the repository and common dependencies, and is the most reproducible option if your local CUDA toolchain is inconsistent.

## First Things To Check If Installation Fails

### `pip install -e .` fails while building the SAM 2 extension

Use:

```bash
SAM2_BUILD_CUDA=0 pip install -e .
```

The extension is optional for most workflows in this repository.

### PyTorch wheel mismatch

Make sure your PyTorch install matches your driver and desired CUDA runtime. If in doubt, reinstall PyTorch first and then rerun:

```bash
pip install -r requirements.txt
SAM2_BUILD_CUDA=0 pip install -e .
```

### Missing dataset path errors

Several scripts assume fixed paths such as `dataset/` and `checkpoints/`. Before debugging model code, verify that those directories exist and contain the expected files.

## Project Layout

Important paths:

- `sam2/`: vendored SAM 2 implementation and configs
- `preprocess/`: dataset parsing and preprocessing utilities
- `config/`: dataset, architecture, and label mappings
- `train.py`, `train_dpp.py`, `train_dpp_*`: training entrypoints and experiment variants
- `test.py`, `eval.py`: evaluation and inference utilities
- `download_ckpts.sh`: checkpoint downloader
- `precompute_semantickitti.py`: offline cache generation
- `verify_semk_cache.py`: cache verification utility

## Citation

If you use this repository, please cite the corresponding workshop paper:

```bibtex
@inproceedings{kuhn2026rangesam,
  title={RangeSAM: On the Potential of Visual Foundation Models for Range-View Segmentation},
  booktitle={Proceedings of the WACV 2026 Foundational Models Beyond the Visual Spectrum (FoMoV) Workshop},
  year={2026},
  url={https://openaccess.thecvf.com/content/WACV2026W/FoMoV/html/Kuhn_RangeSAM_On_the_Potential_of_Visual_Foundation_Models_for_Range-View_WACVW_2026_paper.html}
}
```

If you need the exact published BibTeX from CVF, use the citation block on the paper page linked above.

## Acknowledgements

This repository builds on top of SAM 2 and related LiDAR segmentation tooling.

- SAM 2: https://github.com/facebookresearch/sam2
- WACV 2026 FoMoV Workshop paper page: https://openaccess.thecvf.com/content/WACV2026W/FoMoV/html/Kuhn_RangeSAM_On_the_Potential_of_Visual_Foundation_Models_for_Range-View_WACVW_2026_paper.html
```shell
git clone https://github.com/WZH0120/SAM2-UNet.git
cd SAM2-UNet/
```

## Prepare Datasets
You can refer to the following repositories and their papers for the detailed configurations of the corresponding datasets.
- Camouflaged Object Detection. Please refer to [FEDER](https://github.com/ChunmingHe/FEDER). [#issue [#13](https://github.com/WZH0120/SAM2-UNet/issues/13), [#44](https://github.com/WZH0120/SAM2-UNet/issues/44)]
- Salient Object Detection. Please refer to [SALOD](https://github.com/moothes/SALOD).
- Marine Animal Segmentation. Please refer to [MASNet](https://github.com/zhenqifu/MASNet).
- Mirror Detection. Please refer to [HetNet](https://github.com/Catherine-R-He/HetNet).
- Polyp Segmentation. Please refer to [PraNet](https://github.com/DengPingFan/PraNet).

## Requirements
Our project does not depend on installing SAM2. If you have already configured an environment for SAM2, then directly using this environment should also be fine. You may also create a new conda environment:

```shell
conda create -n sam2-unet python=3.10
conda activate sam2-unet
pip install -r requirements.txt
```

## Training
If you want to train your own model, please download the pre-trained segment anything 2 (not SAM2.1, [#issue [#18](https://github.com/WZH0120/SAM2-UNet/issues/18), [#30](https://github.com/WZH0120/SAM2-UNet/issues/30)]) from the [official repository](https://github.com/facebookresearch/segment-anything-2). You can also directly download `sam2_hiera_large.pt` from [here](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt). After the above preparations, you can run `train.sh` to start your training.

## Testing
Our pre-trained models and prediction maps can be found on [Google Drive](https://drive.google.com/drive/folders/1w2fK8kLhtEmMWZ6G6w9_J17xwgfm3lev?usp=drive_link). Also, you can run `test.sh` to obtain your own predictions.

## Evaluation
After obtaining the prediction maps, you can run `eval.sh` to get most of the quantitative results. For the evaluation of mirror detection, please refer to `eval.py` in [HetNet](https://github.com/Catherine-R-He/HetNet) to obtain the results.

## Other Interesting Works
If you are interested in designing SAM2-based methods, the following papers may be helpful:

[2025.03] [Adapting Vision Foundation Models for Real-time Ultrasound Image Segmentation](https://arxiv.org/abs/2503.24368)

[2025.03] [DSU-Net:An Improved U-Net Model Based on DINOv2 and SAM2 with Multi-scale Cross-model Feature Enhancement](https://arxiv.org/abs/2503.21187)

[2025.03] [Research on recognition of diabetic retinopathy hemorrhage lesions based on fine tuning of segment anything model](https://www.nature.com/articles/s41598-025-92665-7)

[2025.03] [SAM2-ELNet: Label Enhancement and Automatic Annotation for Remote Sensing Segmentation](https://arxiv.org/abs/2503.12404)

[2025.02] [Fine-Tuning SAM2 for Generalizable Polyp Segmentation with a Channel Attention-Enhanced Decoder](https://ojs.sgsci.org/journals/amr/article/view/311)

[2025.02] [FE-UNet: Frequency Domain Enhanced U-Net with Segment Anything Capability for Versatile Image Segmentation](https://arxiv.org/abs/2502.03829)

[2025.01] [Progressive Self-Prompting Segment Anything Model for Salient Object Detection in Optical Remote Sensing Images](https://doi.org/10.3390/rs17020342)

[2024.12] [Adapting SAM2 Model from Natural Images for Tooth Segmentation in Dental Panoramic X-Ray Images](https://doi.org/10.3390/e26121059)

[2024.11] [SAM-I2I: Unleash the Power of Segment Anything Model for Medical Image Translation](https://arxiv.org/abs/2411.12755)

## Citation and Star
Please cite the following paper and star this project if you use this repository in your research. Thank you!
```
@InProceedings{Kuhn_2026_WACV,
    author    = {K\"uhn, Paul Julius and Nguyen, Duc Anh and Kuijper, Arjan and Sinha, Saptarshi Neil},
    title     = {RangeSAM: On the Potential of Visual Foundation Models for Range-View represented LiDAR segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {March},
    year      = {2026},
    pages     = {1540-1548}
}
```

## Acknowledgement
[segment anything 2](https://github.com/facebookresearch/segment-anything-2)
