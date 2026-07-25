# Install

## Environment

```bash
git clone https://github.com/Kashu7100/eden_gvhmr
cd eden_gvhmr

conda create -y -n gvhmr python=3.10
conda activate gvhmr

# 1. Prerequisites that must match your CUDA toolkit (NOT auto-installed by the package,
#    because pinning them would break co-installation into an existing GPU env):
#      - torch / torchvision
#      - pytorch3d (a hard inference dependency)
#    The versions below reproduce the reference setup (CUDA 12.1); adjust for your CUDA.
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install "pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl"

# 2. Install GVHMR (the importable package is `hmr4d`). Optional extras: dpvo, render, slam, full.
pip install -e .
# or, to also pull the optional backends:  pip install -e ".[full]"
```

The distribution name is `eden-gvhmr`; the import name stays `hmr4d`.
`requirements.txt` remains as an exact reproduction of the original upstream environment
(`pip install -r requirements.txt`) — the `pip install .` path above is preferred.

> **Co-installing into another env (e.g. Eden)?** `chumpy` (used to load SMPL `.pkl`)
> requires `numpy < 1.24`, which conflicts with newer environments. GVHMR is therefore best
> installed in its **own** environment; an integration such as an Eden extension should call
> it out-of-process (subprocess/worker) rather than sharing a single interpreter.

### Optional: DPVO (not recommended if you want fast inference speed)
```bash
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
pip install numba pypose
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/
pip install -e .
```

## Inputs & Outputs

```bash
mkdir inputs
mkdir outputs
```

**Weights**

```bash
mkdir -p inputs/checkpoints

# 1. You need to sign up for downloading [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/). And the checkpoints should be placed in the following structure:

inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPLX (We predict SMPLX params + evaluation)
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)

# 2. Download other pretrained models from Google-Drive (By downloading, you agree to the corresponding licences): https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
```

> **Checkpoint location.** By default GVHMR looks for these checkpoints under
> `./inputs/checkpoints` (i.e. run from a directory laid out like this repo). If you install
> the package and run from elsewhere, point it at your checkpoint directory with the
> `GVHMR_CHECKPOINT_ROOT` environment variable, or via `GVHMR(checkpoint_root=...)` /
> `--checkpoint_root` in code. Body-model regressors and Hydra configs are shipped inside the
> package; only the weights above must be downloaded.

**Data**

We provide preprocessed data for training and evaluation.
Note that we do not intend to distribute the original datasets, and you need to download them (annotation, videos, etc.) from the original websites.
*We're unable to provide the original data due to the license restrictions.*
By downloading the preprocessed data, you agree to the original dataset's terms of use and use the data for research purposes only.

You can download them from [Google-Drive](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link). Please place them in the "inputs" folder and execute the following commands:

```bash
cd inputs
# Train
tar -xzvf AMASS_hmr4d_support.tar.gz
tar -xzvf BEDLAM_hmr4d_support.tar.gz
tar -xzvf H36M_hmr4d_support.tar.gz
# Test
tar -xzvf 3DPW_hmr4d_support.tar.gz
tar -xzvf EMDB_hmr4d_support.tar.gz
tar -xzvf RICH_hmr4d_support.tar.gz

# The folder structure should be like this:
inputs/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
└── RICH/hmr4d_support/
```
