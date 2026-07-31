# Install

## Environment

```bash
git clone https://github.com/Kashu7100/eden_gvhmr
cd eden_gvhmr

conda create -y -n gvhmr python=3.10
conda activate gvhmr

# 1. torch / torchvision must match your CUDA toolkit, so they are not auto-installed (pinning
#    them would break co-installation into an existing GPU env). The versions below reproduce
#    the reference setup (CUDA 12.1); adjust for your CUDA.
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 2. Install GVHMR (the importable package is `hmr4d`). Optional extras: dpvo, render, full.
pip install -e .
# or, to also pull the optional DPVO backend:  pip install -e ".[full]"
```

**Inference needs no compiled dependency.** pytorch3d used to be required, but the only part
GVHMR needed on the inference path was its pure-torch rotation helpers, and those are now
vendored in `hmr4d/utils/rotation_conversions.py` (copied verbatim from pytorch3d 0.7.8, BSD-3).
`pip install -e .` is therefore enough to run `python -m hmr4d.demo`.

### Optional: mesh rendering (`--render`)

Only `render=True` still needs the real pytorch3d, for its rasterizer. It must be built against
your exact torch/CUDA; prebuilt wheels exist for a few combinations, otherwise it builds from
source (a couple of minutes, needs `nvcc`):

```bash
# prebuilt, if one matches your python/CUDA/torch:
pip install "pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl"
# otherwise, from source:
CUDA_HOME=/usr/local/cuda pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

The demo has been verified end-to-end with this recipe on an RTX 3080 (driver 580, CUDA 12.6
runtime — the cu121 wheels are forward-compatible).

### Versions: floors, not pins

`pyproject.toml` declares floors so GVHMR installs next to a modern stack. Two constraints are
real, and neither is expressible in metadata:

- **torch decides your numpy.** torch declares no numpy requirement at any version, but torch
  < 2.4 is compiled against numpy 1.x and dies at import under numpy 2. Install torch first and
  the rest follows.
- **ultralytics ≥ 8.4 is required by modern torch.** torch 2.6 flipped `torch.load` to
  `weights_only=True`; ultralytics 8.2.42 (upstream's pin) predates that and its own loader
  raises `UnpicklingError` on `yolov8x.pt`. That is why the floor exists.

A third constraint is a version *pairing*: av 14 changed the API imageio's pyav plugin uses, so
imageio 2.34.1 against av >= 14 fails to read the video at all. Hence the `imageio>=2.37` floor.

Two of these versions are *part of the numerics*. Measured one change at a time on the bundled
tennis clip, against the reference pins:

| change | effect on the estimate |
| --- | --- |
| numpy 1.23.5 -> 1.26.4 | **bit-identical** |
| imageio 2.34.1 -> 2.37.4 | **bit-identical** |
| timm 0.9.12 -> 1.0.28 | **bit-identical** |
| av 13.0 -> 17.1 (decoder) | 3.6 deg max on body joints |
| ultralytics 8.2.42 -> 8.4.113 (detector) | 8.5 px on the boxes, 3.4 deg max on body joints |

So the drift is the video decoder and the person detector — not numpy, and not timm. Neither is
wrong; they just are not the versions the released checkpoint was tuned against. Within one
fixed environment the pipeline is deterministic: two clean installs of the same versions
reproduce each other bit-for-bit.

### Reproducing the reference numbers

The released checkpoint was validated on the stack below. There is no `reproducible` extra,
because it would have to relax the floors above and an extra can only narrow — so it is a manual
recipe. It needs **python ≤ 3.11** (numpy 1.23.5 ships no cp312 wheel) and **torch 2.3**:

```bash
uv venv --python 3.10 && uv pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
uv pip install -e . --no-deps          # skip the modern floors
uv pip install numpy==1.23.5 imageio==2.34.1 av==13.0.0 timm==0.9.12 ultralytics==8.2.42 \
  pytorch-lightning hydra-core hydra-zen hydra-colorlog omegaconf einops "opencv-python<5" \
  scikit-image termcolor rich joblib ffmpeg-python huggingface-hub yacs tqdm smplx trimesh \
  cython_bbox lapx wis3d pycolmap \
  "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa8"
```

The distribution name is `eden-gvhmr`; the import name stays `hmr4d`.
`requirements.txt` remains as an exact reproduction of the original upstream environment
(`pip install -r requirements.txt`) — the `pip install .` path above is preferred.

> **Co-installing into another env (e.g. Eden)?** This now resolves cleanly. The two things
> that used to prevent it are gone: `pytorch3d` is no longer needed for inference, and the
> `numpy<1.24` cap was an artifact of chumpy 0.70 (chumpy master, pinned here, imports fine
> under numpy 2). Installing GVHMR into a current Eden environment is purely additive — no
> downgrades. Note that you then run on that environment's `av`/`imageio`/`timm`/`ultralytics`,
> so results shift slightly from the reference stack above.

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

# 2. Download the pretrained models (by downloading, you agree to the corresponding licences):

gvhmr-download-checkpoints            # add --dpvo for the optional DPVO backend

inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth                      # optional (DPVO backend only)
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
```

`gvhmr-download-checkpoints` writes to the checkpoint root (`--checkpoint_root` /
`$GVHMR_CHECKPOINT_ROOT`, default `./inputs/checkpoints`) and skips files that already exist.

> **Why not the Google-Drive folder?** The upstream
> [Drive folder](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link)
> is chronically rate-limited ("Too many users have viewed or downloaded this file recently"),
> which makes `gdown` fail on every file, folder download included. The command above pulls the
> identical weights from a Hugging Face mirror (`camenduru/GVHMR`, override with `--repo`).
> It does **not** cover the licence-gated SMPL/SMPLX body models in step 1.

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
