"""Fetch the pretrained weights GVHMR needs into the checkpoint root.

The upstream instructions point at a Google-Drive folder that is chronically rate-limited
("Too many users have viewed or downloaded this file recently"), which makes ``gdown`` fail
on every file. This pulls the same weights from a Hugging Face mirror instead.

Run as ``gvhmr-download-checkpoints`` or ``python -m hmr4d.download``.

Only the *pretrained* weights are covered. The SMPL/SMPLX body models are licence-gated and
must still be downloaded by hand after registering at https://smpl.is.tue.mpg.de/ and
https://smpl-x.is.tue.mpg.de/ -- see docs/INSTALL.md for the expected layout.

By downloading these weights you agree to the licences of the original releases.
"""

import argparse
import shutil
from pathlib import Path

from hmr4d import get_checkpoint_root

HF_REPO = "camenduru/GVHMR"

REQUIRED = (
    "gvhmr/gvhmr_siga24_release.ckpt",
    "hmr2/epoch=10-step=25000.ckpt",
    "vitpose/vitpose-h-multi-coco.pth",
    "yolo/yolov8x.pt",
)
# Only needed for the optional DPVO camera backend; the default SimpleVO does not use it.
DPVO = "dpvo/dpvo.pth"

BODY_MODELS = (
    "body_models/smplx/SMPLX_NEUTRAL.npz",
    "body_models/smpl/SMPL_NEUTRAL.pkl",
)


def download(checkpoint_root=None, include_dpvo=False, repo=HF_REPO):
    """Download the pretrained weights into ``checkpoint_root``, skipping existing files."""
    from huggingface_hub import hf_hub_download

    root = Path(checkpoint_root) if checkpoint_root is not None else get_checkpoint_root()
    root.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED + ((DPVO,) if include_dpvo else ()):
        dst = root / name
        if dst.exists():
            print(f"[skip]     {name} (already present)")
            continue
        print(f"[download] {name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Copy out of the HF cache so the tree stays self-contained and cache-eviction safe.
        shutil.copy(hf_hub_download(repo, name), dst)

    missing = [n for n in BODY_MODELS if not (root / n).exists()]
    if missing:
        print("\nStill missing (licence-gated, download by hand -- see docs/INSTALL.md):")
        for name in missing:
            print(f"  {root / name}")

    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=None,
        help="Destination directory. Defaults to ./inputs/checkpoints; also settable via $GVHMR_CHECKPOINT_ROOT.",
    )
    parser.add_argument("--dpvo", action="store_true", help="Also fetch dpvo.pth (optional DPVO backend).")
    parser.add_argument("--repo", type=str, default=HF_REPO, help=f"Hugging Face repo to mirror from (default {HF_REPO}).")
    args = parser.parse_args()

    root = download(checkpoint_root=args.checkpoint_root, include_dpvo=args.dpvo, repo=args.repo)
    print(f"\nCheckpoint root: {root.resolve()}")


if __name__ == "__main__":
    main()
