"""GVHMR single-video demo CLI.

Runnable as ``python -m hmr4d.demo``, via the ``gvhmr-demo`` console script, or
through the ``tools/demo/demo.py`` shim. Thin wrapper over :class:`hmr4d.demo.GVHMR`.
"""

import argparse

import torch

from hmr4d.demo import GVHMR
from hmr4d.utils.pylogger import Log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("--ckpt_path", type=str, default=None, help="GVHMR checkpoint (default from config)")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="Focal length of fullframe camera in mm. Leave it as None to use default values."
        "For iPhone 15p, the [0.5x, 1x, 2x, 3x] lens have typical values [13, 24, 48, 77]."
        "If the camera zoom in a lot, you can try 135, 200 or even larger values.",
    )
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    return parser.parse_args()


def main():
    args = parse_args()
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    model = GVHMR(ckpt_path=args.ckpt_path)
    model.recover(
        args.video,
        static_cam=args.static_cam,
        use_dpvo=args.use_dpvo,
        f_mm=args.f_mm,
        output_root=args.output_root,
        render=True,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
