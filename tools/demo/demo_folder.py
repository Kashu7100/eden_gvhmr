"""Run the GVHMR demo over every ``.mp4`` in a folder.

Uses the importable :class:`hmr4d.demo.GVHMR` so the model is loaded once and reused
across all videos (the previous version spawned a fresh ``demo.py`` subprocess per
video, reloading the model each time).
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from hmr4d.demo import GVHMR
from hmr4d.utils.pylogger import Log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument("-d", "--output_root", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None, help="GVHMR checkpoint (default from config)")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    args = parser.parse_args()

    folder = Path(args.folder)
    mp4_paths = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.MP4")))
    Log.info(f"Found {len(mp4_paths)} .mp4 files in {folder}")

    model = GVHMR(ckpt_path=args.ckpt_path)
    for mp4_path in tqdm(mp4_paths):
        Log.info(f"Running: {mp4_path}")
        model.recover(
            mp4_path,
            static_cam=args.static_cam,
            output_root=args.output_root,
            render=True,
        )


if __name__ == "__main__":
    main()
