"""Run the GVHMR demo over every ``.mp4`` in a folder.

Uses the importable :class:`hmr4d.demo.GVHMR` so the model is loaded once and reused
across all videos (the original spawned a fresh ``demo.py`` subprocess per video,
reloading the model each time). Because everything runs in one process, each video is
guarded so a single failure is logged and skipped instead of aborting the whole batch,
and the CUDA cache is cleared between videos. A hard CUDA error can still corrupt the
process for the remaining videos — re-run those separately if the summary reports them.
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from hmr4d.demo import GVHMR
from hmr4d.utils.pylogger import Log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument("-d", "--output_root", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None, help="GVHMR checkpoint (default from checkpoint root)")
    parser.add_argument("--checkpoint_root", type=str, default=None, help="Directory holding external checkpoints")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    args = parser.parse_args()

    folder = Path(args.folder)
    mp4_paths = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.MP4")))
    Log.info(f"Found {len(mp4_paths)} .mp4 files in {folder}")

    model = GVHMR(ckpt_path=args.ckpt_path, checkpoint_root=args.checkpoint_root)
    failed = []
    for mp4_path in tqdm(mp4_paths):
        Log.info(f"Running: {mp4_path}")
        try:
            model.recover(
                mp4_path,
                static_cam=args.static_cam,
                output_root=args.output_root,
                render=True,
            )
        except Exception as exc:  # keep going so one bad video doesn't abort the batch
            Log.error(f"[Failed] {mp4_path}: {type(exc).__name__}: {exc}")
            failed.append(mp4_path)
        finally:
            torch.cuda.empty_cache()

    if failed:
        Log.error(f"{len(failed)}/{len(mp4_paths)} videos failed: {[str(p) for p in failed]}")


if __name__ == "__main__":
    main()
