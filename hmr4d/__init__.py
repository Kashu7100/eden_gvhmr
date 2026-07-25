import os
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_ROOT_ENV = "GVHMR_CHECKPOINT_ROOT"


def get_checkpoint_root() -> Path:
    """Root directory holding the external, user-downloaded checkpoints.

    These are NOT shipped with the package (SMPL/SMPLX body models, GVHMR/HMR2/
    ViTPose/YOLO/DPVO weights). Resolution order:

    1. ``$GVHMR_CHECKPOINT_ROOT`` if set (recommended when the package is installed
       and run from an arbitrary working directory, e.g. an out-of-process worker);
    2. ``./inputs/checkpoints`` if it exists (the original repo convention: run from
       a directory laid out like the repo);
    3. ``PROJ_ROOT/inputs/checkpoints`` (an editable install run from any CWD).

    The fallback chain preserves the original behavior in the common cases while
    making the checkpoint location overridable without requiring ``CWD == repo root``.
    """
    env = os.environ.get(CHECKPOINT_ROOT_ENV)
    if env:
        return Path(env).expanduser()
    cwd_candidate = Path("inputs/checkpoints")
    if cwd_candidate.exists():
        return cwd_candidate
    return PROJ_ROOT / "inputs/checkpoints"


def os_chdir_to_proj_root():
    """useful for running notebooks in different directories."""
    os.chdir(PROJ_ROOT)
