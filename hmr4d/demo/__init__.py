"""Importable GVHMR demo pipeline.

Exposes a small :class:`GVHMR` class so GVHMR can be run programmatically (e.g. as
an Eden extension) instead of only through the ``tools/demo/demo.py`` CLI:

    >>> from hmr4d.demo import GVHMR
    >>> model = GVHMR(ckpt_path="inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt")
    >>> out = model.recover("video.mp4", static_cam=True)
    >>> out["smpl_params_global"], out["smpl_params_incam"], out["K_fullimg"]

The individual stages remain available in :mod:`hmr4d.demo.pipeline` for callers
that want finer control.
"""

import os
from pathlib import Path

import torch

from hmr4d import get_checkpoint_root, CHECKPOINT_ROOT_ENV
from hmr4d.utils.instantiate import instantiate
from hmr4d.utils.pylogger import Log
from hmr4d.utils.net_utils import detach_to_cpu
from hmr4d.utils.video_io_utils import merge_videos_horizontal
from hmr4d.demo.pipeline import (
    build_cfg,
    run_preprocess,
    load_data_dict,
    render_incam,
    render_global,
)

__all__ = ["GVHMR", "build_cfg", "run_preprocess", "load_data_dict", "render_incam", "render_global"]


class GVHMR:
    """World-grounded human motion recovery from a monocular video.

    The pretrained model is loaded once (lazily, on the first :meth:`recover` call)
    and reused across videos. Preprocess extractors (tracker / ViTPose / ViT feature
    extractor) are instantiated per call, matching the original demo script.

    GVHMR is CUDA-only; select a GPU with ``CUDA_VISIBLE_DEVICES``.

    Parameters
    ----------
    ckpt_path : str | Path | None
        GVHMR checkpoint. ``None`` derives it from the checkpoint root as
        ``<checkpoint_root>/gvhmr/gvhmr_siga24_release.ckpt``.
    checkpoint_root : str | Path | None
        Directory holding the external, user-downloaded checkpoints (body models,
        GVHMR/HMR2/ViTPose/YOLO/DPVO weights). When given, it is exported via
        ``$GVHMR_CHECKPOINT_ROOT`` so every loader resolves against it — this is what
        lets the package run from an arbitrary working directory (e.g. an
        out-of-process worker) rather than requiring ``CWD == repo root``. ``None``
        keeps the default resolution (see :func:`hmr4d.get_checkpoint_root`).
        Note: the env var is process-wide, so the most recent value wins.
    """

    def __init__(self, ckpt_path=None, checkpoint_root=None):
        if checkpoint_root is not None:
            os.environ[CHECKPOINT_ROOT_ENV] = str(Path(checkpoint_root).expanduser())
        self.ckpt_path = str(ckpt_path) if ckpt_path is not None else None
        self._model = None

    def _ensure_model(self, cfg):
        if self._model is None:
            Log.info("[HMR4D] Loading model")
            model = instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            self._model = model.eval().cuda()
        return self._model

    @torch.no_grad()
    def recover(
        self,
        video,
        *,
        static_cam=False,
        use_dpvo=False,
        f_mm=None,
        output_root=None,
        render=False,
        verbose=False,
    ):
        """Run the full pipeline on ``video`` and return the recovered SMPL-X params.

        Parameters
        ----------
        video : str | Path
            Input video.
        static_cam : bool
            Skip visual odometry (assume a static camera).
        use_dpvo : bool
            Use DPVO instead of the default SimpleVO for camera estimation.
        f_mm : int | None
            Full-frame focal length in mm; ``None`` estimates it.
        output_root : str | Path | None
            Where intermediate/preprocess artifacts are written
            (default ``outputs/demo``).
        render : bool
            Also render the in-camera / global overlay videos.
        verbose : bool
            Save intermediate preprocess overlays.

        Returns
        -------
        dict
            ``{"smpl_params_global", "smpl_params_incam", "K_fullimg", "output_dir"}``.
        """
        ckpt_path = self.ckpt_path or str(get_checkpoint_root() / "gvhmr/gvhmr_siga24_release.ckpt")
        cfg = build_cfg(
            video,
            static_cam=static_cam,
            verbose=verbose,
            use_dpvo=use_dpvo,
            f_mm=f_mm,
            output_root=output_root,
            ckpt_path=ckpt_path,
        )

        run_preprocess(cfg)
        data = load_data_dict(cfg)

        if not Path(cfg.paths.hmr4d_results).exists():
            model = self._ensure_model(cfg)
            Log.info("[HMR4D] Predicting")
            tic = Log.sync_time()
            pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            data_time = data["length"] / 30
            Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
            torch.save(pred, cfg.paths.hmr4d_results)
        else:
            Log.info(f"[HMR4D] results from {cfg.paths.hmr4d_results}")
            pred = torch.load(cfg.paths.hmr4d_results)

        if render:
            render_incam(cfg)
            render_global(cfg)
            if not Path(cfg.paths.incam_global_horiz_video).exists():
                Log.info("[Merge Videos]")
                merge_videos_horizontal(
                    [cfg.paths.incam_video, cfg.paths.global_video],
                    cfg.paths.incam_global_horiz_video,
                )

        return {
            "smpl_params_global": pred["smpl_params_global"],
            "smpl_params_incam": pred["smpl_params_incam"],
            "K_fullimg": pred["K_fullimg"],
            "output_dir": cfg.output_dir,
        }
