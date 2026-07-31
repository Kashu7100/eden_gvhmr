"""Hydra config store for training.

Hydra is a **training-only** dependency (the ``train`` extra). The model modules register their
config groups here at import time, and they are imported by inference too, so this module has to
stay importable when Hydra is absent — hence the fallbacks below. Inference does not compose
configs at all: it builds the model from :data:`hmr4d.demo.pipeline.DEMO_MODEL_CFG` via
:func:`hmr4d.utils.instantiate.instantiate`, and never touches Hydra's global state. That is what
lets GVHMR run inside a host process which owns Hydra itself.
"""

import argparse
import os

HYDRA_AVAILABLE = True
try:
    from hydra.core.config_store import ConfigStore
    from hydra_zen import builds  # noqa: F401  (re-exported; model modules import it from here)

    os.environ["HYDRA_FULL_ERROR"] = "1"
    MainStore = ConfigStore.instance()
except ImportError:  # inference-only install
    HYDRA_AVAILABLE = False

    class _UnavailableNode:
        """Stand-in for a hydra-zen config node.

        Callable and self-returning because the registration sites both build a node and then
        specialize it — ``cfg_base = builds(...)`` followed by ``cfg_base(stats_name=...)``.
        """

        def __call__(self, *args, **kwargs):
            return self

    def builds(*args, **kwargs):  # noqa: D103 - registration is a no-op without Hydra
        return _UnavailableNode()

    class _UnavailableStore:
        """Swallows the module-scope ``MainStore.store(...)`` calls when Hydra is not installed."""

        def store(self, *args, **kwargs):
            return None

    MainStore = _UnavailableStore()


def _require_hydra(what):
    if not HYDRA_AVAILABLE:
        raise ImportError(
            f"{what} needs Hydra, which is a training-only dependency here. Install it with "
            "`pip install eden-gvhmr[train]`. Inference does not require it."
        )


def register_store_gvhmr():
    """Register group options to MainStore"""
    _require_hydra("register_store_gvhmr()")
    from . import store_gvhmr  # noqa: F401


def parse_args_to_cfg():
    """
    Use minimal Hydra API to parse args and return cfg.
    This function don't do _run_hydra which create log file hierarchy.
    """
    _require_hydra("parse_args_to_cfg()")
    from hydra import compose, initialize_config_module

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", "-cn", default="train")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    return cfg
