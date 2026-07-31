"""A minimal, Hydra-free stand-in for ``hydra.utils.instantiate``.

GVHMR's configs are plain ``{"_target_": "pkg.mod.Cls", **kwargs}`` maps, and that convention is
all the model code ever used Hydra's ``instantiate`` for. Reimplementing it here means the
*inference* path does not touch Hydra at all — no ``initialize_config_module``, no
``GlobalHydra`` singleton — so GVHMR can run inside a host process that already owns Hydra's
global state (Eden does this). Training still composes its configs with Hydra; this function
accepts the ``DictConfig`` that produces just as happily as a plain ``dict``.

Only the two behaviours the model code relies on are implemented: ``_target_`` resolution and
``_recursive_``. Anything already constructed is passed through unchanged, so callers may hand
in a live ``nn.Module`` instead of a config.
"""

from __future__ import annotations

import importlib
from typing import Any


def _is_config(obj: Any) -> bool:
    """True for a mapping that names a class/function to build (dict or OmegaConf DictConfig)."""
    try:
        return "_target_" in obj
    except TypeError:  # not a container
        return False


def _locate(target: str) -> Any:
    """Resolve a dotted ``pkg.module.attr`` path, trying successively shorter module prefixes.

    ``importlib`` cannot tell ``a.b.C`` (attribute ``C`` of module ``a.b``) from a module
    ``a.b.C`` without trying, so walk the split points from longest module first.
    """
    parts = target.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name, attrs = ".".join(parts[:split]), parts[split:]
        try:
            obj = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr in attrs:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"Could not resolve _target_ {target!r} to an importable object.")


def instantiate(config: Any, *, _recursive_: bool = True, **overrides: Any) -> Any:
    """Build the object described by ``config``.

    ``config`` that is not a ``_target_`` mapping is returned as-is, which is what makes an
    already-constructed module a valid substitute for its config. With ``_recursive_=False``
    nested ``_target_`` maps are passed through untouched, matching Hydra — the callers here
    all rely on that, instantiating their children themselves.
    """
    if not _is_config(config):
        return config

    kwargs = {key: config[key] for key in config if key != "_target_"}
    if _recursive_:
        kwargs = {key: instantiate(value) if _is_config(value) else value for key, value in kwargs.items()}
    kwargs.update(overrides)
    return _locate(str(config["_target_"]))(**kwargs)
