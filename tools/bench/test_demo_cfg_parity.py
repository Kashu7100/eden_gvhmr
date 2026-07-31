"""Check that the Hydra-free demo config still matches the Hydra one.

Inference builds its model from ``hmr4d.demo.pipeline.DEMO_MODEL_CFG`` instead of composing
``hmr4d/configs/demo.yaml``, so the two descriptions of the same model could drift apart — and a
drift would land as a checkpoint loaded into a mismatched architecture. This composes the Hydra
config and asserts they agree, field for field.

Standalone script, matching the repo's other tools/bench/* checks. Exits 0 on pass, non-zero on
failure; skips (still 0) when Hydra is not installed, since it is a training-only dependency.

Usage:
    python tools/bench/test_demo_cfg_parity.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _to_plain(node):
    """Normalize a config node (dict / DictConfig / SimpleNamespace) to nested plain dicts.

    Explicit isinstance checks, not duck typing: a ``hasattr(node, "__dict__")`` probe recurses
    forever on ordinary objects, whose ``__dict__`` contains descriptors that have one too.
    """
    from types import SimpleNamespace

    if isinstance(node, SimpleNamespace):
        node = vars(node)
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(node, DictConfig):
            node = OmegaConf.to_container(node, resolve=True)
    except ImportError:
        pass
    if isinstance(node, dict):
        return {key: _to_plain(value) for key, value in node.items()}
    return node


def main() -> int:
    try:
        from hydra import compose, initialize_config_module
    except ImportError:
        print("SKIP  Hydra not installed (training-only dependency); nothing to compare against.")
        return 0

    from hmr4d.configs import register_store_gvhmr
    from hmr4d.demo.pipeline import DEMO_MODEL_CFG

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=parity"])

    hydra_model = _to_plain(cfg.model)
    python_model = _to_plain(DEMO_MODEL_CFG)

    if hydra_model == python_model:
        print("PASS  DEMO_MODEL_CFG matches the composed Hydra config")
        return 0

    print("FAIL  DEMO_MODEL_CFG has drifted from hmr4d/configs/demo.yaml")
    for path, a, b in _diff(hydra_model, python_model):
        print(f"        {path}: hydra={a!r}  python={b!r}")
    return 1


def _diff(a, b, path=""):
    """Yield (path, hydra_value, python_value) for every leaf that differs."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            yield from _diff(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}" if path else key)
    elif a != b:
        yield path, a, b


if __name__ == "__main__":
    sys.exit(main())
