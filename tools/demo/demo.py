"""Single-video GVHMR demo CLI (repo-local shim).

The implementation lives in the installable package so it also works after
``pip install`` (``python -m hmr4d.demo`` or the ``gvhmr-demo`` console script) and
is importable programmatically:

    from hmr4d.demo import GVHMR
    GVHMR().recover("video.mp4", static_cam=True)
"""

from hmr4d.demo.__main__ import main

if __name__ == "__main__":
    main()
