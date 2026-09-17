"""CLI entry point for the perception layer.

    python -m competition.vision <image>

Kept as a separate module from `quality.py` deliberately. Running
`python -m competition.vision.quality` directly would make Python import
`quality` twice — once via the package `__init__`, once as `__main__` — which
emits a RuntimeWarning about unpredictable behaviour. A dedicated `__main__`
avoids that entirely while leaving the package re-exports intact.
"""

from __future__ import annotations

from competition.vision.quality import main

if __name__ == "__main__":
    raise SystemExit(main())
