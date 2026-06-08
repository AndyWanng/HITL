from __future__ import annotations

import runpy
import sys
from pathlib import Path


NEXT_ROOT = Path(__file__).resolve().parent
PIPELINE = NEXT_ROOT / "scripts" / "run_next_pipeline.py"
DEFAULT_CONFIG = NEXT_ROOT / "configs" / "server_next.yaml"


def main() -> None:
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
    runpy.run_path(str(PIPELINE), run_name="__main__")


if __name__ == "__main__":
    main()
