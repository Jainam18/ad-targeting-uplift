"""Read and write the JSON results that the memo and figures are built from."""

import json
from pathlib import Path
from typing import Any

from uplift.config import RESULTS


def write_results(name: str, payload: dict[str, Any], results_dir: Path = RESULTS) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_results(name: str, results_dir: Path = RESULTS) -> dict[str, Any]:
    return json.loads((results_dir / f"{name}.json").read_text())
