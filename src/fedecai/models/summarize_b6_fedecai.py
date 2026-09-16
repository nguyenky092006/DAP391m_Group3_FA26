"""Guard the FedECAI summary until a human selects a backbone and new runs exist."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    config = json.loads((ROOT / "configs/models/b6_fedecai_v1.json").read_text(encoding="utf-8"))
    if config.get("backbone_status") != "SELECTED" or not config.get("selected_backbone"):
        raise RuntimeError(
            "FedECAI summary is unavailable: the centralized candidate table and "
            "explicit human backbone selection are required first"
        )
    raise FileNotFoundError(
        "No active new-protocol E1-E7 result set exists; do not reuse quarantined legacy reports"
    )


if __name__ == "__main__":
    main()
