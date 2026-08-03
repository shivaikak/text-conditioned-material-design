from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import get_nested, set_nested


def test_nested_configuration_change() -> None:
    config = {"training": {"lr": 0.0003}}
    old = set_nested(config, "training.lr", 0.0002)
    assert old == 0.0003
    assert get_nested(config, "training.lr") == 0.0002
