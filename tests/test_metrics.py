from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate_generated import rank_biserial_from_u


def test_rank_biserial_bounds() -> None:
    assert rank_biserial_from_u(0, 10, 10) == 1.0
    assert rank_biserial_from_u(100, 10, 10) == -1.0
