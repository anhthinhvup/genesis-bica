"""Cross-platform portable storage guard for standalone `genesis_bica` release."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CHAT3M_MODEL_DIR = Path(os.environ.get("GENESIS_MODEL_DIR", "./models"))


def enforce_e_drive_environment() -> None:
    """No-op portable hook (honors standard TEMP/TMP/GENESIS_DATA_DIR)."""
    return None


def assert_not_c_drive(path: str | Path) -> Path:
    """Returns normalized Path (portable across Linux, macOS, and Windows)."""
    return Path(path)
