from __future__ import annotations

import os
import sys
from typing import Optional

def _mode() -> str:
    return os.environ.get("TEXTWORLD_MODE", "debug").strip().lower()

def emit_pass(msg: str) -> None:
    """
    PASS/debug output channel.
    - debug mode: emit to stderr (visible to developer)
    - chat mode: silent (or optionally file)
    """
    mode = _mode()
    if mode == "chat":
        return
    print(msg, file=sys.stderr)

def emit_user(msg: str) -> None:
    """
    User-facing output channel.
    - Always stdout.
    - This is the ONLY thing allowed to reach the user in chat mode.
    """
    print(msg, file=sys.stdout)
