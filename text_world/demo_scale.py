from __future__ import annotations
import os

def is_fast() -> bool:
    return os.environ.get("TEXT_DEMO_FAST", "0") == "1"

def sent_samples(default: int) -> int:
    if is_fast():
        return int(os.environ.get("TEXT_DEMO_SENT_SAMPLES", "4000"))
    return int(os.environ.get("TEXT_DEMO_SENT_SAMPLES", str(default)))

def anchors(default: int) -> int:
    if is_fast():
        return int(os.environ.get("TEXT_DEMO_ANCHORS", "8"))
    return int(os.environ.get("TEXT_DEMO_ANCHORS", str(default)))

def reps(default: int) -> int:
    if is_fast():
        return int(os.environ.get("TEXT_DEMO_REPS", "3"))
    return int(os.environ.get("TEXT_DEMO_REPS", str(default)))

def trials(default: int) -> int:
    if is_fast():
        return int(os.environ.get("TEXT_DEMO_TRIALS", "50"))
    return int(os.environ.get("TEXT_DEMO_TRIALS", str(default)))
