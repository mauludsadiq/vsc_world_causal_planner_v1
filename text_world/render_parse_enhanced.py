from __future__ import annotations

import re

_CONF_RE = re.compile(r"^\s*CONF\((?P<c>[0-9]+(?:\.[0-9]+)?)\):\s*")

def _normalize_base_sentence(t: str) -> str:
    s = str(t).strip()

    s = s.strip(" \t\r\n")
    s = s.lstrip(", ").strip()

    s = re.sub(r"\s+", " ", s).strip()

    s = re.sub(r"\s*,\s*\.\s*$", ".", s)
    s = re.sub(r"\.\s*,\s*$", ".", s)

    s = re.sub(r"\.{2,}$", ".", s)

    if not s.endswith("."):
        s = s + "."

    return s

class EnhancedRenderParse:
    def __init__(self, base_render, base_parse):
        self.base_render = base_render
        self.base_parse = base_parse

    def render(self, state_id: int, strength: float = 0.75, sugar: bool = True) -> str:
        base = str(self.base_render(int(state_id))).strip()

        if sugar:
            s = max(0.0, min(1.0, float(strength)))
            return f"CONF({s:.3f}): {base}"

        return base

    def parse(self, text: str):
        t = str(text).strip()
        conf = 1.0

        m = _CONF_RE.match(t)
        if m:
            conf = float(m.group("c"))
            t = t[m.end():].strip()

        stripped = _normalize_base_sentence(t)
        state_id = int(self.base_parse(stripped))
        return state_id, conf
