from __future__ import annotations
import re
from typing import List

from text_world.state import (
    SentenceState,
    REQ_FACTS,
    STYLE_NEUTRAL,
    STYLE_FORMAL,
    LEN_SHORT,
    LEN_MED,
    LEN_LONG,
)

_TAG_RE = re.compile(r"<M=(\d+);C=(\d+);S=(\d+);L=(\d+)>$")

_FACT_TPL_NEUTRAL = [
    "the device supports wireless charging",
    "the battery lasts all day",
    "the screen is shatter-resistant",
]
_FACT_TPL_FORMAL = [
    "the device supports wireless charging",
    "the battery sustains all-day operation",
    "the display exhibits enhanced shatter resistance",
]

_CONNECTOR_BY_LEN = {
    LEN_SHORT: ".",
    LEN_MED: "; additionally,",
    LEN_LONG: "; furthermore,",
}

def render_sentence(st: SentenceState) -> str:
    tpl = _FACT_TPL_FORMAL if st.style == STYLE_FORMAL else _FACT_TPL_NEUTRAL
    parts: List[str] = []

    for i in range(3):
        if (st.fact_mask >> i) & 1:
            parts.append(tpl[i])

    if not parts:
        core = "the statement is unspecified"
    else:
        conn = _CONNECTOR_BY_LEN[st.length]
        if st.length == LEN_SHORT:
            core = parts[0]
            if len(parts) > 1:
                core = core + " and " + parts[1]
            if len(parts) > 2:
                core = core + " and " + parts[2]
        else:
            core = parts[0]
            for p in parts[1:]:
                core = core + conn + " " + p

    if st.contradiction == 1:
        core = core + "; however, the sentence also asserts the opposite"

    if st.style == STYLE_FORMAL:
        sentence = core[0].upper() + core[1:] + "."
    else:
        sentence = core + "."

    tag = f"<M={st.fact_mask};C={st.contradiction};S={st.style};L={st.length}>"
    return sentence + " " + tag

def parse_sentence(text: str) -> SentenceState:
    text = text.strip()
    m = _TAG_RE.search(text)
    if not m:
        raise ValueError("missing canonical state tag at end of sentence")
    fact_mask = int(m.group(1))
    contradiction = int(m.group(2))
    style = int(m.group(3))
    length = int(m.group(4))
    if fact_mask < 0 or fact_mask > 7:
        raise ValueError("fact_mask out of range")
    if contradiction not in (0, 1):
        raise ValueError("contradiction out of range")
    if style not in (STYLE_NEUTRAL, STYLE_FORMAL):
        raise ValueError("style out of range")
    if length not in (LEN_SHORT, LEN_MED, LEN_LONG):
        raise ValueError("length out of range")
    return SentenceState(fact_mask=fact_mask, contradiction=contradiction, style=style, length=length)
