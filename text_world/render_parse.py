from __future__ import annotations

import re
from typing import Dict, List, Tuple

from text_world.state import (
    SentenceState,
    STYLE_NEUTRAL,
    STYLE_FORMAL,
    LEN_SHORT,
    LEN_MED,
    LEN_LONG,
    LEN_XLONG,
)

_TAG_RE = re.compile(r"<M=(\d+);C=(\d+);S=(\d+);L=(\d+)>\s*$")

_FACT_NEUTRAL = [
    "the device supports Qi wireless charging",
    "the battery lasts through a full day of typical use",
    "the display resists cracks from minor drops",
    "the camera performs well in low light",
]

_FACT_FORMAL = [
    "the device supports Qi-compatible wireless charging",
    "the battery sustains a full day of typical operation",
    "the display exhibits improved resistance to cracking from minor drops",
    "the camera demonstrates strong low-light performance",
]

_UNSPEC_BY_LEN = {
    LEN_SHORT: "the statement is unspecified",
    LEN_MED: "the statement is unspecified pending details",
    LEN_LONG: "the statement is unspecified pending additional evidence",
    LEN_XLONG: "the statement is unspecified pending exhaustive detail",
}

_CONN_SHORT = " and "
_CONN_MED = "; additionally, "
_CONN_LONG = "; furthermore, "
_CONN_XLONG = "; moreover, "

_SINGLE_SUFFIX_BY_LEN = {
    LEN_SHORT: "",
    LEN_MED: ", briefly",
    LEN_LONG: ", with additional detail",
    LEN_XLONG: ", with exhaustive detail",
}

_CONTRADICT_CLAUSE = "; however, the sentence also asserts the opposite"


def _tag(st: SentenceState) -> str:
    return f"<M={int(st.fact_mask)};C={int(st.contradiction)};S={int(st.style)};L={int(st.length)}>"


def render_sentence(st: SentenceState) -> str:
    tpl = _FACT_FORMAL if st.style == STYLE_FORMAL else _FACT_NEUTRAL

    facts: List[str] = []
    for i in range(len(tpl)):
        if (int(st.fact_mask) >> i) & 1:
            facts.append(tpl[i])

    if not facts:
        core = _UNSPEC_BY_LEN[int(st.length)]
    else:
        if len(facts) == 1:
            core = facts[0] + _SINGLE_SUFFIX_BY_LEN[int(st.length)]
        else:
            if int(st.length) == LEN_SHORT:
                core = facts[0]
                for f in facts[1:]:
                    core = core + _CONN_SHORT + f
            elif int(st.length) == LEN_MED:
                core = facts[0]
                for f in facts[1:]:
                    core = core + _CONN_MED + f
            elif int(st.length) == LEN_LONG:
                core = facts[0]
                for f in facts[1:]:
                    core = core + _CONN_LONG + f
            else:
                core = facts[0]
                for f in facts[1:]:
                    core = core + _CONN_XLONG + f

    if int(st.contradiction) == 1:
        core = core + _CONTRADICT_CLAUSE

    if int(st.style) == STYLE_FORMAL and core:
        core = core[0].upper() + core[1:]

    return core + ". " + _tag(st)


def render_sentence_sidecar(st: SentenceState) -> Tuple[str, Dict[str, int]]:
    return render_sentence(st), {
        "M": int(st.fact_mask),
        "C": int(st.contradiction),
        "S": int(st.style),
        "L": int(st.length),
    }


def parse_sentence(text: str) -> SentenceState:
    t = str(text).strip()
    m = _TAG_RE.search(t)
    if not m:
        raise ValueError("missing canonical state tag at end of sentence")

    fact_mask = int(m.group(1))
    contradiction = int(m.group(2))
    style = int(m.group(3))
    length = int(m.group(4))

    if fact_mask < 0 or fact_mask > 15:
        raise ValueError("fact_mask out of range")
    if contradiction not in (0, 1):
        raise ValueError("contradiction out of range")
    if style not in (STYLE_NEUTRAL, STYLE_FORMAL):
        raise ValueError("style out of range")
    if length not in (LEN_SHORT, LEN_MED, LEN_LONG, LEN_XLONG):
        raise ValueError("length out of range")

    return SentenceState(fact_mask=fact_mask, contradiction=contradiction, style=style, length=length)
