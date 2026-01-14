from __future__ import annotations
from typing import Dict, List, Tuple

from text_world.state import (
    SentenceState,
    STYLE_NEUTRAL,
    STYLE_FORMAL,
    LEN_SHORT,
    LEN_MED,
    LEN_LONG,
)

_FACT_NEUTRAL = [
    "the device supports Qi wireless charging",
    "the battery lasts through a full day of typical use",
    "the display resists cracks from minor drops",
]

_FACT_FORMAL = [
    "the device supports Qi-compatible wireless charging",
    "the battery sustains a full day of typical operation",
    "the display exhibits improved resistance to cracking from minor drops",
]

_UNSPEC_BY_LEN = {
    LEN_SHORT: "the claim is unspecified",
    LEN_MED: "the claim is unspecified pending details",
    LEN_LONG: "the claim is unspecified pending additional evidence",
}

_CONN_SHORT = " and "
_CONN_MED = "; additionally, "
_CONN_LONG = "; furthermore, "

_SINGLE_SUFFIX_BY_LEN = {
    LEN_SHORT: "",
    LEN_MED: ", briefly",
    LEN_LONG: ", with additional detail",
}

_CONTRADICT_CLAUSE = "; however, the sentence also asserts the opposite"


def render_sentence_clean(st: SentenceState) -> str:
    tpl = _FACT_FORMAL if st.style == STYLE_FORMAL else _FACT_NEUTRAL

    facts: List[str] = []
    for i in range(3):
        if (st.fact_mask >> i) & 1:
            facts.append(tpl[i])

    if not facts:
        core = _UNSPEC_BY_LEN[st.length]
    else:
        if len(facts) == 1:
            core = facts[0] + _SINGLE_SUFFIX_BY_LEN[st.length]
        else:
            if st.length == LEN_SHORT:
                core = facts[0]
                for f in facts[1:]:
                    core = core + _CONN_SHORT + f
            elif st.length == LEN_MED:
                core = facts[0]
                for f in facts[1:]:
                    core = core + _CONN_MED + f
            else:
                core = facts[0]
                for f in facts[1:]:
                    core = core + _CONN_LONG + f

    if st.contradiction == 1:
        core = core + _CONTRADICT_CLAUSE

    if st.style == STYLE_FORMAL:
        core = core[0].upper() + core[1:]

    return core + "."


def render_sentence_sidecar(st: SentenceState) -> Tuple[str, Dict[str, int]]:
    return render_sentence_clean(st), {
        "M": st.fact_mask,
        "C": st.contradiction,
        "S": st.style,
        "L": st.length,
    }


def parse_sentence_clean(text: str) -> SentenceState:
    t = text.strip()
    if not t.endswith("."):
        raise ValueError("sentence must end with '.'")
    t = t[:-1]

    contradiction = 0
    if t.endswith(_CONTRADICT_CLAUSE):
        contradiction = 1
        t = t[: -len(_CONTRADICT_CLAUSE)]

    style = STYLE_FORMAL if (t[:1].isupper()) else STYLE_NEUTRAL
    t_norm = t[0].lower() + t[1:] if t else t

    for L, u in _UNSPEC_BY_LEN.items():
        if t_norm == u:
            return SentenceState(fact_mask=0, contradiction=contradiction, style=style, length=L)

    facts = _FACT_FORMAL if style == STYLE_FORMAL else _FACT_NEUTRAL

    for L, suf in _SINGLE_SUFFIX_BY_LEN.items():
        if suf == "":
            continue
        if t_norm.endswith(suf):
            base = t_norm[: -len(suf)].strip()
            if base in facts:
                m = 1 << facts.index(base)
                return SentenceState(fact_mask=m, contradiction=contradiction, style=style, length=L)

    if t_norm in facts:
        m = 1 << facts.index(t_norm)
        return SentenceState(fact_mask=m, contradiction=contradiction, style=style, length=LEN_SHORT)

    if _CONN_LONG in t_norm:
        parts = [p.strip() for p in t_norm.split(_CONN_LONG) if p.strip()]
        L = LEN_LONG
    elif _CONN_MED in t_norm:
        parts = [p.strip() for p in t_norm.split(_CONN_MED) if p.strip()]
        L = LEN_MED
    elif _CONN_SHORT in t_norm:
        parts = [p.strip() for p in t_norm.split(_CONN_SHORT) if p.strip()]
        L = LEN_SHORT
    else:
        raise ValueError("text not in restricted grammar; cannot parse deterministically")

    m = 0
    for p in parts:
        if p not in facts:
            raise ValueError("unknown fact surface form")
        m |= (1 << facts.index(p))

    return SentenceState(fact_mask=m, contradiction=contradiction, style=style, length=L)
