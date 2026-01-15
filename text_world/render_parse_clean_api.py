from __future__ import annotations

from text_world.state import SentenceState
from text_world.render_parse_clean import (
    parse_sentence_clean,
    render_sentence_clean,
    _FACT_NEUTRAL,
)

FACT_BITS = len(_FACT_NEUTRAL)   # 4 when you have 4 facts
LEN_BITS = 2                    # 4 values
STYLE_BITS = 1
CONTRA_BITS = 1

FACT_MASK = (1 << FACT_BITS) - 1
LEN_MASK = (1 << LEN_BITS) - 1


def state_id_to_sentence_state(state_id: int) -> SentenceState:
    sid = int(state_id)

    fact_mask = sid & FACT_MASK
    sid >>= FACT_BITS

    contradiction = sid & 1
    sid >>= CONTRA_BITS

    style = sid & 1
    sid >>= STYLE_BITS

    length = sid & LEN_MASK

    return SentenceState(
        fact_mask=int(fact_mask),
        contradiction=int(contradiction),
        style=int(style),
        length=int(length),
    )


def sentence_state_to_state_id(st: SentenceState) -> int:
    fact_mask = int(st.fact_mask) & FACT_MASK
    contradiction = int(st.contradiction) & 1
    style = int(st.style) & 1
    length = int(st.length) & LEN_MASK

    sid = 0
    sid |= fact_mask
    sid |= contradiction << FACT_BITS
    sid |= style << (FACT_BITS + CONTRA_BITS)
    sid |= length << (FACT_BITS + CONTRA_BITS + STYLE_BITS)
    return int(sid)


def render_state_clean(state_id: int) -> str:
    st = state_id_to_sentence_state(int(state_id))
    return render_sentence_clean(st)


def parse_state_clean(text: str) -> int:
    st = parse_sentence_clean(str(text))
    return sentence_state_to_state_id(st)
