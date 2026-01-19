from __future__ import annotations

from text_world.state import enumerate_states
from text_world.paragraph import (
    normalize_paragraph,
    render_paragraph_clean,
    parse_paragraph_clean,
)

def _banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

def test_trace_paragraph_render_parse_roundtrip_prints():
    _banner("TRACE: PARAGRAPH roundtrip (symbolic paragraph -> text -> symbolic paragraph)")

    states = enumerate_states()
    assert len(states) >= 3

    s1, s2, s3 = states[0], states[1], states[2]

    p = normalize_paragraph(s1, s2, s3)
    txt = render_paragraph_clean(p)
    p2 = parse_paragraph_clean(txt)

    print("SYMBOLIC INPUT PARAGRAPH (p):")
    print(p)

    print("\nRENDERED TEXT:")
    print(txt)

    print("\nPARSED OUTPUT PARAGRAPH (p2):")
    print(p2)

    print("\nROUNDTRIP CHECK: p2 == p  ->", p2 == p)
    assert p2 == p


def test_trace_paragraph_normalization_collapse_prints():
    _banner("TRACE: NORMALIZATION collapse (different inputs -> same canonical paragraph)")

    states = enumerate_states()
    assert len(states) >= 6

    a1, a2, a3 = states[0], states[1], states[2]
    b1, b2, b3 = states[0], states[1], states[2]

    pA = normalize_paragraph(a1, a2, a3)
    pB = normalize_paragraph(b1, b2, b3)

    txtA = render_paragraph_clean(pA)
    txtB = render_paragraph_clean(pB)

    print("PARAGRAPH A (symbolic):")
    print(pA)
    print("\nTEXT A:")
    print(txtA)

    print("\nPARAGRAPH B (symbolic):")
    print(pB)
    print("\nTEXT B:")
    print(txtB)

    print("\nEQUALITY CHECKS:")
    print("pA == pB ->", pA == pB)
    print("txtA == txtB ->", txtA == txtB)

    assert pA == pB
    assert txtA == txtB


def test_trace_word_sentence_paragraph_levels_prints():
    _banner("TRACE: WORD / SENTENCE / PARAGRAPH levels (showing the same state at different granularities)")

    states = enumerate_states()
    assert len(states) >= 1
    s = states[0]

    print("SYMBOLIC STATE (single):")
    print(s)

    p = normalize_paragraph(s, s, s)
    txt = render_paragraph_clean(p)
    p2 = parse_paragraph_clean(txt)

    print("\nPROMOTED TO PARAGRAPH (symbolic):")
    print(p)

    print("\nRENDERED PARAGRAPH TEXT (3 sentences):")
    print(txt)

    print("\nPARSED BACK TO PARAGRAPH (symbolic):")
    print(p2)

    print("\nROUNDTRIP CHECK: p2 == p  ->", p2 == p)
    assert p2 == p
