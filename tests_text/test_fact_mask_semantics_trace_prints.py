from __future__ import annotations

from text_world.paragraph import normalize_paragraph, render_paragraph_clean, parse_paragraph_clean
from text_world.state import enumerate_states


def _pick_one_per_fact_mask():
    states = enumerate_states()
    by_mask = {}
    for s in states:
        fm = int(getattr(s, "fact_mask"))
        if fm not in by_mask:
            by_mask[fm] = s
        if len(by_mask) >= 64:
            break
    return by_mask


def test_fact_mask_semantics_trace_prints():
    by_mask = _pick_one_per_fact_mask()
    masks = sorted(by_mask.keys())

    print()
    print("=" * 96)
    print("TRACE: FACT_MASK SEMANTICS (one representative SentenceState per fact_mask)")
    print("=" * 96)
    print(f"fact_masks={masks} count={len(masks)}")

    for fm in masks:
        s = by_mask[fm]
        p = normalize_paragraph(s, s, s)
        txt = render_paragraph_clean(p)
        p2 = parse_paragraph_clean(txt)

        print()
        print("-" * 96)
        print(f"FACT_MASK={fm}")
        print("REPRESENTATIVE SENTENCE STATE:")
        print(s)
        print()
        print("RENDERED TEXT (3-sentence surface):")
        print(txt)
        print()
        print("PARSED BACK (ParagraphState):")
        print(p2)
        print()
        print("ROUNDTRIP CHECK: p2 == p  ->", (p2 == p))

    assert len(masks) >= 1
