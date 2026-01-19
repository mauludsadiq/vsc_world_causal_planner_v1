from __future__ import annotations

from typing import Dict, List, Tuple

from text_world.state import enumerate_states
from text_world.paragraph import normalize_paragraph, render_paragraph_clean, parse_paragraph_clean


def _banner(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def test_trace_sentence_parse_catalog_prints():
    _banner("TRACE: SENTENCE PARSE CATALOG (SentenceState -> text -> SentenceState)")

    states = enumerate_states()
    assert len(states) > 0

    seen: Dict[str, Tuple[object, object]] = {}
    rows: List[Tuple[int, object, str, object]] = []

    target_unique = 18

    for i, s in enumerate(states):
        p = normalize_paragraph(s, s, s)
        txt = render_paragraph_clean(p)
        p2 = parse_paragraph_clean(txt)

        ok = (p2 == p)

        if txt not in seen:
            seen[txt] = (p, p2)
            rows.append((i, s, txt, p2))
            if len(rows) >= target_unique:
                break

        assert ok

    print(f"unique_texts={len(rows)} sampled_from={len(states)}")

    for (i, s, txt, p2) in rows:
        print("\n" + "-" * 96)
        print(f"INDEX={i}")
        print("SYMBOLIC SENTENCE STATE:")
        print(s)
        print("\nRENDERED TEXT (3-sentence surface):")
        print(txt)
        print("\nPARSED BACK (ParagraphState):")
        print(p2)

    assert len(rows) >= 3
