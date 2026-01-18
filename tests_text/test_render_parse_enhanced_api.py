from __future__ import annotations


def test_render_sentence_enhanced_exists_and_returns_str() -> None:
    import text_world.render_parse_enhanced as m

    assert hasattr(m, "render_sentence_enhanced")
    out = m.render_sentence_enhanced(0)
    assert isinstance(out, str)
    assert len(out) > 0
