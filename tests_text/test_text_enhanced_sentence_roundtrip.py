from __future__ import annotations

from text_world.render_parse_clean_api import render_state_clean, parse_state_clean
from text_world.render_parse_enhanced import EnhancedRenderParse


def test_text_enhanced_sentence_roundtrip_256():
    rp = EnhancedRenderParse(render_state_clean, parse_state_clean)
    for sid in range(256):
        txt = rp.render(sid, strength=0.65, sugar=True)
        sid2, conf = rp.parse(txt)
        assert sid2 == sid
        assert 0.0 <= conf <= 1.0
