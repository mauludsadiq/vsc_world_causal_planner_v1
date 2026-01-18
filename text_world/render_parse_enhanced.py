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


_ENHANCED_RP_SINGLETON = None


def _enhanced_singleton() -> "EnhancedRenderParse":
    global _ENHANCED_RP_SINGLETON
    if _ENHANCED_RP_SINGLETON is None:
        _ENHANCED_RP_SINGLETON = EnhancedRenderParse()
    return _ENHANCED_RP_SINGLETON


def render_sentence_enhanced(state_id: int, strength: float = 0.75, sugar: bool = True) -> str:
    return _enhanced_singleton().render(int(state_id), strength=float(strength), sugar=bool(sugar))


_ENHANCED_RP_SINGLETON = None


def _enhanced_singleton() -> "EnhancedRenderParse":
    global _ENHANCED_RP_SINGLETON
    if _ENHANCED_RP_SINGLETON is None:
        try:
            from text_world.render_parse_clean_api import render_sentence_clean, parse_sentence_clean
        except Exception:
            from text_world.render_parse_clean import render_sentence_clean, parse_sentence_clean

        def base_render(state_id: int) -> str:

            from dataclasses import MISSING

            from text_world.render_parse_clean import SentenceState, STYLE_FORMAL, STYLE_NEUTRAL

        

            sid = int(state_id)

            style = STYLE_FORMAL if (sid & 0x80) else STYLE_NEUTRAL

            mask = sid & 0x7F

        

            fields = getattr(SentenceState, '__dataclass_fields__', {})

            keys = list(fields.keys())

        

            kw = {}

            if 'style' in fields:

                kw['style'] = style

        

            mask_key = None

            for k in (

                'mask',

                'fact_mask',

                'facts',

                'factbits',

                'fact_bits',

                'facts_mask',

                'bits',

                'state_mask',

            ):

                if k in fields:

                    mask_key = k

                    break

        

            if mask_key is None:

                raise RuntimeError(f'Cannot map sid->SentenceState: fields={keys}')

        

            kw[mask_key] = mask

        

            for name, f in fields.items():

                if name in kw:

                    continue

                if f.default is not MISSING:

                    continue

                if getattr(f, 'default_factory', MISSING) is not MISSING:

                    continue

                kw[name] = 0

        

            st = SentenceState(**kw)

            return render_sentence_clean(st)

        

        def base_parse(text: str):
            return parse_sentence_clean(str(text))

        _ENHANCED_RP_SINGLETON = EnhancedRenderParse(
            base_render=base_render,
            base_parse=base_parse,
        )

    return _ENHANCED_RP_SINGLETON


def render_sentence_enhanced(state_id: int, strength: float = 0.75, sugar: bool = True) -> str:
    return _enhanced_singleton().render(int(state_id), strength=float(strength), sugar=bool(sugar))
