from __future__ import annotations

from typing import Any, List

from text_world.render_parse_clean import render_sentence_clean


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _render_paragraph_clean(p: Any) -> str:
    s1 = _get_attr(p, "s1", None)
    if s1 is None:
        return ""
    return render_sentence_clean(s1)


def render_block_clean(block: Any) -> str:
    paras = _get_attr(block, "paras", None)
    if paras is None:
        return ""
    out: List[str] = []
    for p in list(paras):
        txt = _render_paragraph_clean(p).strip()
        if txt:
            out.append(txt)
    return "\n\n".join(out)
