from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from text_world.state import SentenceState
from text_world.render_parse_clean import render_sentence_clean, parse_sentence_clean

@dataclass(frozen=True)
class ParagraphState:
    s1: SentenceState
    s2: SentenceState
    s3: SentenceState
    rho: int  # 0,1,2

def coherence_bucket(s1: SentenceState, s2: SentenceState, s3: SentenceState) -> int:
    styles_match = int((s1.style == s2.style) and (s2.style == s3.style))
    any_contra = int((s1.contradiction == 1) or (s2.contradiction == 1) or (s3.contradiction == 1))
    if styles_match == 1 and any_contra == 0:
        return 2
    if styles_match == 1 and any_contra == 1:
        return 1
    if styles_match == 0 and any_contra == 0:
        return 1
    return 0

def normalize_paragraph(s1: SentenceState, s2: SentenceState, s3: SentenceState) -> ParagraphState:
    rho = coherence_bucket(s1, s2, s3)
    return ParagraphState(s1=s1, s2=s2, s3=s3, rho=rho)

def render_paragraph_clean(p: ParagraphState) -> str:
    # rho is derived; paragraph rendering is just 3 clean sentences separated by a space
    return f"{render_sentence_clean(p.s1)} {render_sentence_clean(p.s2)} {render_sentence_clean(p.s3)}"

def parse_paragraph_clean(text: str) -> ParagraphState:
    # We assume exactly 3 sentences, each ending with '.'
    t = text.strip()
    if not t:
        raise ValueError("empty paragraph")
    parts = [x.strip() for x in t.split(".") if x.strip() != ""]
    if len(parts) != 3:
        raise ValueError("paragraph must contain exactly 3 sentences")
    sents = [p + "." for p in parts]
    s1 = parse_sentence_clean(sents[0])
    s2 = parse_sentence_clean(sents[1])
    s3 = parse_sentence_clean(sents[2])
    return normalize_paragraph(s1, s2, s3)
