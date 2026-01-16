from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from text_world.agent.intent_schema import SpeakIntent


@dataclass(frozen=True)
class SpeakerSurface:
    seed: int

    def _rng(self, turn: int, sid: Optional[int]) -> random.Random:
        sid_term = 0 if sid is None else int(sid)
        return random.Random(self.seed + 1000003 * int(turn) + 37 * sid_term)

    def speak(self, base_text: str, intent: SpeakIntent, turn: int) -> str:
        r = self._rng(turn=turn, sid=intent.sid)

        base = base_text.strip()
        if not base:
            base = "the claim is unspecified"

        strength = float(intent.strength)
        strength = 0.0 if strength < 0.0 else strength
        strength = 1.0 if strength > 1.0 else strength

        prefixes_soft = [
            "it seems that ",
            "it is likely that ",
            "from the available evidence, ",
            "one reasonable conclusion is that ",
        ]
        prefixes_mid = [
            "the conclusion is that ",
            "the correct outcome is that ",
            "the system indicates that ",
            "the result is that ",
        ]
        prefixes_hard = [
            "it is established that ",
            "it is confirmed that ",
            "it is verified that ",
            "the system proves that ",
        ]

        suffixes_soft = [
            ".",
            " based on current constraints.",
            " under the present conditions.",
            " as written.",
        ]
        suffixes_mid = [
            ".",
            " and this follows from the state.",
            " under the policy selection rule.",
            " with the chosen safe action.",
        ]
        suffixes_hard = [
            ".",
            " and no unsafe branch is permitted.",
            " with safety enforced by constraint.",
            " with the rejected branch made explicit.",
        ]

        if strength < 0.35:
            prefix_pool = prefixes_soft
            suffix_pool = suffixes_soft
        elif strength < 0.75:
            prefix_pool = prefixes_mid
            suffix_pool = suffixes_mid
        else:
            prefix_pool = prefixes_hard
            suffix_pool = suffixes_hard

        pref = r.choice(prefix_pool)
        suf = r.choice(suffix_pool)

        tone = (intent.tone or "neutral").strip().lower()
        if tone == "formal":
            pref = pref.replace("it seems", "it appears").replace("likely", "probable")
        elif tone == "casual":
            pref = pref.replace("it is established that ", "ok, so ").replace("it is confirmed that ", "yep: ")

        out = pref + base
        if not out.endswith(".") and not suf.startswith("."):
            out = out + "."
        out = out + suf if suf.startswith(" ") or suf.startswith(".") else (" " + suf)

        out = " ".join(out.split())
        return out
