from __future__ import annotations

from typing import Optional

from text_world.neural_inverse import decode_text_to_sid


def test_symbolic_first_parser_exceptions_do_not_crash() -> None:
    def parser_clean(_: str) -> Optional[int]:
        raise ValueError("sentence must end with '.'")

    d = decode_text_to_sid(
        "hello",
        seed=0,
        symbolic_first=True,
        parser_clean=parser_clean,
        neural=None,
    )

    assert isinstance(d, dict)
    assert d["mode"] == "reject"
    assert d["reason"] == "no_neural_model_loaded"
