import json
import subprocess


def _run(tmp_path, eps: float):
    out = tmp_path / f"macro_eps_{eps:.2f}.json"
    cmd = [
        "python",
        "-m",
        "experiments.text_block_macro_beam_bench",
        str(out),
        "0",            # seed
        str(eps),       # epsilon
        "40",           # depth
        "32",           # beam
        "4",            # macro_len
        "6",            # topM
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    d = json.loads(out.read_text(encoding="utf-8"))
    bench = d["BLOCK_MACRO_BEAM_BENCH"]
    search = d["BLOCK_MACRO_BEAM_SEARCH"]
    return bench, search


def test_macro_beam_epsilon_step_at_one(tmp_path):
    bench_099, search_099 = _run(tmp_path, 0.99)
    bench_100, search_100 = _run(tmp_path, 1.00)

    # Candidate accounting identity must hold.
    assert int(search_099["n_candidates_total"]) == int(search_099["n_kept_total"]) + int(search_099["n_rejected_total"])
    assert int(search_100["n_candidates_total"]) == int(search_100["n_kept_total"]) + int(search_100["n_rejected_total"])

    # Deterministic candidate pool in this benchmark (seeded).
    assert int(search_099["n_candidates_total"]) == 374544
    assert int(search_100["n_candidates_total"]) == 374544

    # Step: ε < 1 rejects; ε = 1 rejects none.
    assert int(search_099["n_rejected_total"]) == 53343
    assert int(search_099["n_kept_total"]) == 321201
    assert int(search_100["n_rejected_total"]) == 0
    assert int(search_100["n_kept_total"]) == 374544

    # Every rejected counterfactual is a hard risk=1.0 branch.
    risks = sorted({float(x["risk_max_if_taken"]) for x in search_099["rejected_counterfactuals"]})
    assert risks == [1.0]

    # Sanity: best path remains risk 0 in both runs.
    assert float(bench_099["best_risk_max"]) == 0.0
    assert float(bench_100["best_risk_max"]) == 0.0
