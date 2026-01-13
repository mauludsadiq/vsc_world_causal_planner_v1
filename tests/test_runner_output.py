import sys, subprocess, pathlib

def test_runner_prints_explicit_pass_lines():
    root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "vsc_repo.run", "--seed", "0"]
    p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + "\n" + p.stderr
    out = p.stdout
    assert "[PASS]" in out
    for name in [
        "SCM_DO_EFFECT_BACKDOOR",
        "WORLD_MODEL_TRANSITION_L1",
        "PLANNING_VI_EQUALS_BRUTE_FORCE",
        "SAFETY_CONSTRAINT_POLICY_SELECTED",
    ]:
        assert name in out
