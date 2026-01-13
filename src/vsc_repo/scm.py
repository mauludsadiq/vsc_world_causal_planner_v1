from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

@dataclass(frozen=True)
class SCMParams:
    pz: float = 0.5
    ax: float = 2.0
    bx: float = -1.0
    ayx: float = 2.5
    ayz: float = 2.0
    by: float = -2.0

class BinarySCM:
    """Binary SCM with confounding via Z."""

    def __init__(self, params: SCMParams = SCMParams()) -> None:
        self.p = params

    def sample_observational(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        Z = rng.binomial(1, self.p.pz, size=n)
        px = _sigmoid(self.p.ax * Z + self.p.bx)
        X = rng.binomial(1, px, size=n)
        py = _sigmoid(self.p.ayx * X + self.p.ayz * Z + self.p.by)
        Y = rng.binomial(1, py, size=n)
        return {"Z": Z.astype(int), "X": X.astype(int), "Y": Y.astype(int)}

    def sample_interventional(self, n: int, do_X: int, seed: int = 0) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        Z = rng.binomial(1, self.p.pz, size=n)
        X = np.full(shape=n, fill_value=int(do_X), dtype=int)
        py = _sigmoid(self.p.ayx * X + self.p.ayz * Z + self.p.by)
        Y = rng.binomial(1, py, size=n)
        return {"Z": Z.astype(int), "X": X, "Y": Y.astype(int)}

    def true_p_y1_do_x(self, do_X: int, n_mc: int = 200000, seed: int = 0) -> float:
        samp = self.sample_interventional(n=n_mc, do_X=do_X, seed=seed)
        return float(np.mean(samp["Y"]))

def p_y1_given_x(data: Dict[str, np.ndarray], x: int) -> float:
    X, Y = data["X"], data["Y"]
    m = (X == int(x))
    if not np.any(m):
        raise ValueError("No samples for X=x")
    return float(np.mean(Y[m]))

def backdoor_adjustment_p_y1_do_x(data: Dict[str, np.ndarray], x: int) -> float:
    Z, X, Y = data["Z"], data["X"], data["Y"]
    pz0 = float(np.mean(Z == 0))
    pz1 = 1.0 - pz0

    def p_y1_given_xz(xv: int, zv: int) -> float:
        m = (X == int(xv)) & (Z == int(zv))
        if not np.any(m):
            # smoothing if the slice is empty (rare with large n)
            return 0.5
        return float(np.mean(Y[m]))

    return p_y1_given_xz(x, 0) * pz0 + p_y1_given_xz(x, 1) * pz1
