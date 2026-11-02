"""Tiny checks you can run with:  python -m pytest -q"""
import numpy as np
import pandas as pd
from src import margin, backtest


def test_hhi_equal_weights():
    p = pd.Series({"A": 1.0, "B": -1.0, "C": 1.0, "D": -1.0})
    assert abs(margin.hhi(p) - 0.25) < 1e-12


def test_var_es_ordering():
    pnl = np.random.default_rng(0).normal(0, 1, 100_000)
    var, es = margin.var_es(pnl, 0.99)
    assert 2.2 < var < 2.45 and es > var      # normal: VaR99 ~ 2.33, ES99 ~ 2.67


def test_kupiec_exact_rate_not_rejected():
    lr, p = backtest.kupiec_pof(n=1000, x=10, p=0.01)
    assert lr < 1e-9 and p > 0.99


def test_fhs_shape():
    z = np.ones((20, 3))
    R = margin.fhs_scenarios(z, np.array([0.01, 0.02, 0.03]), h=10)
    assert R.shape == (11, 3) and np.allclose(R[0], [0.1, 0.2, 0.3])
