"""Backtesting margin requirements against realised losses."""
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from src.margin import compute_margin


def realised_forward_loss(portfolio: pd.Series, returns: pd.DataFrame, h: int) -> pd.Series:
    """Return the loss at date t from holding the book unchanged over t+1..t+h:

        L_t = -sum_i N_i * (exp(sum_{k=1..h} r_{i,t+k}) - 1)

    The last h dates are NaN.
    """
    r_h = returns[portfolio.index].rolling(h).sum().shift(-h)
    # skipna=False keeps the last h dates NaN, otherwise they become a loss of exactly 0
    return -r_h.apply(np.exp).sub(1).mul(portfolio).sum(axis=1, skipna=False)


def run_backtest(portfolio: pd.Series, returns: pd.DataFrame, z_hist: pd.DataFrame,
                 sigma_daily: pd.DataFrame, adv: pd.DataFrame, cfg,
                 start: str, end: str | None = None, step: int = 1) -> pd.DataFrame:
    """Run compute_margin on each date in [start, end], every `step` days.

    Margin at t uses only data indexed <= t. Returns a DataFrame with columns dyn_margin, var,
    flat_margin, realised_loss, breach_dyn, breach_var and breach_flat, where a breach means
    realised_loss exceeded that level.

    The last h dates have no realised loss yet, so their breach flags are False; summarise
    leaves those rows out.
    """
    loss = realised_forward_loss(portfolio, returns, cfg.horizon)
    dates = returns.loc[start:end].index[::step]

    rows = []
    for t in dates:
        res = compute_margin(portfolio, z_hist.loc[:t], sigma_daily.loc[t], adv.loc[t], cfg)
        rows.append({"dyn_margin": res.total, "var": res.var})

    bt = pd.DataFrame(rows, index=dates)
    bt["flat_margin"] = cfg.flat_margin * portfolio.abs().sum()
    bt["realised_loss"] = loss.reindex(dates)
    bt["breach_dyn"] = bt["realised_loss"] > bt["dyn_margin"]
    bt["breach_var"] = bt["realised_loss"] > bt["var"]
    bt["breach_flat"] = bt["realised_loss"] > bt["flat_margin"]
    return bt


def kupiec_pof(n: int, x: int, p: float) -> tuple[float, float]:
    """Return (LR, p_value) for the Kupiec proportion-of-failures test.

    Tests whether x breaches in n observations fits an expected breach rate p:

        LR = -2 * [ (n-x) ln(1-p) + x ln p  -  (n-x) ln(1-x/n) - x ln(x/n) ]

    with 0 * ln 0 = 0, and p_value = 1 - chi2.cdf(LR, df=1).
    """
    log_expected = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_observed = xlogy(n - x, 1 - x / n) + xlogy(x, x / n)
    lr = max(-2 * (log_expected - log_observed), 0.0)
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def summarise(bt: pd.DataFrame, gross: float, p: float, step_for_test: int) -> pd.DataFrame:
    """Return a table comparing flat and dynamic margin.

    Columns are average and maximum margin (% of gross), number and rate of breaches, worst
    shortfall (max of loss - margin, negative if the margin always held) and, for dynamic
    margin, the Kupiec p-value. Kupiec uses VaR breaches on every `step_for_test`-th row so
    the windows don't overlap. p is the expected breach rate, e.g. 0.01 for 99% VaR.
    """
    done = bt.dropna(subset=["realised_loss"])

    rows = {}
    for name, level, breach in [("Flat", "flat_margin", "breach_flat"),
                                ("Dynamic", "dyn_margin", "breach_dyn")]:
        rows[name] = {
            "avg_margin_pct": 100 * done[level].mean() / gross,
            "max_margin_pct": 100 * done[level].max() / gross,
            "breaches": int(done[breach].sum()),
            "breach_rate": float(done[breach].mean()),
            "worst_shortfall": float((done["realised_loss"] - done[level]).max()),
        }

    tested = done.iloc[::step_for_test]
    _, rows["Dynamic"]["kupiec_p"] = kupiec_pof(len(tested), int(tested["breach_var"].sum()), p)
    return pd.DataFrame.from_dict(rows, orient="index")
