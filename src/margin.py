"""Margin engine: filtered historical simulation plus liquidity, concentration and floor add-ons.

A portfolio is a pd.Series of dollar notionals indexed by ticker (+long, -short).
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class MarginResult:
    gross: float
    var: float                 # 99% h-day VaR (positive = loss), after liquidity scaling
    es_base: float             # ES with no add-ons
    liquidity_addon: float     # ES_liq - es_base
    concentration_addon: float # ES_liq * (multiplier - 1)
    floor_addon: float         # extra needed to reach the floor, else 0
    total: float               # final margin requirement in dollars
    hhi: float
    liq_days: pd.Series        # days to liquidate per position
    pnl_scenarios: np.ndarray  # scenario P&L (after liquidity scaling), for plotting


def standardised_residuals(returns: pd.DataFrame, sigma2_next: pd.DataFrame) -> pd.DataFrame:
    """Return z_t = r_t / sigma_{t|t-1}.

    sigma2_next[t] is the forecast for t+1, so it is shifted one day before dividing.
    """
    return returns / sigma2_next.shift(1).apply(np.sqrt)


def fhs_scenarios(z_window: np.ndarray, sigma_today: np.ndarray, h: int) -> np.ndarray:
    """Return h-day scenario log returns, shape (N-h+1, n).

    z_window is (N days x n assets) of standardised residuals, oldest first, and sigma_today
    is the (n,) daily vol forecast. Each row sums h consecutive days of z and scales it by
    sigma_today. A row uses the same dates for every asset, which preserves correlation.
    """
    c = np.cumsum(z_window, axis=0)
    Z = c[h - 1:] - np.vstack([np.zeros((1, z_window.shape[1])), c[:-h]])
    return Z * sigma_today


def liquidity_days(portfolio: pd.Series, adv_today: pd.Series, participation: float) -> pd.Series:
    """Return the days needed to exit each position: |notional_i| / (participation * ADV_i)."""
    return portfolio.abs() / (participation * adv_today)


def liquidity_scale(days: pd.Series, h: int) -> pd.Series:
    """Return each position's scenario scale: sqrt((h + days_i / 2) / h).

    After the h-day close-out period the position is sold evenly over days_i days, so on
    average it is exposed for an extra days_i / 2 days. Vol grows with sqrt(time).
    """
    return ((h + days / 2) / h) ** 0.5


def portfolio_pnl(portfolio: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Return P&L per scenario: sum_i notional_i * (exp(R_i) - 1)."""
    return (np.exp(R) - 1) @ portfolio


def var_es(pnl: np.ndarray, conf: float) -> tuple[float, float]:
    """Return (VaR, ES) as positive loss numbers.

    VaR is the `conf` quantile of losses, and ES is the mean of the losses at or above VaR.
    """
    losses = -pnl
    var = float(np.quantile(losses, conf))
    return var, float(losses[losses >= var].mean())


def hhi(portfolio: pd.Series) -> float:
    """Return the Herfindahl index of gross weights, sum (|N_i| / gross)^2: 1/n for n equal positions."""
    weights = portfolio.abs() / portfolio.abs().sum()
    return float((weights ** 2).sum())


def concentration_multiplier(h_index: float, k: float, threshold: float) -> float:
    """Return the concentration multiplier: 1 + k * max(0, HHI - threshold)."""
    return 1 + k * max(0.0, h_index - threshold)


def compute_margin(portfolio: pd.Series, z_hist: pd.DataFrame, sigma_today: pd.Series,
                   adv_today: pd.Series, cfg) -> MarginResult:
    """Compute the margin requirement for one date.

    Builds FHS scenarios from the last cfg.fhs_window complete rows of z_hist and takes the
    base ES. It then rescales each position for its liquidation time to get ES_liq, and sets
    total = max(ES_liq * concentration multiplier, cfg.margin_floor * gross). Inputs are
    aligned to portfolio.index, and each add-on is returned separately for the waterfall.
    """
    tickers = portfolio.index
    notionals = portfolio.to_numpy()
    z_window = z_hist[tickers].dropna().iloc[-cfg.fhs_window:].to_numpy()

    R = fhs_scenarios(z_window, sigma_today[tickers].to_numpy(), cfg.horizon)
    _, es_base = var_es(portfolio_pnl(notionals, R), cfg.conf)

    days = liquidity_days(portfolio, adv_today[tickers], cfg.participation)
    scale = liquidity_scale(days, cfg.horizon).to_numpy()
    pnl_liq = portfolio_pnl(notionals, R * scale)
    var, es_liq = var_es(pnl_liq, cfg.conf)

    h_index = hhi(portfolio)
    mult = concentration_multiplier(h_index, cfg.conc_k, cfg.hhi_threshold)
    gross = float(portfolio.abs().sum())
    before_floor = es_liq * mult
    total = max(before_floor, cfg.margin_floor * gross)

    return MarginResult(gross=gross, var=var, es_base=es_base,
                        liquidity_addon=es_liq - es_base,
                        concentration_addon=es_liq * (mult - 1),
                        floor_addon=total - before_floor, total=total, hhi=h_index,
                        liq_days=days, pnl_scenarios=pnl_liq)


def daily_sigma_from_var_h(var_h: pd.DataFrame, h: int) -> pd.DataFrame:
    """Return the daily vol equivalent to an h-day variance forecast: sqrt(var_h / h)."""
    return (var_h / h).apply(np.sqrt)
