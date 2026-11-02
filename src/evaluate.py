"""Forecast evaluation."""
import numpy as np
import pandas as pd

from src.data import realised_vol


def _valid_cells(forecast: pd.DataFrame, realised: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Align on dates and tickers, stack to 1-D, keep the cells where both values exist."""
    f, r = forecast.align(realised, join="inner")
    both = pd.concat([f.stack(), r.stack()], axis=1, keys=["f", "r"]).dropna()
    return both["f"].to_numpy(), both["r"].to_numpy()


def rmse(forecast: pd.DataFrame, realised: pd.DataFrame) -> float:
    """Return the root mean squared error over all (date, ticker) cells where both exist."""
    f, r = _valid_cells(forecast, realised)
    return float(np.sqrt(np.mean((f - r) ** 2)))


def qlike(forecast: pd.DataFrame, realised: pd.DataFrame, eps: float = 1e-12) -> float:
    """Return the mean of RV/F - ln(RV/F) - 1 over all valid cells. 0 is perfect; lower is better."""
    f, rv = _valid_cells(forecast, realised)
    x = (rv + eps) / (f + eps)
    return float(np.mean(x - np.log(x) - 1))


def compare_models(forecasts: dict[str, pd.DataFrame], realised: pd.DataFrame,
                   start: str, end: str | None = None, returns: pd.DataFrame | None = None,
                   market: str = "SPY") -> pd.DataFrame:
    """Return one row per model, scored on the window [start, end].

    Columns are RMSE_vol (RMSE on h-day vol, the sqrt of variance) and QLIKE (on variance). If
    `returns` is given, QLIKE_highvol scores only the dates where the market's 22-day realised
    vol is above its 80th percentile in the window.
    """
    rv = realised.loc[start:end]
    high_vol = None
    if returns is not None:
        market_vol = realised_vol(returns, 22)[market].loc[start:end]
        high_vol = market_vol.index[market_vol > market_vol.quantile(0.8)]

    rows = {}
    for name, forecast in forecasts.items():
        f = forecast.loc[start:end]
        row = {"RMSE_vol": rmse(f.apply(np.sqrt), rv.apply(np.sqrt)), "QLIKE": qlike(f, rv)}
        if high_vol is not None:
            row["QLIKE_highvol"] = qlike(f[f.index.isin(high_vol)], rv)
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")
