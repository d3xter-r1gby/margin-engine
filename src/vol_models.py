"""Volatility forecasting models: EWMA, GARCH(1,1) and LightGBM.

Alignment convention: a forecast stored at date t uses information up to and including t,
and is the forecast for day t+1 (or for days t+1..t+h).

Each model ends by producing `var_h`, a forecast of the h-day variance
sum_{k=1..h} sigma^2_{t+k}, so all models can be compared on the same target.
"""
import numpy as np
import pandas as pd

from src.data import forward_realised_var


# ---------- EWMA (RiskMetrics) ----------
def ewma_variance(r: pd.Series, lam: float = 0.94, init_window: int = 30) -> pd.Series:
    """Return the EWMA variance forecast, stored at index t as the forecast for t+1:

        sigma^2_{t+1|t} = lam * sigma^2_{t|t-1} + (1 - lam) * r_t^2

    The recursion starts from the sample variance of the first `init_window` returns, so the
    earliest rows use later data and aren't strictly out of sample.
    """
    prev = r.iloc[:init_window].var()
    out = pd.Series(index=r.index, dtype=float)
    
    for i, x in enumerate(r.values):
        prev = lam * prev + (1-lam) * x**2
        out.iloc[i] = prev
        
    return out       


def ewma_horizon_var(sigma2_next: pd.Series | pd.DataFrame, h: int):
    """Return the h-day variance forecast. EWMA is flat, so this is h * sigma^2_{t+1|t}."""
    return h * sigma2_next


# ---------- GARCH(1,1) ----------
def fit_garch(r_train: pd.Series) -> dict:
    """Fit a zero-mean GARCH(1,1) with Student-t errors by maximum likelihood.

    Returns {"omega", "alpha", "beta"} in decimal units. Returns are scaled by 100 for the
    optimiser, so omega is divided by 1e4. Pass training data only.
    """
    from arch import arch_model
    
    res = arch_model(100*r_train, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    omega = res.params["omega"]/1e4
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    
    return {"omega": omega, "alpha": alpha, "beta": beta}


def garch_variance(r: pd.Series, params: dict, init_var: float | None = None) -> pd.Series:
    """Run the GARCH recursion with fixed `params`, stored at index t as the forecast for t+1:

        sigma^2_{t+1|t} = omega + alpha * r_t^2 + beta * sigma^2_{t|t-1}

    Starts from `init_var`, or from omega / (1 - alpha - beta) if it is None. Pass `init_var`
    when alpha + beta is close to 1, because that long-run variance is then unusable.
    """
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    
    if init_var is not None:
        prev = init_var
    else:
        prev = omega / (1 - alpha - beta)
    out = pd.Series(index=r.index, dtype=float)
        
    for i, x in enumerate(r.values):
        prev = omega + alpha * x**2 + beta * prev
        out.iloc[i] = prev
        
    return out       

def garch_horizon_var(sigma2_next: pd.Series, params: dict, h: int) -> pd.Series:
    """Return the expected h-day variance, mean-reverting to V_L = omega / (1 - alpha - beta):

        var_h = h * V_L + (sigma^2_{t+1} - V_L) * (1 - phi^h) / (1 - phi),  phi = alpha + beta

    When phi > 0.999, V_L is unusable, so this returns the flat forecast h * sigma^2_{t+1}.
    """
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    
    if alpha + beta > 0.999:
        return h * sigma2_next 
    
    v_l = omega / (1 - alpha - beta)
    phi = alpha + beta
    
    var_h = h * v_l + (sigma2_next - v_l) * (1 - phi**h) / (1 - phi)
    
    return var_h


# ---------- Gradient boosting (LightGBM) ----------
def build_features(returns: pd.DataFrame, close: pd.DataFrame, market: str,
                   ewma_var: pd.DataFrame) -> pd.DataFrame:
    """Return the LightGBM feature table, long format with a (date, ticker) MultiIndex.

    Every feature at t uses data up to and including t. Rows are NaN until the rolling
    windows fill. Columns:
      log_rv1   = log(r_t^2 + eps)
      log_rv5   = log(mean r^2 over the last 5 days)
      log_rv22  = log(mean r^2 over the last 22 days)
      log_ewma  = log(ewma_var)
      ret5      = sum of the last 5 returns (leverage effect)
      mkt_rv22  = log_rv22 of the market ticker, the same for every ticker on a date
      dd60      = close / rolling 60-day max - 1 (drawdown)
    """
    eps = 1e-12
    r2 = returns ** 2
    log_rv22 = np.log(r2.rolling(22).mean() + eps)
    features = {
        "log_rv1": np.log(r2 + eps),
        "log_rv5": np.log(r2.rolling(5).mean() + eps),
        "log_rv22": log_rv22,
        "log_ewma": np.log(ewma_var),
        "ret5": returns.rolling(5).sum(),
        "mkt_rv22": pd.DataFrame({t: log_rv22[market] for t in returns.columns}),
        "dd60": close / close.rolling(60).max() - 1,
    }
    X = pd.concat({name: f.stack() for name, f in features.items()}, axis=1)
    return X.rename_axis(["date", "ticker"])


def build_target(returns: pd.DataFrame, horizon: int) -> pd.Series:
    """Return the log of forward realised h-day variance, long format (date, ticker)."""
    y = np.log(forward_realised_var(returns, horizon) + 1e-12).stack()
    return y.rename_axis(["date", "ticker"])


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series,
               X_valid: pd.DataFrame, y_valid: pd.Series):
    """Fit an LGBMRegressor of up to 2000 trees and return it.

    Training stops once the loss on the validation set has not improved for 100 rounds.
    """
    import lightgbm as lgb

    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31,
                              min_child_samples=200, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def predict_lgbm_var_h(model, X: pd.DataFrame, bias_factor: float = 1.0) -> pd.Series:
    """Return h-day variance forecasts from a model trained on log variance.

    exp(prediction) estimates the median rather than the mean, so it is multiplied by
    `bias_factor`.
    """
    return pd.Series(np.exp(model.predict(X)) * bias_factor, index=X.index, name="var_h")


def to_wide(s: pd.Series) -> pd.DataFrame:
    """Convert a long (date, ticker) Series to a wide DataFrame."""
    return s.unstack()
