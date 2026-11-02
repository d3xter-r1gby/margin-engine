"""Download data, fit all models, run the backtests and save what the dashboard needs to
data/artifacts.pkl. Run with: python -m scripts.run_models
"""
import pickle

import numpy as np
import pandas as pd
from config import CFG
from config import CONCENTRATED_BOOK, DIVERSIFIED_BOOK
from src import data, vol_models, evaluate, margin, backtest


def main():
    close, volume = data.load_or_download(CFG.tickers, CFG.start)
    rets = data.log_returns(close)
    target = data.forward_realised_var(rets, CFG.horizon)
    adv = data.average_dollar_volume(close, volume)

    # EWMA
    ewma = rets.apply(vol_models.ewma_variance, lam=CFG.ewma_lambda)
    ewma_var_h = vol_models.ewma_horizon_var(ewma, CFG.horizon)

    # GARCH: fit on the training period, then run the recursion over all data
    train = rets.loc[:CFG.train_end]
    garch_var_h = {}
    for t in CFG.tickers:
        p = vol_models.fit_garch(train[t])
        phi = p["alpha"] + p["beta"]
        init_var = None
        if phi >= 0.999:
            print(f"warning: {t} alpha + beta = {phi:.4f}")
            init_var = (train[t] ** 2).mean()
        sigma2 = vol_models.garch_variance(rets[t], p, init_var=init_var)
        garch_var_h[t] = vol_models.garch_horizon_var(sigma2, p, CFG.horizon)
    garch_var_h = pd.DataFrame(garch_var_h)

    # LightGBM: one pooled model, split by date into train / valid / test
    h = CFG.horizon
    X = vol_models.build_features(rets, close, CFG.market, ewma).dropna()
    y = vol_models.build_target(rets, h).dropna().rename("target")
    Xy = X.join(y, how="inner")
    dates = Xy.index.get_level_values(0)
    # drop the last h days of each split: their targets use returns from the next split
    last_train = rets.index[rets.index <= CFG.train_end][-h - 1]
    last_valid = rets.index[rets.index <= CFG.valid_end][-h - 1]
    tr = Xy[dates <= last_train]
    va = Xy[(dates > CFG.train_end) & (dates <= last_valid)]
    features = list(X.columns)
    model = vol_models.train_lgbm(tr[features], tr["target"], va[features], va["target"])

    # Jensen correction: exp(prediction) estimates the median, not the mean
    bias_factor = np.exp(tr["target"]).mean() / np.exp(model.predict(tr[features])).mean()
    lgbm_var_h = vol_models.to_wide(vol_models.predict_lgbm_var_h(model, X, bias_factor))
    lgbm_var_h = lgbm_var_h.reindex(index=rets.index, columns=rets.columns)

    def valid_rmse(pred):
        return np.sqrt(((pred - va["target"]) ** 2).mean())

    print(f"bias factor: {bias_factor:.3f}, trees: {model.best_iteration_}")
    print(f"valid RMSE on log var: LightGBM {valid_rmse(model.predict(va[features])):.4f}, "
          f"naive EWMA {valid_rmse(va['log_ewma'] + np.log(h)):.4f}")
    print(pd.Series(model.feature_importances_, index=features).sort_values(ascending=False))

    # Score every model on the test period
    var_h = {"EWMA": ewma_var_h, "GARCH": garch_var_h, "LightGBM": lgbm_var_h}
    test_start = str(rets.index[rets.index > CFG.valid_end][0].date())
    metrics = evaluate.compare_models(var_h, target, test_start, returns=rets, market=CFG.market)
    print(metrics.round(4))

    # Standardised residuals from EWMA drive the FHS scenarios
    z = margin.standardised_residuals(rets, ewma)

    # Backtest each preset book with EWMA from 2016
    # TODO: add a 2022+ backtest comparison of EWMA vs GARCH vs LightGBM
    sigma_daily = margin.daily_sigma_from_var_h(ewma_var_h, h)
    backtests, bt_summary = {}, {}
    for name, book in {"Concentrated": CONCENTRATED_BOOK, "Diversified": DIVERSIFIED_BOOK}.items():
        port = pd.Series(book)
        backtests[name] = backtest.run_backtest(port, rets, z, sigma_daily, adv, CFG, "2016-01-01")
        bt_summary[name] = backtest.summarise(backtests[name], port.abs().sum(), 1 - CFG.conf, h)
        print(name, bt_summary[name].round(4), sep="\n")

    artifacts = {
        "close": close, "rets": rets, "adv": adv, "target": target,
        "var_h": var_h, "metrics": metrics, "z": z,
        "backtests": backtests, "bt_summary": bt_summary,
    }
    with open("data/artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print("saved data/artifacts.pkl")


if __name__ == "__main__":
    main()
