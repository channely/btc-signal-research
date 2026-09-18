import argparse
import hashlib
import io
import json
import platform
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAY_MS = 86_400_000
STRATEGIES = ["trend_model", "buy_hold", "sma200", "vol_hold"]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_data(data):
    if data.empty or not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Empty, duplicate or unsorted candle dates")
    expected = pd.date_range(data.index.min(), data.index.max(), freq="D")
    if not data.index.equals(expected):
        raise ValueError("Missing daily candles; do not silently fill gaps")
    prices = data[["open", "high", "low", "close"]]
    if not np.isfinite(data.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Invalid numeric candle data")
    if (data["high"] < prices.max(axis=1)).any() or (data["low"] > prices.min(axis=1)).any():
        raise ValueError("Inconsistent OHLC ranges")
    if (data["volume"] < 0).any():
        raise ValueError("Negative volume")
    if data.index.max() >= pd.Timestamp.now(tz="UTC").tz_localize(None).normalize():
        raise ValueError("Incomplete UTC daily candle")


def load_data(config):
    path = ROOT / "btc_daily.csv"
    manifest_path = ROOT / "data_manifest.json"
    if path.exists():
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("Cached data hash mismatch")
        protocol_hash = hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
        if protocol_hash != manifest["protocol_sha256_before_download"]:
            raise ValueError("Protocol changed after the data snapshot; use a separate research version")
        data = pd.read_csv(path, index_col="date", parse_dates=["date"])
        validate_data(data)
        return data, manifest
    start = int(pd.Timestamp(config["download_start"], tz="UTC").timestamp() * 1000)
    stop = int(pd.Timestamp.now(tz="UTC").normalize().timestamp() * 1000)
    rows, urls = [], []
    while start < stop:
        params = urllib.parse.urlencode({
            "symbol": config["symbol"], "interval": "1d", "startTime": start,
            "endTime": stop - 1, "limit": 1000,
        })
        url = config["source"] + "?" + params
        request = urllib.request.Request(url, headers={"User-Agent": "BTC-TVM-Research/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.load(response)
        if not isinstance(batch, list) or not batch or not all(len(row) == 12 for row in batch):
            raise ValueError("Unexpected Binance candle response")
        if int(batch[-1][0]) < start:
            raise ValueError("Pagination did not advance")
        rows.extend(batch)
        urls.append(url)
        start = int(batch[-1][0]) + DAY_MS
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_base", "taker_quote", "unused"]
    raw = pd.DataFrame(rows, columns=columns)
    raw = raw.loc[raw["close_time"].astype("int64") < stop]
    data = raw[["open", "high", "low", "close", "volume"]].astype(float)
    data.index = pd.DatetimeIndex(pd.to_datetime(raw["open_time"], unit="ms"), name="date")
    validate_data(data)
    data.to_csv(path, float_format="%.10f")
    manifest = {
        "source": config["source"], "symbol": config["symbol"], "interval": "1d",
        "timezone": "UTC", "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_closed_candle": str(data.index.min().date()),
        "last_closed_candle": str(data.index.max().date()), "rows": len(data),
        "requests": urls, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol_sha256_before_download": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
    }
    save_json(manifest_path, manifest)
    return data, manifest


def indicators(data, config, sma=None, momentum=None):
    close = data["close"]
    moving_average = close.rolling(sma or config["sma_days"]).mean()
    momentum_return = close / close.shift(momentum or config["momentum_days"]) - 1
    volatility = np.log(close).diff().rolling(config["volatility_days"]).std(ddof=1)
    volatility *= np.sqrt(config["annualization_days"])
    score = (close > moving_average).astype(float) + (momentum_return > 0).astype(float)
    valid = moving_average.notna() & momentum_return.notna() & volatility.notna()
    score = score.where(valid)
    vol_weight = (config["target_annual_volatility"] / volatility).clip(0, config["max_position"])
    return pd.DataFrame({
        "close": close, "sma": moving_average, "momentum": momentum_return,
        "volatility": volatility, "score": score,
        "trend_model": vol_weight.where(score == 2, 0).where(valid),
        "buy_hold": pd.Series(1.0, index=data.index).where(valid),
        "sma200": (close > moving_average).astype(float).where(valid),
        "vol_hold": vol_weight.where(valid),
    })


def simulate(market_returns, desired_weights, cost, band):
    if not market_returns.index.equals(desired_weights.index):
        raise ValueError("Returns and positions have different timestamps")
    if desired_weights.isna().any() or not desired_weights.between(0, 1).all():
        raise ValueError("Invalid target weights")
    previous_weight = 0.0
    records = []
    for daily_return, target in zip(market_returns.to_numpy(), desired_weights.to_numpy()):
        trade = (abs(target - previous_weight) >= band or
                 (target == 0 and previous_weight != 0) or
                 (target > 0 and previous_weight == 0))
        if trade:
            difference = target - previous_weight
            denominator = 1 + cost * target if difference >= 0 else 1 - cost * target
            turnover = abs(difference / denominator)
            weight = target
        else:
            turnover, weight = 0.0, previous_weight
        fee_fraction = cost * turnover
        portfolio_return = (1 - fee_fraction) * (1 + weight * daily_return) - 1
        previous_weight = weight * (1 + daily_return) / (1 + weight * daily_return)
        records.append((portfolio_return, weight, turnover, fee_fraction))
    result = pd.DataFrame(records, index=market_returns.index,
                          columns=["return", "weight", "turnover", "cost_fraction"])
    result["equity"] = (1 + result["return"]).cumprod()
    return result


def backtest(data, features, config, start, stop=None, cost_bps=None, delay=None):
    lag = config["signal_to_execution_open_days"] if delay is None else delay
    cost_bps = config["one_way_fee_bps"] + config["one_way_slippage_bps"] if cost_bps is None else cost_bps
    returns = data["open"].shift(-1) / data["open"] - 1
    desired = features[STRATEGIES].shift(lag)
    eligible = returns.notna() & desired.notna().all(axis=1) & (data.index >= pd.Timestamp(start))
    if stop:
        eligible &= data.index < pd.Timestamp(stop)
    return {name: simulate(returns.loc[eligible], desired.loc[eligible, name], cost_bps / 10000,
                           config["rebalance_band"]) for name in STRATEGIES}


def statistics(result):
    returns = result["return"].to_numpy()
    equity = np.r_[1.0, np.cumprod(1 + returns)]
    drawdown = equity / np.maximum.accumulate(equity) - 1
    volatility = returns.std(ddof=1) * np.sqrt(365)
    cagr = equity[-1] ** (365 / len(returns)) - 1
    max_dd = drawdown.min()
    underwater, longest = 0, 0
    for value in drawdown:
        underwater = underwater + 1 if value < -1e-12 else 0
        longest = max(longest, underwater)
    return {
        "days": len(returns), "start_open": str(result.index.min().date()),
        "end_valuation_open": str((result.index.max() + pd.Timedelta(days=1)).date()),
        "total_return": float(equity[-1] - 1), "cagr": float(cagr),
        "annual_volatility": float(volatility),
        "sharpe": float(returns.mean() * 365 / volatility) if volatility > 0 else 0.0,
        "max_drawdown": float(max_dd), "calmar": float(cagr / abs(max_dd)) if max_dd else 0.0,
        "worst_day": float(returns.min()), "longest_underwater_days": longest,
        "mean_position": float(result["weight"].mean()),
        "exposed_days_fraction": float((result["weight"] > 0).mean()),
        "trading_days": int((result["turnover"] > 1e-12).sum()),
        "one_way_turnover": float(result["turnover"].sum()),
        "sum_cost_fractions": float(result["cost_fraction"].sum()),
    }


def block_indices(length, block, repetitions, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, length, size=(repetitions, int(np.ceil(length / block))))
    indices = (starts[:, :, None] + np.arange(block)) % length
    return indices.reshape(repetitions, -1)[:, :length]


def interval(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def sharpe_uncertainty(results, config):
    records = []
    a = results["trend_model"]["return"].to_numpy()
    for comparator in ["buy_hold", "vol_hold", "sma200"]:
        b = results[comparator]["return"].to_numpy()
        for block in config["bootstrap_block_days"]:
            indices = block_indices(len(a), block, config["bootstrap_repetitions"], config["bootstrap_seed"])
            aa, bb = a[indices], b[indices]
            sa = aa.mean(axis=1) / aa.std(axis=1, ddof=1) * np.sqrt(365)
            sb = bb.mean(axis=1) / bb.std(axis=1, ddof=1) * np.sqrt(365)
            records.append({"comparator": comparator, "block_days": block,
                            "difference": statistics(results["trend_model"])["sharpe"] - statistics(results[comparator])["sharpe"],
                            "ci95": interval(sa - sb)})
    return records


def forecast_analysis(data, features, config, calibration_end, evaluation_start, evaluation_end=None):
    output, tables = {}, []
    lag = config["signal_to_execution_open_days"]
    calibration_end = pd.Timestamp(calibration_end)
    for horizon in config["forecast_horizons_days"]:
        forward = data["open"].shift(-lag - horizon) / data["open"].shift(-lag) - 1
        label_end = data.index + pd.Timedelta(days=lag + horizon)
        valid = forward.notna() & features["score"].notna()
        train_mask = valid & (data.index >= config["research_start"]) & (label_end < calibration_end)
        test_mask = valid & (data.index >= evaluation_start)
        if evaluation_end:
            test_mask &= label_end < pd.Timestamp(evaluation_end)
        train = pd.DataFrame({"score": features.loc[train_mask, "score"], "forward_return": forward.loc[train_mask]})
        train["up"] = (train["forward_return"] > 0).astype(int)
        prior = float((train["up"].sum() + 1) / (len(train) + 2))
        probabilities, calibration = {}, []
        for score in [0, 1, 2]:
            group = train.loc[train["score"] == score]
            probabilities[score] = float((group["up"].sum() + 1) / (len(group) + 2))
            calibration.append({
                "score": score, "n_overlapping": len(group), "p_up": probabilities[score],
                "return_q10": float(group["forward_return"].quantile(0.1)),
                "return_median": float(group["forward_return"].median()),
                "return_q90": float(group["forward_return"].quantile(0.9)),
            })
        test = pd.DataFrame({"score": features.loc[test_mask, "score"], "forward_return": forward.loc[test_mask]})
        test["up"] = (test["forward_return"] > 0).astype(int)
        test["probability"] = test["score"].map(probabilities)
        test["baseline_probability"] = prior
        test["horizon"] = horizon
        prediction, target = test["probability"].to_numpy(), test["up"].to_numpy()
        loss = (prediction - target) ** 2
        base_loss = (prior - target) ** 2
        improvements = base_loss - loss
        ci = []
        for block in config["bootstrap_block_days"]:
            indices = block_indices(len(test), block, config["bootstrap_repetitions"], config["bootstrap_seed"])
            ci.append({"block_days": block, "ci95": interval(improvements[indices].mean(axis=1))})
        ranks = pd.Series(prediction).rank().to_numpy()
        positive = int(target.sum())
        negative = len(target) - positive
        auc = (ranks[target == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
        regimes = [{"score": int(score), "n_overlapping": len(group),
                    "observed_p_up": float(group["up"].mean()),
                    "mean_forward_return": float(group["forward_return"].mean())}
                   for score, group in test.groupby("score")]
        output[str(horizon)] = {
            "n_calibration_overlapping": len(train), "n_evaluation_overlapping": len(test),
            "latest_calibration_label_end": str(label_end[train_mask].max().date()),
            "calibration": calibration, "baseline_probability": prior,
            "brier": float(loss.mean()), "baseline_brier": float(base_loss.mean()),
            "brier_improvement": float(improvements.mean()), "brier_improvement_ci95": ci,
            "direction_accuracy": float(((prediction >= 0.5) == target).mean()),
            "baseline_direction_accuracy": float(((prior >= 0.5) == target).mean()),
            "auc": float(auc), "observed_regimes": regimes,
        }
        tables.append(test)
    return output, pd.concat(tables).rename_axis("signal_date")


def run_checks(data, features, config):
    checks = {}
    cutoff = pd.Timestamp("2023-06-30")
    prefix = indicators(data.loc[:cutoff], config)
    pd.testing.assert_frame_equal(prefix, features.loc[:cutoff])
    checks["indicator_prefix_invariance"] = True
    changed = data.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close"]] *= 3
    pd.testing.assert_frame_equal(indicators(changed, config).loc[:cutoff], features.loc[:cutoff])
    checks["future_price_mutation_does_not_change_past_signals"] = True
    original = backtest(data, features, config, config["research_start"])["trend_model"]
    mutated = backtest(changed, indicators(changed, config), config, config["research_start"])["trend_model"]
    pd.testing.assert_frame_equal(original.loc[:cutoff - pd.Timedelta(days=1)],
                                  mutated.loc[:cutoff - pd.Timedelta(days=1)])
    checks["future_price_mutation_does_not_change_past_pnl"] = True
    dates = pd.date_range("2020-01-01", periods=3)
    simple_return = pd.Series([0.1, -0.05, 0.02], index=dates)
    full = simulate(simple_return, pd.Series(1.0, index=dates), 0, 0.05)
    np.testing.assert_allclose(full["equity"], (1 + simple_return).cumprod())
    checks["buy_hold_matches_price_ratio_without_costs"] = True
    cash = simulate(simple_return, pd.Series(0.0, index=dates), 0.005, 0.05)
    np.testing.assert_allclose(cash["equity"], 1)
    checks["cash_has_zero_return_and_fees"] = True
    roundtrip = simulate(pd.Series(0.0, index=dates), pd.Series([1.0, 0.0, 0.0], index=dates), 0.0015, 0.05)
    np.testing.assert_allclose(roundtrip["equity"].iloc[-1], (1 - 0.0015) / (1 + 0.0015))
    checks["entry_exit_cost_accounting"] = True
    lagged = features["trend_model"].shift(config["signal_to_execution_open_days"])
    np.testing.assert_allclose(lagged.iloc[2:], features["trend_model"].iloc[:-2], equal_nan=True)
    checks["two_day_signal_execution_lag"] = True
    for name, result in backtest(data, features, config, config["holdout_start"]).items():
        assert result["weight"].between(-1e-12, 1 + 1e-12).all(), name
        assert np.isfinite(result.to_numpy()).all(), name
    checks["finite_unleveraged_portfolios"] = True
    assert config["research_start"] >= str(data.index[config["sma_days"] + 2].date())
    checks["sufficient_warmup_before_evaluation"] = True
    return checks


def save_plot(results, forecasts):
    colors = ["#0072B2", "#666666", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for (name, result), color in zip(results.items(), colors):
        dates = result.index + pd.Timedelta(days=1)
        initial = result.index[0]
        equity = pd.Series(np.r_[1, result["equity"].to_numpy()], index=pd.DatetimeIndex([initial]).append(dates))
        axes[0].plot(equity.index, equity, label=name, color=color, linewidth=1.4)
        axes[1].plot(equity.index, (equity / equity.cummax() - 1) * 100, label=name, color=color, linewidth=1)
    axes[0].set(title="BTC-TVM-1 | Retrospective holdout | Net of 15 bps one-way costs",
                ylabel="Equity (initial = 1, log scale)", yscale="log", xlabel="UTC valuation date")
    axes[0].legend(ncol=2)
    axes[1].set(ylabel="Drawdown (%)", xlabel="UTC valuation date")
    result = results["trend_model"]
    axes[2].plot(result.index, result["weight"] * 100, color=colors[0])
    axes[2].set(ylabel="BTC weight (%)", xlabel="UTC execution date", ylim=(-3, 103),
                title="Exposure is risk control, not proof of predictive alpha")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(ROOT / "holdout.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "protocol.json").read_text())
    data, manifest = load_data(config)
    features = indicators(data, config)
    checks = run_checks(data, features, config)
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    periods = {
        "calibration": (config["research_start"], config["initial_calibration_end_exclusive"]),
        "validation": (config["initial_calibration_end_exclusive"], config["validation_end_exclusive"]),
        "holdout": (config["holdout_start"], None),
        "full_descriptive_only": (config["research_start"], None),
    }
    metrics, period_results, rows = {}, {}, []
    for period, (start, stop) in periods.items():
        results = backtest(data, features, config, start, stop)
        period_results[period] = results
        metrics[period] = {name: statistics(result) for name, result in results.items()}
        rows.extend({"period": period, "strategy": name, **value} for name, value in metrics[period].items())
    pd.DataFrame(rows).to_csv(ROOT / "metrics.csv", index=False)
    yearly = []
    for year in range(2019, data.index.max().year + 1):
        results = backtest(data, features, config, f"{year}-01-01", f"{year + 1}-01-01")
        yearly.extend({"year": year, "strategy": name, **statistics(result)} for name, result in results.items())
    pd.DataFrame(yearly).to_csv(ROOT / "yearly_fresh_start.csv", index=False)
    holdout = period_results["holdout"]
    pd.concat(holdout, names=["strategy", "execution_date"]).to_csv(ROOT / "holdout_daily.csv")
    uncertainty = sharpe_uncertainty(holdout, config)
    validation_forecast, _ = forecast_analysis(
        data, features, config, config["initial_calibration_end_exclusive"],
        config["initial_calibration_end_exclusive"], config["validation_end_exclusive"])
    forecast, predictions = forecast_analysis(
        data, features, config, config["validation_end_exclusive"], config["holdout_start"])
    predictions.to_csv(ROOT / "holdout_predictions.csv")
    neighbors = []
    for sma in config["sensitivity_sma_days"]:
        for momentum in config["sensitivity_momentum_days"]:
            variant = indicators(data, config, sma, momentum)
            result = backtest(data, variant, config, config["holdout_start"])["trend_model"]
            neighbors.append({"sma": sma, "momentum": momentum, **statistics(result)})
    pd.DataFrame(neighbors).to_csv(ROOT / "sensitivity.csv", index=False)
    stress = []
    for cost in config["stress_total_one_way_cost_bps"]:
        result = backtest(data, features, config, config["holdout_start"], cost_bps=cost)["trend_model"]
        stress.append({"case": f"cost_{cost}bps", **statistics(result)})
    for delay in config["stress_execution_delay_days"]:
        result = backtest(data, features, config, config["holdout_start"], delay=delay)["trend_model"]
        stress.append({"case": f"delay_{delay}_days", **statistics(result)})
    pd.DataFrame(stress).to_csv(ROOT / "stress.csv", index=False)
    rolling = []
    full = period_results["full_descriptive_only"]
    for name, result in full.items():
        net = (1 + result["return"]).rolling(365 * 3).apply(np.prod, raw=True) - 1
        net = net.dropna()
        rolling.append({"strategy": name, "windows_overlapping": len(net),
                        "minimum_3year_total_return": float(net.min()),
                        "positive_window_fraction": float((net > 0).mean())})
    current = features.iloc[-1]
    model = metrics["holdout"]["trend_model"]
    acceptance = {
        "holdout_net_cagr_positive": model["cagr"] > 0,
        "holdout_max_drawdown_no_worse_than": model["max_drawdown"] >= config["acceptance"]["holdout_max_drawdown_no_worse_than"],
        "holdout_sharpe_above_buy_hold": model["sharpe"] > metrics["holdout"]["buy_hold"]["sharpe"],
        "holdout_sharpe_above_vol_hold": model["sharpe"] > metrics["holdout"]["vol_hold"]["sharpe"],
        "positive_sharpe_difference_lower_95pct_bound_vs_vol_hold_all_blocks": all(
            item["ci95"][0] > 0 for item in uncertainty if item["comparator"] == "vol_hold"),
        "forecast_brier_improvement_lower_95pct_bound_positive_both_horizons_all_blocks": all(
            item["ci95"][0] > 0 for horizon in forecast.values() for item in horizon["brier_improvement_ci95"]),
        "at_least_7_of_9_neighborhood_variants_positive_cagr_and_dd_above_minus_40pct": sum(
            item["cagr"] > 0 and item["max_drawdown"] >= -0.4 for item in neighbors) >= 7,
        "50bps_cost_stress_positive_cagr": next(item for item in stress if item["case"] == "cost_50bps")["cagr"] > 0,
        "extra_day_delay_positive_cagr": next(item for item in stress if item["case"] == "delay_3_days")["cagr"] > 0,
    }
    assert set(acceptance) == set(config["acceptance"])
    summary = {
        "model": config["name"], "data": manifest, "metrics": metrics,
        "validation_forecast": validation_forecast, "holdout_forecast": forecast,
        "holdout_sharpe_uncertainty": uncertainty, "rolling_3year_descriptive": rolling,
        "sensitivity": neighbors, "stress": stress, "checks": checks,
        "acceptance": acceptance, "all_acceptance_checks_passed": all(acceptance.values()),
        "latest_closed_candle_signal": {
            "candle_date_utc": str(data.index[-1].date()), "close_usdt": float(current["close"]),
            "sma200": float(current["sma"]), "momentum63": float(current["momentum"]),
            "annual_volatility30": float(current["volatility"]), "score": int(current["score"]),
            "unexecuted_target_btc_weight": float(current["trend_model"]),
            "earliest_assumed_execution_utc": str((data.index[-1] + pd.Timedelta(days=2)).date()),
        },
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "research_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save_json(ROOT / "summary.json", summary)
    save_plot(holdout, forecast)
    print(json.dumps({"holdout": metrics["holdout"], "forecast": forecast,
                      "acceptance": acceptance, "latest": summary["latest_closed_candle_signal"]}, indent=2))


if __name__ == "__main__":
    main()
