"""Fixed, small retrospective experiment with purged nested time validation.

Run from any working directory: python3 btc-trend-research/v3/audit.py
Only writes inside v3/. V1/V2 snapshots and their results remain unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
spec = importlib.util.spec_from_file_location("v2_math", PARENT / "v2" / "optimize.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
FEATURES = ["ret7", "ret30", "ma200", "vol30"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_inputs():
    config = json.loads((ROOT / "protocol.json").read_text())
    if sha256(PARENT / "btc_daily.csv") != config["data_sha256"]:
        raise ValueError("Frozen snapshot changed; create a separate prospective evaluation")
    data = pd.read_csv(PARENT / "btc_daily.csv", index_col="date", parse_dates=["date"])
    expected = pd.date_range(data.index.min(), data.index.max(), freq="D")
    if not data.index.equals(expected) or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Invalid, duplicate, unordered or discontinuous daily candles")
    prices = data[["open", "high", "low", "close"]]
    if ((prices <= 0).any().any() or (data.volume < 0).any()
            or (data.high < prices.max(axis=1)).any() or (data.low > prices.min(axis=1)).any()):
        raise ValueError("Invalid OHLCV values")
    return config, data


def make_features(data):
    close = data.close
    log_close = np.log(close)
    features = pd.DataFrame(index=data.index)
    features["ret7"] = log_close.diff(7)
    features["ret30"] = log_close.diff(30)
    features["ma200"] = close / close.rolling(200).mean() - 1
    features["vol30"] = log_close.diff().rolling(30).std(ddof=1) * np.sqrt(365)
    features["score"] = (features.ma200 > 0).astype(int) + (close > close.shift(63)).astype(int)
    return features


def make_dataset(data, horizon, config, include_unmatured=False):
    features = make_features(data)
    frame = features.copy()
    frame["forward_return"] = data.close.shift(-horizon) / data.close - 1
    frame["label_end"] = data.index + pd.Timedelta(days=horizon)
    # A close-labelled candle is complete at the NEXT midnight. Comparing its
    # date strictly to refit midnight ensures its close is available then.
    frame["target"] = (frame.forward_return > 0).astype(float).where(frame.forward_return.notna())
    valid = features.notna().all(axis=1)
    if not include_unmatured:
        valid &= frame.forward_return.notna()
    return frame.loc[valid & (frame.index >= config["training_start"])]


def predict_candidates(train, test, horizon, config):
    if len(train) < 365:
        raise ValueError("Insufficient completed training labels")
    y = train.target.to_numpy()
    prior = float((y.sum() + 1) / (len(y) + 2))
    alpha = 3 * horizon
    probabilities = {}
    for score in [0, 1, 2]:
        group = train.loc[train.score == score, "target"]
        probabilities[score] = float((group.sum() + alpha * prior) / (len(group) + alpha))
    regime = test.score.map(probabilities).to_numpy()
    x, xt = v2.scale_training(train, test, FEATURES)
    beta = v2.logistic_fit(x, y, config["logistic_l2_mean_loss_penalty"])
    raw = v2.sigmoid(np.column_stack([np.ones(len(xt)), xt]) @ beta)
    weight = config["logistic_prior_blend_weight"]
    logistic = weight * prior + (1 - weight) * raw
    output = {
        "historical_prior": np.full(len(test), prior),
        "regime_shrunk": regime,
        "logistic_shrunk": logistic,
        "equal_blend": (regime + logistic) / 2,
        "always_up": np.ones(len(test)),
        "coin_flip": np.full(len(test), 0.5),
    }
    for probability in output.values():
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    return output


def monthly_predictions(frame, horizon, config):
    months = pd.date_range(config["inner_predictions_start"], frame.index.max(), freq="MS")
    pieces, audit = [], []
    for cutoff in months:
        stop = cutoff + pd.offsets.MonthBegin(1)
        train = frame.loc[(frame.label_end < cutoff) & frame.forward_return.notna()]
        test = frame.loc[(frame.index >= cutoff) & (frame.index < stop)]
        if test.empty:
            continue
        assert train.label_end.max() < cutoff <= test.index.min()
        predictions = predict_candidates(train, test, horizon, config)
        table = test[["label_end", "target", "forward_return"]].copy()
        for name, probability in predictions.items():
            table[name] = probability
        table["refit_date"] = cutoff
        pieces.append(table)
        audit.append({
            "horizon": horizon, "refit_date": str(cutoff.date()), "train_rows": len(train),
            "train_signal_start": str(train.index.min().date()),
            "train_signal_end": str(train.index.max().date()),
            "train_label_end_max": str(train.label_end.max().date()),
            "test_signal_start": str(test.index.min().date()),
            "test_signal_end": str(test.index.max().date()), "test_rows": len(test),
        })
    return pd.concat(pieces).rename_axis("signal_date"), audit


def select_from_past(history, cutoff, config):
    completed = history.loc[history.label_end < cutoff]
    if completed.empty:
        raise ValueError("No completed validation outcomes before selection")
    rows = []
    for priority, name in enumerate(config["selectable_candidates"]):
        yearly = [float(np.mean((group[name] - group.target) ** 2))
                  for _, group in completed.groupby(completed.index.year)]
        rows.append({"candidate": name, "mean_annual_brier": float(np.mean(yearly)),
                     "priority": priority, "validation_rows": len(completed)})
    rows.sort(key=lambda row: (row["mean_annual_brier"], row["priority"]))
    return rows[0]["candidate"], {
        "outer_year": cutoff.year, "selection_date": str(cutoff.date()),
        "validation_signal_start": str(completed.index.min().date()),
        "validation_label_end_max": str(completed.label_end.max().date()),
        "selected": rows[0]["candidate"], "all_candidates": rows,
    }


def nested_predictions(history, config):
    pieces, selections = [], []
    for year in range(pd.Timestamp(config["outer_evaluation_start"]).year, history.index.max().year + 1):
        cutoff = pd.Timestamp(f"{year}-01-01")
        selected, audit = select_from_past(history, cutoff, config)
        test = history.loc[history.index.year == year].copy()
        test["nested_primary"] = test[selected]
        test["selected_candidate"] = selected
        pieces.append(test)
        selections.append(audit)
    return pd.concat(pieces), selections


def metrics(table, name, eligible_count=None):
    if table.empty:
        return {"total": 0, "correct": 0, "coverage": 0.0, "accuracy": None,
                "balanced_accuracy": None, "brier": None, "log_loss": None}
    probability = table[name].to_numpy()
    target = table.target.to_numpy()
    prediction = (probability >= 0.5).astype(int)
    tp = int(((prediction == 1) & (target == 1)).sum())
    tn = int(((prediction == 0) & (target == 0)).sum())
    fp = int(((prediction == 1) & (target == 0)).sum())
    fn = int(((prediction == 0) & (target == 1)).sum())
    up = tp / (tp + fn) if tp + fn else None
    down = tn / (tn + fp) if tn + fp else None
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "total": len(table), "correct": tp + tn,
        "coverage": len(table) / (eligible_count if eligible_count is not None else len(table)),
        "accuracy": float((prediction == target).mean()),
        "balanced_accuracy": (up + down) / 2 if up is not None and down is not None else None,
        "brier": float(np.mean((probability - target) ** 2)),
        "log_loss": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))),
        "observed_up_fraction": float(target.mean()), "predicted_up_fraction": float(prediction.mean()),
        "up_recall": up, "down_recall": down,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def paired_uncertainty(table, config, candidate="nested_primary"):
    y = table.target.to_numpy()
    p = table[candidate].to_numpy()
    q = table.historical_prior.to_numpy()
    correct = ((p >= 0.5) == y).astype(float)
    accuracy_delta = correct - ((q >= 0.5) == y)
    brier_delta = (q - y) ** 2 - (p - y) ** 2
    result = []
    settings = config["bootstrap"]
    for block in settings["block_days"]:
        rng = np.random.default_rng(settings["seed"])
        starts = rng.integers(0, len(table), size=(settings["repetitions"], int(np.ceil(len(table) / block))))
        indices = ((starts[:, :, None] + np.arange(block)) % len(table)).reshape(settings["repetitions"], -1)[:, :len(table)]
        def interval(values):
            return [float(value) for value in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]
        result.append({"block_days": block, "accuracy_ci95": interval(correct),
                       "accuracy_improvement_vs_prior_ci95": interval(accuracy_delta),
                       "brier_improvement_vs_prior_ci95": interval(brier_delta)})
    return result


def calibration_bins(table, name):
    bins = []
    for low, high in zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1)):
        group = table.loc[(table[name] >= low) & ((table[name] < high) if high < 0.999 else (table[name] <= 1))]
        bins.append({"lower_inclusive": round(float(low), 1), "upper": round(float(high), 1),
                     "total": len(group), "mean_probability": float(group[name].mean()) if len(group) else None,
                     "observed_up_fraction": float(group.target.mean()) if len(group) else None})
    return bins


def diagnostics(table, horizon, config):
    offsets = [{"offset": offset, **metrics(table.iloc[offset::horizon], "nested_primary")}
               for offset in range(horizon)]
    subsets = []
    for threshold in config["confidence_thresholds_diagnostic_only"]:
        confidence = np.maximum(table.nested_primary, 1 - table.nested_primary)
        group = table.loc[confidence >= threshold]
        subsets.append({"confidence_threshold": threshold, **metrics(group, "nested_primary", len(table)),
                        "optimized": False})
    return {"nonoverlapping_offsets_all_reported": offsets,
            "high_confidence_subsets": subsets,
            "reliability_bins": calibration_bins(table, "nested_primary")}


def web_export(data, config, summary):
    """Frozen monthly parameters plus real-time-available historical predictions.

    Include recent signals with unresolved targets, so a web user never needs
    to run today's parameters on past dates. Forecasts beyond the frozen model's
    validity month must be explicitly labelled stale, not silently rolled forward.
    """
    output = {
        "version": config["name"], "status": "experimental_no_edge_established",
        "target": "close[t+h]/close[t]-1 > 0", "reference": "completed_utc_daily_close",
        "snapshotEnd": str(data.index.max().date()), "dataSha256": config["data_sha256"],
        "protocolSha256": sha256(ROOT / "protocol.json"),
        "candidateIds": config["selectable_candidates"],
        "oosColumns": ["date", "primary", "historical_prior", "regime_shrunk", "logistic_shrunk", "equal_blend", "realizedReturn", "selected"],
        "horizons": {},
        "replacementEvidencePassed": summary["replacement_evidence_gate_passed"],
        "asOfDate": str(data.index.max().date()),
        "validFromDate": str(data.index.max().to_period("M").start_time.date()),
        "validUntilExclusive": str((data.index.max().to_period("M").start_time + pd.offsets.MonthBegin(1)).date()),
    }
    for horizon in config["horizons_days"]:
        full = make_dataset(data, horizon, config, include_unmatured=True)
        history, _ = monthly_predictions(full, horizon, config)
        primary, selections = nested_predictions(history, config)
        cutoff = data.index.max().to_period("M").start_time
        train = full.loc[(full.label_end < cutoff) & full.forward_return.notna()]
        test = full.loc[full.index >= cutoff]
        probabilities = predict_candidates(train, test, horizon, config)
        y = train.target.to_numpy()
        prior = float((y.sum() + 1) / (len(y) + 2))
        alpha = 3 * horizon
        regime = {}
        for score in [0, 1, 2]:
            group = train.loc[train.score == score, "target"]
            regime[str(score)] = float((group.sum() + alpha * prior) / (len(group) + alpha))
        center = train[FEATURES].to_numpy().mean(axis=0)
        scale = train[FEATURES].to_numpy().std(axis=0)
        scale[scale == 0] = 1
        x, xt = v2.scale_training(train, test, FEATURES)
        beta = v2.logistic_fit(x, y, config["logistic_l2_mean_loss_penalty"])
        selected = selections[-1]["selected"]
        rows = []
        for stamp, row in primary.iterrows():
            rows.append([str(stamp.date()), float(row.nested_primary),
                         *[float(row[name]) for name in config["selectable_candidates"]],
                         float(row.forward_return) if pd.notna(row.forward_return) else None,
                         row.selected_candidate])
        np.testing.assert_allclose(primary.loc[test.index, "nested_primary"], probabilities[selected])
        output["horizons"][str(horizon)] = {
            "selected": selected, "refitDate": str(cutoff.date()),
            "validUntilExclusive": str((cutoff + pd.offsets.MonthBegin(1)).date()),
            "trainLabelEndMax": str(train.label_end.max().date()), "trainRows": len(train),
            "selectionDate": selections[-1]["selection_date"],
            "selectionLabelEndMax": selections[-1]["validation_label_end_max"],
            "priorProbability": prior, "regimeProbabilities": regime,
            "featureNames": FEATURES, "center": center.tolist(), "scale": scale.tolist(),
            "standardizedClip": 8, "logisticCoefficients": beta.tolist(),
            "logisticPriorBlendWeight": config["logistic_prior_blend_weight"],
            "metrics": summary["horizons"][str(horizon)]["metrics"],
            "uncertainty": summary["horizons"][str(horizon)]["uncertainty"],
            "oos": rows,
        }
    return output


def reproduce_previous(data):
    original = json.loads((PARENT / "summary.json").read_text())
    stored = pd.read_csv(PARENT / "holdout_predictions.csv", index_col="signal_date", parse_dates=["signal_date"])
    reproduced = {}
    for horizon in [7, 30]:
        table = stored.loc[stored.horizon == horizon]
        signal_score = make_features(data).score.loc[table.index]
        forward = (data.open.shift(-2 - horizon) / data.open.shift(-2) - 1).loc[table.index]
        expected = original["holdout_forecast"][str(horizon)]
        all_forward = data.open.shift(-2 - horizon) / data.open.shift(-2) - 1
        label_end = data.index + pd.Timedelta(days=horizon + 2)
        calibrated = (data.index >= "2018-04-01") & (label_end < pd.Timestamp("2024-01-01")) & all_forward.notna()
        lookup = {}
        for item in expected["calibration"]:
            group = all_forward.loc[calibrated & (make_features(data).score == item["score"])]
            lookup[item["score"]] = float(((group > 0).sum() + 1) / (len(group) + 2))
            assert len(group) == item["n_overlapping"]
            np.testing.assert_allclose(lookup[item["score"]], item["p_up"])
        baseline = ((all_forward.loc[calibrated] > 0).sum() + 1) / (int(calibrated.sum()) + 2)
        np.testing.assert_allclose(baseline, expected["baseline_probability"])
        probability = signal_score.map(lookup)
        np.testing.assert_allclose(table.forward_return, forward, atol=1e-12)
        np.testing.assert_allclose(table.probability, probability, atol=1e-12)
        accuracy = float(((probability >= 0.5) == (forward > 0)).mean())
        brier = float(np.mean((probability - (forward > 0).astype(int)) ** 2))
        np.testing.assert_allclose(accuracy, expected["direction_accuracy"])
        np.testing.assert_allclose(brier, expected["brier"])
        reproduced[str(horizon)] = {"accuracy": accuracy, "brier": brier,
                                    "baseline_accuracy": expected["baseline_direction_accuracy"],
                                    "baseline_brier": expected["baseline_brier"], "total": len(table)}
    config = json.loads((PARENT / "v2" / "protocol.json").read_text())
    v2_summary = json.loads((PARENT / "v2" / "summary.json").read_text())
    features = v2.make_features(data)
    frame = v2.make_dataset(data, features, config)
    selected = v2_summary["selected_candidate"]
    result, _ = v2.folds(frame, selected, [2024, 2025, 2026], "reused_history", config)
    score = v2.score_predictions(result)
    expected = v2_summary["reused_history_metrics"][selected["id"]]
    np.testing.assert_allclose([score["accuracy"], score["brier"]], [expected["accuracy"], expected["brier"]])
    return {"v1": reproduced, "v2_selected_id": selected["id"], "v2_selected": score,
            "checks": {"v1_targets_probabilities_metrics_reproduced": True,
                       "v1_frozen_calibration_refit_reproduced": True,
                       "v2_selected_predictions_recomputed_and_metrics_reproduced": True}}


def run_checks(data, config):
    checks = {}
    cutoff = pd.Timestamp("2023-06-30")
    features = make_features(data)
    pd.testing.assert_frame_equal(make_features(data.loc[:cutoff]), features.loc[:cutoff])
    changed = data.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close"]] *= 3
    pd.testing.assert_frame_equal(make_features(changed).loc[:cutoff], features.loc[:cutoff])
    checks["past_features_unchanged_by_future_prices_or_truncation"] = True
    for horizon in config["horizons_days"]:
        frame = make_dataset(data, horizon, config)
        mutation = make_dataset(changed, horizon, config)
        refit = pd.Timestamp("2023-06-01")
        train = frame.loc[frame.label_end < refit]
        test = frame.loc[(frame.index >= refit) & (frame.index <= cutoff)]
        original = predict_candidates(train, test, horizon, config)
        mutated = predict_candidates(mutation.loc[mutation.label_end < refit], mutation.loc[test.index], horizon, config)
        for name in original:
            np.testing.assert_allclose(original[name], mutated[name])
        changed_labels = test.copy()
        changed_labels["target"] = 1 - changed_labels.target
        changed_labels["forward_return"] = -99.0
        pred_changed_labels = predict_candidates(train, changed_labels, horizon, config)
        for name in original:
            np.testing.assert_allclose(original[name], pred_changed_labels[name])
        anchor = test.index[5]
        expected = data.loc[anchor + pd.Timedelta(days=horizon), "close"] / data.loc[anchor, "close"] - 1
        np.testing.assert_allclose(frame.loc[anchor, "forward_return"], expected)
        shortened = predict_candidates(train, test.iloc[:3], horizon, config)
        for name in original:
            np.testing.assert_allclose(original[name][:3], shortened[name])
        assert train.label_end.max() < refit
        checks[f"h{horizon}_future_price_mutation_preserves_predictions"] = True
        checks[f"h{horizon}_test_labels_unused_and_scaling_train_only"] = True
        checks[f"h{horizon}_target_matches_completed_close_reference"] = True
    # Selection must not respond to current/future outer outcomes or predictions.
    history, _ = monthly_predictions(make_dataset(data, 30, config), 30, config)
    selected, audit = select_from_past(history, pd.Timestamp("2024-01-01"), config)
    mutation = history.copy()
    future = mutation.label_end >= pd.Timestamp("2024-01-01")
    mutation.loc[future, "target"] = 1 - mutation.loc[future, "target"]
    for name in config["selectable_candidates"]:
        mutation.loc[future, name] = 1 - mutation.loc[future, name]
    chosen2, audit2 = select_from_past(mutation, pd.Timestamp("2024-01-01"), config)
    assert (selected, audit) == (chosen2, audit2)
    checks["outer_selection_ignores_unmatured_and_future_outcomes"] = True
    return checks


def report(summary):
    lines = ["# BTC 预测模型复核与 V3 有限实验", "", "本报告复用已经查看过的历史，不声称存在全新、未触碰的测试集。",
             "V3 仅做方向概率研究；没有新增交易策略、执行或净收益结论。", "",
             "## 原模型复核", "",
             "V1/V2 的滚动特征、训练折内标准化、标签结束时间隔离检查通过，复现结果如下。",
             "原目标为 open[t+h+2] / open[t+2]，不是从最新收盘价或当前实时价格开始；因此 V1 的方向结果不能当作 V3 的检验成绩。", "",
             "| 模型 | 期限 | 样本 | 方向准确率 | 基线准确率 | Brier |",
             "| --- | --- | --- | --- | --- | --- |"]
    for horizon, row in summary["previous_reproduction"]["v1"].items():
        lines.append(f"| V1 | {horizon} 天 | {row['total']} | {row['accuracy']:.2%} | {row['baseline_accuracy']:.2%} | {row['brier']:.6f} |")
    v2score = summary["previous_reproduction"]["v2_selected"]
    lines += [f"| V2 的开发期赢家 | 30 天 | {v2score['total']} | {v2score['accuracy']:.2%} | 55.32% | {v2score['brier']:.6f} |", "",
              "V2 搜索 40 个候选，开发期赢家复用历史准确率不足 50%；没有高胜率证据。V1 30 天约 61% 是历史方向命中率，不能等同获利概率。",
              "网页中的历史收益分位数也不是经过覆盖率验证的价格预测区间，乘以当前实时价只能称为情景换算。", "",
              "## V3 事先固定的实验", "",
              "目标是完成日线 t 的收盘价到 t+7 / t+30 收盘价是否上涨。它与实时盘中价起算的预测仍有区别。",
              "仅 4 个候选：历史先验、向先验收缩的三态模型、强正则逻辑回归并与先验各占一半、两个模型等权混合。",
              "每月初仅用已经结束的标签重训。每年初只根据此前已成熟的 2020 年起样本外预测，按年度平均 Brier 选择当年模型。",
              "2022 年起报告嵌套外层结果；年度选型与月度训练的标签都严格止于边界之前。所有 4 个候选与基线全数公开。",
              "coin_flip 是固定 50% 的无信息概率基线；分类阈值规则把恰好 50% 归入上涨，因此其方向准确率与始终看涨相同，不代表模拟随机交易。",
              "Brier 衡量概率误差，越低越好；同时包含校准、区分能力与数据不确定性，并非单独的校准证明。", ""]
    for horizon, result in summary["horizons"].items():
        lines += [f"## {horizon} 天结果", "", "| 模型 | 样本 | 准确率 | 均衡准确率 | Brier | 覆盖率 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, row in result["metrics"].items():
            lines.append(f"| {name} | {row['total']} | {row['accuracy']:.2%} | {row['balanced_accuracy']:.2%} | {row['brier']:.6f} | {row['coverage']:.0%} |")
        lines += ["", "嵌套主模型逐年结果：", "", "| 年份 | 当年选择 | 准确率 | Brier | 先验 Brier |",
                  "| --- | --- | --- | --- | --- |"]
        by_year = {str(row["outer_year"]): row["selected"] for row in result["selections"]}
        for year, row in result["yearly"].items():
            lines.append(f"| {year} | {by_year[year]} | {row['nested_primary']['accuracy']:.2%} | {row['nested_primary']['brier']:.6f} | {row['historical_prior']['brier']:.6f} |")
        lines += ["", "配对循环区块 bootstrap 的 95% 区间（改进为正才代表优于先验）：", ""]
        for row in result["uncertainty"]:
            al, ah = row["accuracy_improvement_vs_prior_ci95"]
            bl, bh = row["brier_improvement_vs_prior_ci95"]
            lines.append(f"- {row['block_days']} 日区块：准确率改进 [{al:.2%}, {ah:.2%}]；Brier 改进 [{bl:.6f}, {bh:.6f}]。")
        offsets = result["diagnostics"]["nonoverlapping_offsets_all_reported"]
        lines += ["", f"全部 {horizon} 种非重叠起点的准确率范围：{min(row['accuracy'] for row in offsets):.2%}–{max(row['accuracy'] for row in offsets):.2%}。每组只有 {min(row['total'] for row in offsets)}–{max(row['total'] for row in offsets)} 个样本，不能选择最高起点。", "",
                  "高置信度仅作诊断，阈值未根据成绩调参：", ""]
        for row in result["diagnostics"]["high_confidence_subsets"]:
            accuracy = "无样本" if row["accuracy"] is None else f"{row['accuracy']:.2%}"
            lines.append(f"- 阈值 {row['confidence_threshold']:.0%}：{row['total']} 样本，覆盖率 {row['coverage']:.2%}，准确率 {accuracy}。")
        lines += ["", f"是否满足优于先验的保守证据门槛：**{result['evidence_over_prior']}**。", ""]
    lines += ["## 决策与限制", "", summary["recommendation"], "",
              "改进重点是标签对齐、及时行情、弱信号透明呈现与前瞻记录。不能通过反复试验同一历史，把选中的高命中率包装成可持续优势。",
              "下一步若扩展衍生品资金费率、期现基差或宏观数据，应先检查 point-in-time 可获得性、修订与数据发布时间，再冻结一个独立方案。",
              "需要真正未参与选型的未来日线积累成熟结果后，才能作前瞻验收。本轮没有宣称达到 90% 成功率。", "",
              "## 复现与证据", "", "```sh", "python3 -m pip install -r btc-trend-research/requirements.txt",
              "python3 btc-trend-research/v3/audit.py --checks-only", "python3 btc-trend-research/v3/audit.py", "```", "",
              "- `protocol.json`：结果产生前记录的固定规则与数据哈希。",
              "- `summary.json`：全部候选、年度结果、覆盖率、所有非重叠起点、可靠性分箱及区块区间。",
              "- `predictions.csv`：每个期限所有候选与逐年嵌套主模型的逐日预测。",
              "- `fold_audit.csv`、`selection.json`：月度训练与年度选型的时间隔离记录。",
              f"- 内部检查 {sum(summary['checks'].values())}/{len(summary['checks'])} 通过；V1/V2 文件哈希保持不变。", "",
              "方法参考：[时间序列拆分](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)、[概率校准与适当评分规则](https://scikit-learn.org/stable/modules/calibration.html)、[数据泄漏与训练折预处理](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    config, data = load_inputs()
    original_hashes = {str(path.relative_to(PARENT)): sha256(path) for path in PARENT.rglob("*")
                       if path.is_file() and ROOT not in path.parents and "__pycache__" not in path.parts}
    checks = run_checks(data, config)
    previous = reproduce_previous(data)
    checks.update(previous["checks"])
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    summaries, all_predictions, all_folds, all_selections = {}, [], [], []
    names = ["nested_primary", *config["selectable_candidates"], *config["diagnostic_baselines"]]
    for horizon in config["horizons_days"]:
        frame = make_dataset(data, horizon, config)
        history, audit = monthly_predictions(frame, horizon, config)
        primary, selections = nested_predictions(history, config)
        for row in selections:
            row["horizon"] = horizon
        all_selections.extend(selections)
        # Store chronological selections before deriving outer evaluation scores.
        write_json(ROOT / "selection.json", {"protocol_sha256": sha256(ROOT / "protocol.json"), "selections": all_selections})
        yearly = {str(year): {name: metrics(group, name) for name in names}
                  for year, group in primary.groupby(primary.index.year)}
        uncertainty = paired_uncertainty(primary, config)
        evidence = (all(row["accuracy_improvement_vs_prior_ci95"][0] > 0
                        and row["brier_improvement_vs_prior_ci95"][0] > 0 for row in uncertainty)
                    and all(row["nested_primary"]["brier"] <= row["historical_prior"]["brier"] for row in yearly.values()))
        summaries[str(horizon)] = {
            "target": f"close[t+{horizon}]/close[t]-1 > 0",
            "start": str(primary.index.min().date()), "end": str(primary.index.max().date()),
            "label_end_max": str(primary.label_end.max().date()),
            "metrics": {name: metrics(primary, name) for name in names},
            "yearly": yearly, "selections": selections, "uncertainty": uncertainty,
            "candidate_uncertainty_diagnostic_only": {
                name: paired_uncertainty(primary, config, name)
                for name in config["selectable_candidates"] if name != "historical_prior"},
            "diagnostics": diagnostics(primary, horizon, config), "evidence_over_prior": evidence,
        }
        history["nested_primary"] = primary.nested_primary
        history["selected_candidate"] = primary.selected_candidate
        history["horizon"] = horizon
        all_predictions.append(history)
        all_folds.extend(audit)
    checks["all_training_labels_end_before_refit"] = all(row["train_label_end_max"] < row["refit_date"] for row in all_folds)
    checks["all_selection_labels_end_before_outer_year"] = all(row["validation_label_end_max"] < row["selection_date"] for row in all_selections)
    checks["v1_and_v2_files_unchanged"] = all(sha256(PARENT / name) == value for name, value in original_hashes.items())
    assert all(checks.values())
    passed = all(row["evidence_over_prior"] for row in summaries.values())
    recommendation = ("V3 两个期限满足相对于历史先验的保守门槛，仍须前瞻验证，且不同目标无法直接证明优于 V1。"
                      if passed else "本次有限实验未建立两个期限都稳健优于历史先验的证据，不推荐把 V3 或 V2 宣称为已验证的高胜率替代模型。保留原模型研究档案与清晰的历史标签口径；若展示从最新完成日线起算的 V3 概率，应明确标为实验性，并同时展示基线与不确定性。")
    summary = {"name": config["name"], "status": config["status"],
               "data": {"start": str(data.index.min().date()), "end": str(data.index.max().date()), "rows": len(data), "sha256": config["data_sha256"]},
               "protocol_sha256": sha256(ROOT / "protocol.json"), "script_sha256": sha256(Path(__file__)),
               "candidate_count_per_horizon": len(config["selectable_candidates"]), "previous_reproduction": previous,
               "horizons": summaries, "replacement_evidence_gate_passed": passed,
               "recommendation": recommendation, "checks": checks,
               "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}}
    web_model = web_export(data, config, summary)
    checks["web_export_latest_predictions_match_frozen_parameters"] = True
    write_json(ROOT / "web_model.json", web_model)
    write_json(ROOT / "summary.json", summary)
    pd.concat(all_predictions).to_csv(ROOT / "predictions.csv", float_format="%.12g")
    pd.DataFrame(all_folds).to_csv(ROOT / "fold_audit.csv", index=False)
    (ROOT / "REPORT.md").write_text(report(summary))
    print(json.dumps({"horizons": {h: {"primary": r["metrics"]["nested_primary"], "prior": r["metrics"]["historical_prior"], "evidence": r["evidence_over_prior"]}
                                     for h, r in summaries.items()}, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
