import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
BASIC = ["ret7", "ret30", "ret63", "ma50", "ma200", "vol30", "rsi14", "vol_ratio"]
EXTRA = ["ret1", "ret3", "ret14", "ret126", "ma20", "vol90", "atr14", "volume_z20", "range", "close_position"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    config = json.loads((ROOT / "protocol.json").read_text())
    if sha256(PARENT / "btc_daily.csv") != config["data_sha256"]:
        raise ValueError("Data snapshot differs from the frozen protocol")
    data = pd.read_csv(PARENT / "btc_daily.csv", index_col="date", parse_dates=["date"])
    if not data.index.equals(pd.date_range(data.index.min(), data.index.max(), freq="D")):
        raise ValueError("Daily candles are not unique and continuous")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Nonfinite input data")
    original = json.loads((PARENT / "summary.json").read_text())
    return config, data, original


def make_features(data):
    close = data["close"]
    log_close = np.log(close)
    daily = log_close.diff()
    features = pd.DataFrame(index=data.index)
    for days in [1, 3, 7, 14, 30, 63, 126]:
        features[f"ret{days}"] = log_close.diff(days)
    for days in [20, 50, 200]:
        features[f"ma{days}"] = close / close.rolling(days).mean() - 1
    for days in [30, 90]:
        features[f"vol{days}"] = daily.rolling(days).std(ddof=1) * np.sqrt(365)
    gains = close.diff().clip(lower=0).rolling(14).mean()
    losses = (-close.diff().clip(upper=0)).rolling(14).mean()
    features["rsi14"] = (gains / (gains + losses)).where(gains + losses != 0, 0.5)
    features["vol_ratio"] = features["vol30"] / features["vol90"]
    true_range = pd.concat([data["high"] - data["low"],
                            (data["high"] - close.shift()).abs(),
                            (data["low"] - close.shift()).abs()], axis=1).max(axis=1)
    features["atr14"] = true_range.rolling(14).mean() / close
    log_volume = np.log1p(data["volume"])
    features["volume_z20"] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()
    features["range"] = (data["high"] - data["low"]) / close
    spread = data["high"] - data["low"]
    features["close_position"] = ((close - data["low"]) / spread).where(spread != 0, 0.5)
    features["score"] = ((features["ma200"] > 0).astype(int) + (features["ret63"] > 0).astype(int))
    for name in BASIC:
        features[name + "_squared"] = features[name] ** 2
    features["trend_momentum"] = features["ma200"] * features["ret63"]
    features["rsi_trend"] = (features["rsi14"] - 0.5) * features["ma200"]
    features["momentum_vol"] = features["ret30"] * features["vol30"]
    return features


def feature_names(kind):
    if kind == "basic":
        return BASIC
    if kind == "expanded":
        return BASIC + EXTRA
    if kind == "interactions":
        return BASIC + [name + "_squared" for name in BASIC] + ["trend_momentum", "rsi_trend", "momentum_vol"]
    raise ValueError(kind)


def make_dataset(data, features, config):
    lag, horizon = config["execution_lag_days"], config["forecast_horizon_days"]
    forward = data["open"].shift(-lag - horizon) / data["open"].shift(-lag) - 1
    frame = features.copy()
    frame["forward_return"] = forward
    frame["label_end"] = frame.index + pd.Timedelta(days=lag + horizon)
    frame["target"] = (forward > 0).astype(int)
    valid = features.notna().all(axis=1) & forward.notna() & (frame.index >= config["training_start"])
    frame = frame.loc[valid]
    if not np.isfinite(frame.drop(columns="label_end").to_numpy()).all():
        raise ValueError("Nonfinite derived features")
    return frame


def candidates(config):
    values = [
        {"id": "always_up", "family": "baseline", "window": None},
        {"id": "historical_prior", "family": "baseline", "window": None},
    ]
    for window in config["training_windows_days"]:
        suffix = "expanding" if window is None else f"{window}d"
        values.append({"id": f"regime_{suffix}", "family": "regime", "window": window})
        for feature_set in config["feature_sets"]:
            for penalty in config["logistic_l2_mean_loss_penalties"]:
                values.append({"id": f"logistic_{feature_set}_{suffix}_l2_{penalty}",
                               "family": "logistic", "window": window,
                               "features": feature_set, "penalty": penalty})
            for neighbors in config["knn_neighbors"]:
                values.append({"id": f"knn_{feature_set}_{suffix}_k_{neighbors}",
                               "family": "knn", "window": window,
                               "features": feature_set, "neighbors": neighbors})
    assert len(values) == config["candidate_count"]
    return values


def scale_training(train, test, names):
    x, xt = train[names].to_numpy(), test[names].to_numpy()
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    return np.clip((x - center) / scale, -8, 8), np.clip((xt - center) / scale, -8, 8)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def logistic_fit(x, y, penalty):
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x.shape[1])
    beta[0] = np.log((y.sum() + 1) / (len(y) - y.sum() + 1))
    diagonal = np.r_[0.0, np.repeat(penalty, x.shape[1] - 1)]
    for _ in range(100):
        scores = x @ beta
        probability = sigmoid(scores)
        gradient = x.T @ (probability - y) / len(y) + diagonal * beta
        if np.max(np.abs(gradient)) < 1e-7:
            return beta
        weight = probability * (1 - probability)
        hessian = (x.T * weight) @ x / len(y) + np.diag(diagonal)
        step = np.linalg.solve(hessian, gradient)
        old_loss = np.mean(np.logaddexp(0, scores) - y * scores) + 0.5 * np.sum(diagonal * beta ** 2)
        factor = 1.0
        for _ in range(30):
            proposal = beta - factor * step
            z = x @ proposal
            new_loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(diagonal * proposal ** 2)
            if new_loss <= old_loss:
                beta = proposal
                break
            factor *= 0.5
        else:
            raise ArithmeticError("Logistic line search failed")
    raise ArithmeticError("Logistic optimization did not converge")


def predict(train, test, spec):
    y = train["target"].to_numpy()
    if len(train) < 200:
        raise ValueError("Insufficient training history")
    prior = (y.sum() + 1) / (len(y) + 2)
    if spec["family"] == "baseline":
        return np.full(len(test), 1.0 if spec["id"] == "always_up" else prior)
    if spec["family"] == "regime":
        probabilities = {}
        for score in [0, 1, 2]:
            subset = train.loc[train["score"] == score, "target"]
            probabilities[score] = (subset.sum() + 1) / (len(subset) + 2)
        return test["score"].map(probabilities).to_numpy()
    x, xt = scale_training(train, test, feature_names(spec["features"]))
    if spec["family"] == "logistic":
        beta = logistic_fit(x, y, spec["penalty"])
        return sigmoid(np.column_stack([np.ones(len(xt)), xt]) @ beta)
    if spec["family"] == "knn":
        k = spec["neighbors"]
        distances = np.sum(xt ** 2, axis=1)[:, None] + np.sum(x ** 2, axis=1)[None, :] - 2 * xt @ x.T
        indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
        return (y[indices].sum(axis=1) + 1) / (k + 2)
    raise ValueError(spec)


def folds(frame, spec, years, phase, config):
    predictions, audit = [], []
    for year in years:
        cutoff = pd.Timestamp(f"{year}-01-01")
        stop = pd.Timestamp(f"{year + 1}-01-01")
        train_mask = frame["label_end"] < cutoff
        if spec["window"] is not None:
            train_mask &= frame.index >= cutoff - pd.Timedelta(days=spec["window"])
        test_mask = (frame.index >= cutoff) & (frame.index < stop)
        if phase == "development":
            test_mask &= frame["label_end"] < pd.Timestamp(config["development_label_end_exclusive"])
        train, test = frame.loc[train_mask], frame.loc[test_mask]
        if test.empty:
            raise ValueError("Empty evaluation year")
        assert train["label_end"].max() < cutoff
        probability = predict(train, test, spec)
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError("Invalid prediction probabilities")
        out = test[["label_end", "target", "forward_return"]].copy()
        out["probability"] = probability
        out["prediction"] = (probability >= config["classification_threshold"]).astype(int)
        out["candidate"] = spec["id"]
        out["year"] = year
        predictions.append(out)
        audit.append({"phase": phase, "candidate": spec["id"], "year": year,
                      "train_rows": len(train), "test_rows": len(test),
                      "train_signal_start": str(train.index.min().date()),
                      "train_label_end_max": str(train["label_end"].max().date()),
                      "refit_date": str(cutoff.date()), "test_signal_end": str(test.index.max().date())})
    return pd.concat(predictions).rename_axis("signal_date"), audit


def score_predictions(table):
    y = table["target"].to_numpy()
    pred = table["prediction"].to_numpy()
    probability = table["probability"].to_numpy()
    correct = int((pred == y).sum())
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    up_recall = tp / (tp + fn) if tp + fn else None
    down_recall = tn / (tn + fp) if tn + fp else None
    balanced = (up_recall + down_recall) / 2 if up_recall is not None and down_recall is not None else None
    return {"total": len(table), "correct": correct, "accuracy": correct / len(table),
            "coverage": 1.0, "brier": float(np.mean((probability - y) ** 2)),
            "balanced_accuracy": balanced, "observed_up_fraction": float(y.mean()),
            "predicted_up_fraction": float(pred.mean()), "up_recall": up_recall,
            "down_recall": down_recall, "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def bootstrap_indices(size, block, config):
    settings = config["bootstrap"]
    rng = np.random.default_rng(settings["seed"])
    starts = rng.integers(size, size=(settings["repetitions"], int(np.ceil(size / block))))
    return ((starts[:, :, None] + np.arange(block)) % size).reshape(settings["repetitions"], -1)[:, :size]


def confidence_intervals(table, reference, config):
    assert table.index.equals(reference.index)
    correct = (table["prediction"] == table["target"]).to_numpy().astype(float)
    base_correct = (reference["prediction"] == reference["target"]).to_numpy().astype(float)
    output = []
    for block in config["bootstrap"]["block_days"]:
        index = bootstrap_indices(len(table), block, config)
        accuracy = correct[index].mean(axis=1)
        difference = (correct - base_correct)[index].mean(axis=1)
        output.append({"block_days": block,
                       "accuracy_ci95": np.quantile(accuracy, [0.025, 0.975]).tolist(),
                       "accuracy_improvement_vs_prior_ci95": np.quantile(difference, [0.025, 0.975]).tolist()})
    return output


def selected_diagnostics(table, config):
    threshold = config["secondary_confidence_threshold"]
    confident = table.loc[np.maximum(table["probability"], 1 - table["probability"]) >= threshold]
    selective = {"confidence_threshold": threshold, "total": len(confident),
                 "coverage": len(confident) / len(table),
                 "accuracy": float((confident["prediction"] == confident["target"]).mean()) if len(confident) else None,
                 "counts_toward_primary_goal": False}
    offsets = []
    for offset in range(config["forecast_horizon_days"]):
        subset = table.iloc[offset::config["forecast_horizon_days"]]
        offsets.append({"offset": offset, "total": len(subset),
                        "accuracy": float((subset["prediction"] == subset["target"]).mean())})
    return {"high_confidence_subset": selective, "nonoverlapping_offsets_all_reported": offsets}


def run_checks(data, features, frame, config, specs):
    checks = {}
    cutoff = pd.Timestamp("2022-06-30")
    pd.testing.assert_frame_equal(make_features(data.loc[:cutoff]), features.loc[:cutoff])
    checks["features_prefix_invariance"] = True
    changed = data.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close"]] *= 2
    changed_features = make_features(changed)
    pd.testing.assert_frame_equal(changed_features.loc[:cutoff], features.loc[:cutoff])
    checks["future_changes_do_not_change_past_features"] = True
    row = frame.loc[pd.Timestamp("2021-07-01")]
    expected = data.loc[pd.Timestamp("2021-08-02"), "open"] / data.loc[pd.Timestamp("2021-07-03"), "open"] - 1
    np.testing.assert_allclose(row["forward_return"], expected)
    checks["label_matches_future_entry_and_exit_prices"] = True
    train = frame.loc[frame["label_end"] < "2022-01-01"]
    test = frame.loc[(frame.index >= "2022-01-01") & (frame.index < "2022-02-01")]
    for family in ["regime", "logistic", "knn"]:
        spec = next(item for item in specs if item["family"] == family)
        before = predict(train, test, spec)
        relabeled = test.copy()
        relabeled["target"] = 1 - relabeled["target"]
        relabeled["forward_return"] = 999
        np.testing.assert_array_equal(before, predict(train, relabeled, spec))
        np.testing.assert_allclose(before[:5], predict(train, test.iloc[:5], spec))
    checks["evaluation_labels_never_enter_predictions"] = True
    checks["test_subset_does_not_change_scaling_or_predictions"] = True
    rng = np.random.default_rng(123)
    x = rng.normal(size=(100, 3))
    x = np.concatenate([x, x])
    y = np.r_[np.zeros(100), np.ones(100)]
    beta = logistic_fit(x, y, 0.01)
    np.testing.assert_allclose(sigmoid(np.column_stack([np.ones(len(x)), x]) @ beta), 0.5, atol=1e-7)
    checks["balanced_identical_features_give_half_probability"] = True
    flipped_frame = make_dataset(changed, changed_features, config)
    spec = next(item for item in specs if item["family"] == "logistic")
    original_predictions, _ = folds(frame, spec, [2022], "reused_history", config)
    mutated_predictions, _ = folds(flipped_frame, spec, [2022], "reused_history", config)
    np.testing.assert_allclose(original_predictions.loc[:cutoff, "probability"],
                               mutated_predictions.loc[:cutoff, "probability"])
    checks["future_price_changes_do_not_change_past_model_predictions"] = True
    return checks


def render_report(summary):
    lines = [
        "BTC-Direction-V2：方向预测优化报告", "",
        "结论：" + ("在本次复用历史检验中达到点估计目标，但不代表未来保证。" if summary["primary_goal_achieved"] else "未达到全覆盖方向准确率超过90%的目标。"),
        "主要口径：预测close t之后open t+2至open t+32的涨跌；严格大于0为上涨，其余为非上涨。",
        "准确率=正确预测日期数/全部可评估日期数，阈值固定0.5，所有日期必须预测。",
        "不把价格拟合R²、Token准确率、置信度、忽略小波动后的命中率替代本指标。", "",
        "一、搜索范围与选择规则",
        "预先冻结40组：基准2组、状态概率2组、正则逻辑回归18组、近邻18组。",
        "特征集为basic/expanded/interactions；训练窗为扩展历史/730日；每年重新拟合。",
        "basic：7/30/63日对数收益、50/200日均线偏离、30日年化波动、14日简单滚动RSI、30/90日波动比。",
        "expanded额外加入1/3/14/126日收益、20日均线偏离、90日波动、ATR14/价格、成交量z分数、日振幅及收盘区间位置。",
        "interactions加入basic平方项和3个预定义交互项，没有看检验期结果后选择特征。",
        "逻辑回归L2均值损失惩罚为0.001/0.01/0.1；近邻k为25/75/150。",
        "标准化均值与方差只来自各次训练集，标准化值固定裁剪到[-8,8]。",
        "候选按2020-2023年各年准确率均值降序，再按最差年准确率降序、Brier升序、ID排序。",
        "开发期获胜成绩受40次比较的选择偏差影响，不作为无偏收益或准确率估计。",
        "本轮最优仅指此候选集合与选择规则，不声称全局最优。", "",
        "二、时序与数据",
        f"冻结BTCUSDT数据：{summary['data']['start']}至{summary['data']['end']}，{summary['data']['rows']}根日线。",
        "训练起点2018-04-01；每年元旦拟合，所有训练标签必须在该日之前结束。",
        "例如2024年拟合时，2023年12月涉及2024收益的标签不能进入训练。",
        "开发期最后一批标签必须在2024-01-01之前结束，保证选择时不使用2024收益。",
        "2024年后的赢家检验每年只用当年元旦已知的历史标签重新拟合；不使用未来年份。",
        "所有主模型在相同可评估日期比较，未按预测正确与否删样本。",
        "重要：2024年后数据已在V1研究中看过，本轮只是复用历史检验，不是全新独立测试或前瞻实验。", "",
        "三、开发期选出的唯一主候选",
        json.dumps(summary["selected_candidate"], ensure_ascii=False),
        f"开发期平均年度准确率：{summary['selected_development']['mean_year_accuracy']:.2%}；最差年度：{summary['selected_development']['worst_year_accuracy']:.2%}。",
        "该选择在计算复用历史指标之前写入selection.json；查看检验后不更换主候选。", "",
        "四、复用历史检验（全部覆盖）",
        "模型 | 正确/总数 | accuracy | 均衡准确率 | Brier",
    ]
    for name, values in summary["reused_history_metrics"].items():
        lines.append(f"{name} | {values['correct']}/{values['total']} | {values['accuracy']:.2%} | {values['balanced_accuracy']:.2%} | {values['brier']:.6f}")
    lines.extend(["", "主候选逐年表现："])
    for year, values in summary["selected_yearly"].items():
        lines.append(f"{year}：{values['correct']}/{values['total']} = {values['accuracy']:.2%}；上涨召回{values['up_recall']:.2%}，下跌/持平召回{values['down_recall']:.2%}。")
    lines.extend(["", "主候选统计不确定性：配对循环区块bootstrap1000次，标签有30日重叠，不当作独立交易。"])
    for item in summary["selected_confidence_intervals"]:
        low, high = item["accuracy_ci95"]
        dl, dh = item["accuracy_improvement_vs_prior_ci95"]
        lines.append(f"{item['block_days']}日区块：准确率95%区间[{low:.2%}, {high:.2%}]，相对历史先验基准改进区间[{dl:.2%}, {dh:.2%}]。")
    diag = summary["selected_diagnostics"]
    subset = diag["high_confidence_subset"]
    accuracy = "无样本，不可定义" if subset["accuracy"] is None else f"{subset['accuracy']:.2%}"
    offsets = diag["nonoverlapping_offsets_all_reported"]
    lines.extend([
        "这些区间条件于已选模型及预测，不涵盖所有参数搜索不确定性，也不保证市场平稳。",
        f"高置信度辅助诊断：预测置信度至少90%的样本{str(subset['total'])}个，覆盖率{subset['coverage']:.2%}，实际准确率{accuracy}。",
        "高置信度子集不计入90%主目标；没有预测时绝不能报100%准确率。",
        f"每30天抽取一个不重叠标签，报告所有30种起点：准确率范围{min(x['accuracy'] for x in offsets):.2%}至{max(x['accuracy'] for x in offsets):.2%}。",
        "各起点只有少量样本且彼此相关，不可选择其中最高者宣称达标。", "",
        "五、是否达到90%与停止理由",
        f"主候选accuracy>90%且coverage=100%：{summary['primary_goal_achieved']}。",
        f"所有检验年度均>90%：{summary['every_reused_year_above_90pct']}。",
        f"所有区块设定的95%准确率区间下界均>90%：{summary['all_accuracy_ci_lower_bounds_above_90pct']}。",
        "预先固定的三轮40候选已完成。若仍未达标，继续反复针对同一历史调参将放大过拟合，不能把它包装成可靠提升。",
        "后续可信的改进需要新的数据来源或前瞻数据，以及单独冻结的研究方案，不是保证一个预设数字。",
        "本轮只验证方向分类，未为V2增加交易执行及净收益回测；方向命中率不等于可盈利性。",
        "不做空、不加杠杆等V1交易限制没有因分类实验被改动；本轮不执行任何交易。", "",
        "六、复现与完整记录",
        "运行 python3 /Users/cc/code/github/qoder-test/btc-trend-research/v2/optimize.py",
        "检查 python3 /Users/cc/code/github/qoder-test/btc-trend-research/v2/optimize.py --checks-only",
        "summary.json：accuracy使用统一数字字段，范围0到1，并同时提供correct、total、coverage。",
        "development_leaderboard.csv：全部40个候选，禁止只展示获胜者。",
        "development_predictions.csv：所有候选开发期逐日预测。",
        "reused_history_predictions.csv：预先选定的家族赢家、基准与V1逐日预测。",
        "fold_audit.csv：各次训练最大标签时间与预测时间，可审计隔离。",
        "selection.json：开发期选择与协议/数据哈希。protocol.json：固定实验规则。",
        "原V1目录内已有文件保持不变。无新依赖、插件、外部付费服务或API请求。",
        "无法读取credits账单；通过限制40候选、仅本地计算控制开销，不声称实际credits精确值。",
        f"数据SHA256：{summary['data']['sha256']}",
        f"内部检查：{sum(summary['checks'].values())}/{len(summary['checks'])}通过。",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    config, data, original = load_inputs()
    original_hashes = {path.name: sha256(path) for path in PARENT.iterdir() if path.is_file()}
    features = make_features(data)
    frame = make_dataset(data, features, config)
    specs = candidates(config)
    checks = run_checks(data, features, frame, config, specs)
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    rows, development_predictions, audit = [], [], []
    for spec in specs:
        table, fold_audit = folds(frame, spec, config["development_years"], "development", config)
        score = score_predictions(table)
        annual = [score_predictions(group)["accuracy"] for _, group in table.groupby("year")]
        rows.append({**spec, **{key: value for key, value in score.items() if key != "confusion"},
                     "mean_year_accuracy": float(np.mean(annual)), "worst_year_accuracy": float(min(annual)),
                     **{f"accuracy_{year}": score_predictions(group)["accuracy"] for year, group in table.groupby("year")}})
        development_predictions.append(table)
        audit.extend(fold_audit)
    board = pd.DataFrame(rows).sort_values(
        ["mean_year_accuracy", "worst_year_accuracy", "brier", "id"], ascending=[False, False, True, True])
    board.to_csv(ROOT / "development_leaderboard.csv", index=False)
    pd.concat(development_predictions).to_csv(ROOT / "development_predictions.csv")
    selected_id = str(board.iloc[0]["id"])
    by_id = {spec["id"]: spec for spec in specs}
    family_winners = {family: str(board.loc[board["family"] == family].iloc[0]["id"])
                      for family in ["baseline", "regime", "logistic", "knn"]}
    selection = {"selected_candidate": by_id[selected_id], "family_winners": family_winners,
                 "protocol_sha256": sha256(ROOT / "protocol.json"), "data_sha256": config["data_sha256"]}
    selection_path = ROOT / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Selection changed; do not overwrite an already evaluated experiment")
    write_json(selection_path, selection)
    print("Selected before reused-history evaluation:", json.dumps(selection["selected_candidate"]), flush=True)
    reused_ids = sorted(set(family_winners.values()) | {selected_id, "historical_prior", "always_up"})
    reused_years = list(range(pd.Timestamp(config["reused_history_start"]).year, frame.index.max().year + 1))
    tables, reused_metrics = {}, {}
    for candidate_id in reused_ids:
        table, fold_audit = folds(frame, by_id[candidate_id], reused_years, "reused_history", config)
        tables[candidate_id] = table
        reused_metrics[candidate_id] = score_predictions(table)
        audit.extend(fold_audit)
    reference = tables["historical_prior"]
    original_predictions = pd.read_csv(PARENT / "holdout_predictions.csv", index_col="signal_date", parse_dates=["signal_date"])
    original_predictions = original_predictions.loc[original_predictions["horizon"] == 30]
    assert original_predictions.index.equals(reference.index)
    np.testing.assert_array_equal(original_predictions["up"], reference["target"])
    v1 = reference.copy()
    v1["probability"] = original_predictions["probability"]
    v1["prediction"] = (v1["probability"] >= 0.5).astype(int)
    v1["candidate"] = "V1_frozen_probability"
    tables["V1_frozen_probability"] = v1
    reused_metrics["V1_frozen_probability"] = score_predictions(v1)
    np.testing.assert_allclose(reused_metrics["V1_frozen_probability"]["accuracy"],
                               original["holdout_forecast"]["30"]["direction_accuracy"])
    for table in tables.values():
        assert table.index.equals(reference.index)
        assert table["prediction"].notna().all()
    checks["all_reused_models_share_dates_and_full_coverage"] = True
    checks["original_v1_accuracy_reproduced"] = True
    checks["all_folds_labels_end_before_refit"] = all(
        pd.Timestamp(item["train_label_end_max"]) < pd.Timestamp(item["refit_date"]) for item in audit)
    assert checks["all_folds_labels_end_before_refit"]
    checks["v1_files_unchanged"] = all(sha256(PARENT / name) == value for name, value in original_hashes.items())
    assert checks["v1_files_unchanged"]
    pd.concat(tables.values()).to_csv(ROOT / "reused_history_predictions.csv")
    pd.DataFrame(audit).to_csv(ROOT / "fold_audit.csv", index=False)
    selected = tables[selected_id]
    confidence = confidence_intervals(selected, reference, config)
    selected_yearly = {str(year): score_predictions(group) for year, group in selected.groupby("year")}
    summary = {"name": config["name"], "target_horizon_days": config["forecast_horizon_days"],
               "data": {"start": str(data.index.min().date()), "end": str(data.index.max().date()),
                        "rows": len(data), "sha256": config["data_sha256"]},
               "reused_signal_start": str(selected.index.min().date()),
               "reused_signal_end": str(selected.index.max().date()),
               "reused_label_end_max": str(selected["label_end"].max().date()),
               "candidate_count": len(specs), **selection,
               "selected_development": {key: float(board.iloc[0][key]) for key in ["accuracy", "mean_year_accuracy", "worst_year_accuracy", "brier"]},
               "reused_history_metrics": reused_metrics, "selected_yearly": selected_yearly,
               "selected_confidence_intervals": confidence,
               "selected_diagnostics": selected_diagnostics(selected, config),
               "primary_goal_achieved": reused_metrics[selected_id]["accuracy"] > config["required_accuracy_strictly_greater_than"] and reused_metrics[selected_id]["coverage"] == config["required_coverage"],
               "every_reused_year_above_90pct": all(item["accuracy"] > 0.9 for item in selected_yearly.values()),
               "all_accuracy_ci_lower_bounds_above_90pct": all(item["accuracy_ci95"][0] > 0.9 for item in confidence),
               "checks": checks, "script_sha256": sha256(Path(__file__)),
               "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}}
    write_json(ROOT / "summary.json", summary)
    (ROOT / "report.txt").write_text(render_report(summary))
    print(json.dumps({"selected": selected_id, "reused_history_metrics": reused_metrics,
                      "selected_yearly": selected_yearly, "selected_confidence_intervals": confidence,
                      "primary_goal_achieved": summary["primary_goal_achieved"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
