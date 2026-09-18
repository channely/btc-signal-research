# Web 模型接口

`python3 btc-trend-research/v3/audit.py` 导出 `web_model.json`；`python3 btc-web/build_data.py` 将其打包为 `window.BTC_DATA.v3`。不修改 V1/V2 研究文件。

顶层 `asOfDate` 是数据快照最后完成日线，`validFromDate` / `validUntilExclusive` 是最新冻结参数适用的信号日期范围。该版本为 `[2026-09-01, 2026-10-01)`。届满必须提示过期并停止用它作当前预测。行情刷新不等于模型重新训练。

每个期限位于 `horizons['7']` 或 `horizons['30']`：

- `selected`：当年元旦仅依据已成熟、过去样本外预测选出的候选；本次两个期限均为 `historical_prior`。
- `refitDate` / `trainLabelEndMax`：月度重训日期与最后成熟训练标签的结束日。严格后者小于前者。
- `selectionDate` / `selectionLabelEndMax`：年度选型时间隔离记录。
- `priorProbability`：历史上涨先验；不是市场特征带来的预测优势。
- `regimeProbabilities`：得分 `0`、`1`、`2` 对应的收缩状态概率，得分为 `(close > SMA200) + (close > close[t-63])`。
- `featureNames` / `center` / `scale`：逻辑回归特征顺序及训练集均值、总体标准差。
- `logisticCoefficients`：第 0 项为截距，其余对应特征系数。标准化特征裁剪到 `±standardizedClip`。
- `logisticPriorBlendWeight`：将逻辑回归原始概率向先验收缩的权重（当前 0.5）。
- `metrics` / `uncertainty`：2022 年起复用历史的统计，不是当前概率的保证。
- `oos`：从 2022-01-01 至快照末日，每天按当时参数得到的预测。数组列顺序由顶层 `oosColumns` 给出。

特征计算：

```text
ret7  = ln(close[t] / close[t-7])
ret30 = ln(close[t] / close[t-30])
ma200 = close[t] / mean(close[t-199:t]) - 1
vol30 = sample_std(ln(close[i]/close[i-1]), i=t-29...t) * sqrt(365)
z     = clip((features - center) / scale, -8, 8)
raw   = sigmoid(coef[0] + sum(coef[i+1] * z[i]))
logistic_shrunk = 0.5 * priorProbability + 0.5 * raw
regime_shrunk   = regimeProbabilities[String(score)]
equal_blend    = 0.5 * logistic_shrunk + 0.5 * regime_shrunk
```

`oos` 每行为：

```text
[date, primary, historical_prior, regime_shrunk, logistic_shrunk,
 equal_blend, realizedReturn|null, selected]
```

所有历史概率都来自对应日期当时可得的训练标签和选型记录；最后尚未成熟的标签为 `null`。不要用最新冻结参数重算早于 `validFromDate` 的历史，也不要对不同来源或用户修改过的 CSV 盲用内置 OOS 表。新输入只可在有效日期范围内使用冻结参数，并应显式标注数据来源。

预测参考价严格是**完成 UTC 日线的收盘价**，目标是该收盘到未来 7/30 日收盘是否上涨。最新实时 ticker 可作为行情参考，但不能替换目标参考价后沿用同一概率或回测命中率。冻结模型不产生经过验证的价格区间，也不声称有交易获利能力。

V3 的历史逐年选型全部回落到先验。界面宜标为“实验性概率 / 未建立预测优势”，保留 V1 的独立历史研究口径；V1 使用延迟两天后的开盘到开盘标签，两者不能直接横比。
