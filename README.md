# 序列 SIGNAL · BTC 研究助手

[打开应用](https://channely.github.io/btc-signal-research/) · [GitHub 仓库](https://github.com/channely/btc-signal-research)

手机与桌面均可使用的静态 BTC/USDT 研究台。打开页面会读取 Binance 公开行情，也可手动刷新、导入 CSV、选择历史信号日期、切换 7／30 天期限、查看回测和导出分析。无账号、无 API 密钥、无交易执行。

## 行情与预测

- **最新报价**：本次请求获得的 BTC/USDT 价格，注明获取时间；不是持续推送的实时报价。
- **日线信号**：仅使用已完成的 UTC 日线。通过交易所时间排除未收盘数据，绝不把盘中报价伪装成完整日线。
- **V1 历史规则**：200 日均线、63 日动量、30 日波动率；概率使用 2018—2023 年冻结校准，标签是信号日后第 2 日开盘至再过 7／30 日开盘的变化。
- **冻结回测**：原始数据截至 2026-09-16；刷新和导入只改变当前行情，不重新计算历史回测数字。61.38% 是 V1 的 30 日重叠样本方向准确率，并非未来成功率或收益率。
- **V3 收盘目标实验**：从所选日线收盘到 7／30 天后收盘，固定 4 个候选、月度训练、年度按过去成熟验证结果选型。主策略回落到历史先验，未建立额外预测优势；当前显示“观望”。当前冻结参数只适用于 2026-10-01 之前的信号日期，超期不继续冒用。详见 [实验报告](btc-trend-research/v3/REPORT.md)。
- **弱信号与不确定性**：此前 V2 的 40 组实验未证明复杂模型稳定优于 V1。完整历史已被查看，所有本次历史实验均属于回溯研究。

网页请求只发往 `https://data-api.binance.vision`，仅获取公开市场数据，不发送导入文件。网络、限流或地区限制会显示失败原因，保留原有数据。CSV 和在线行情仅在本次页面会话中保存。

## 本地运行与检查

```sh
python3 -m http.server 8876 --bind 127.0.0.1 --directory btc-web
# 打开 http://127.0.0.1:8876
node --test btc-web/*.test.cjs
node --check btc-web/app.js
```

复现 V3 研究（建议在独立虚拟环境安装研究依赖）：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r btc-trend-research/requirements.txt
.venv/bin/python btc-trend-research/v3/audit.py --checks-only
.venv/bin/python btc-trend-research/v3/audit.py
python3 btc-web/build_data.py
```

CSV 必须包含 `date,open,high,low,close,volume`，日期为 UTC `YYYY-MM-DD`，至少 201 根连续完整日线。导入前应自行确认交易对和计价单位；文件最大 5 MB。示例可在“数据与方法”下载。

研究脚本位于 `btc-trend-research/`，V1/V2 协议、数据哈希、逐日预测和失败实验保留在仓库。重建网页历史数据使用 `python3 btc-web/build_data.py`。请不要为追求更漂亮的命中率反复对相同检验期选参。

## 发布

仓库现为**公开仓库**，按所有者决定使用 GitHub Free 的 GitHub Pages。`.github/workflows/pages.yml` 在推送 `main` 后执行模型与行情测试，仅打包 `btc-web` 的网页资源到 Pages。网页无需后台服务。

日常使用直接在页面刷新行情；部署不需要改写冻结历史数据或提交最新价格。Pages 成功部署不意味着行情供应商在所有地区都可访问，应用因此保留 CSV 导入和离线快照。

## 解释边界

预测概率、方向命中率和扣费收益是不同指标。重叠标签不是独立交易，降低持仓造成的回撤下降也不是预测能力提升。当前没有证据支持稳定的高胜率，更不能保证收益；模型适合作为有证据和局限说明的研究参考。

方法参考：[时间序列切分与 gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)、[回测过拟合研究](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)、[Binance 公开市场数据接口](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)。
