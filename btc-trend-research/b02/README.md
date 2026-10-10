# B02: audited, append-only refresh

This pipeline appends completed Binance BTCUSDT UTC candles to a separate snapshot, preserves raw evidence and SHA-256 hashes, and reuses V3's fixed four candidates without expanding the search. It never replaces original V1/V2/V3 files, deployed data, or expiry guards. Every run needs a new output directory; errors leave `FAILED.json` and do not authorize activation.

## Run

From the repository root, with pinned `btc-trend-research/requirements.txt` installed in `.venv`:

```sh
.venv/bin/python -m unittest discover -s btc-trend-research/b02 -p 'test_*.py' -v
.venv/bin/python btc-trend-research/b02/refresh.py --output btc-trend-research/b02/runs/NEW_RUN
# Optional --as-of YYYY-MM-DD is an exclusive UTC cutoff.
# Fully offline replay of a saved source snapshot, into another new directory:
.venv/bin/python btc-trend-research/b02/refresh.py --replay btc-trend-research/b02/runs/20261010-reviewed-final --output btc-trend-research/b02/runs/REPLAY
```

Replays verify raw response hashes, exchange-time evidence, contiguous coverage, the frozen prefix and normalized OHLCV. Raw hashes detect modification relative to a manifest; they are not independent third-party signatures. The manifest records each original source URL. Only the last frozen candle is re-fetched for overlap verification, not the entire old exchange history. The staged model uses October monthly refits but remains unauthorized for production. Its V3 protocol hash identifies the unchanged source protocol; `refreshProtocolSha256` points to B02's run protocol, which records the implementation hash and original protocol hash. The acquired CSV's own hash is in the manifest and staged model.

## Chronology and interpretation

Training uses `label_end < refit_date`; selection uses `label_end < outer_year_start`. Signal-day close becomes available at the next UTC midnight. Monthly coefficients can update, but 2026's annual candidate selection cannot change from 2026 outcomes. Scaling is fitted only on training rows; future price mutation, test-label mutation and truncation checks protect chronology. Existing stored matured predictions must reproduce within numerical tolerance.

The report separately scores all chronological outer folds, newly matured outcomes, and genuinely new signal dates after September 16. The latter are reconstructed now, not prospectively timestamped forecasts. Outputs include all candidates, baselines, yearly scores, all nonoverlapping offsets, block bootstrap intervals, selections, matured fold audits and current forecast-fold audits. Empty new 30-day samples remain empty, never substituted with older newly matured signals.

No trading strategy is tested. There are no fee/slippage/funding/turnover assumptions or net-return results. Probability scores and classification accuracy are not after-cost profitability.

## Release and evidence gates

1. Data integrity, frozen artifact preservation and leakage checks must pass.
2. Keep V3's unchanged gate: both horizons must improve accuracy and Brier versus the historical prior, paired 95% lower bounds strictly positive for all 30/60/90-day blocks, and no outer year with worse Brier.
3. Forward evidence must be collected and evaluated under a separately frozen protocol before any validated-edge claim. This refresh does not invent a favorable sample-size or acceptance threshold after seeing results.
4. An explicit review/release decision is required before moving a staged artifact into web data or changing validity dates. This script never performs that action. A failure continues to leave the existing production model expired.

The reviewed October 10 run fails the predictive evidence gate. Both primary candidates remain `historical_prior`; keep the app's expired V3 state unchanged.
