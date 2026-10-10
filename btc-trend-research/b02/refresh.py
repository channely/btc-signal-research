"""Append-only, staged V3 refresh. Never edits frozen research or deployed web data."""
import argparse
import hashlib
import importlib.util
import json
import platform
import re
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
spec = importlib.util.spec_from_file_location('v3_audit', RESEARCH / 'v3/audit.py')
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
DAY_MS = 86400000
BASE = 'https://data-api.binance.vision/api/v3'
COLUMNS = ['open', 'high', 'low', 'close', 'volume']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'BTC-Audited-Refresh/1.0'}), timeout=30) as response:
        return response.read()


def validate(data, cutoff):
    if data.index.tz is not None or not data.index.equals(data.index.normalize()):
        raise ValueError('Candles must be timezone-naive UTC midnight dates')
    if data.empty or list(data.columns) != COLUMNS or not data.index.equals(pd.date_range(data.index.min(), data.index.max(), freq='D')):
        raise ValueError('Empty, duplicate, unordered, non-UTC or discontinuous candles')
    if data.index.max() >= cutoff:
        raise ValueError('Incomplete candle at acquisition cutoff')
    values = data.to_numpy()
    prices = data[COLUMNS[:4]]
    if not np.isfinite(values).all() or (prices <= 0).any().any() or (data.volume < 0).any() or (data.high < prices.max(axis=1)).any() or (data.low > prices.min(axis=1)).any():
        raise ValueError('Invalid OHLCV')


def decode_batch(raw, start, stop):
    batch = json.loads(raw)
    if not isinstance(batch, list) or not batch:
        raise ValueError('Missing Binance candles')
    records, dates = [], []
    expected = start
    for row in batch:
        if not isinstance(row, list) or len(row) != 12 or row[0] != expected or row[6] != expected + DAY_MS - 1 or row[6] >= stop:
            raise ValueError('Invalid candle timestamps, duplicate, gap or incomplete candle')
        records.append([float(x) for x in row[1:6]])
        dates.append(pd.Timestamp(expected, unit='ms'))
        expected += DAY_MS
    return pd.DataFrame(records, columns=COLUMNS, index=pd.DatetimeIndex(dates, name='date'))


def acquire(out, frozen, as_of):
    rawdir = out / 'raw'
    rawdir.mkdir()
    raw = read_json(BASE + '/time')
    (rawdir / 'time.json').write_bytes(raw)
    server_ms = json.loads(raw)['serverTime']
    local_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Fail closed if the exchange clock or requested day is ahead of the local clock.
    if not isinstance(server_ms, int) or server_ms > local_ms + 300000:
        raise ValueError('Exchange clock ahead of local clock')
    cutoff = pd.Timestamp(server_ms, unit='ms').normalize()
    if as_of:
        requested = pd.Timestamp(as_of)
        if requested != requested.normalize() or requested.tzinfo is not None or requested > cutoff:
            raise ValueError('as-of must be a completed-data UTC midnight, not in the future')
        cutoff = requested
    # Re-fetch the last frozen candle too, to verify continuity against the source.
    start = int(frozen.index.max().timestamp() * 1000)
    stop = int(cutoff.timestamp() * 1000)
    if stop <= start + DAY_MS:
        raise ValueError('No newly completed daily candles available')
    pieces, requests = [], []
    while start < stop:
        url = BASE + '/klines?' + urllib.parse.urlencode({'symbol': 'BTCUSDT', 'interval': '1d', 'startTime': start, 'endTime': stop - 1, 'limit': 1000})
        body = read_json(url)
        name = f'klines-{len(requests):03d}.json'
        (rawdir / name).write_bytes(body)
        batch = decode_batch(body, start, stop)
        pieces.append(batch)
        requests.append({'url': url, 'raw_file': 'raw/' + name, 'sha256': hashlib.sha256(body).hexdigest()})
        start = int(batch.index.max().timestamp() * 1000) + DAY_MS
    fresh = pd.concat(pieces)
    validate(fresh, cutoff)
    if fresh.index.max() != cutoff - pd.Timedelta(days=1):
        raise ValueError('Source does not reach requested final closed day')
    if not np.allclose(fresh.iloc[0].to_numpy(), frozen.iloc[-1].to_numpy(), rtol=0, atol=1e-8):
        raise ValueError('Source overlap differs from frozen candle; investigate without rewriting history')
    combined = pd.concat([frozen, fresh.iloc[1:]])
    validate(combined, cutoff)
    combined.to_csv(out / 'btc_daily.csv', float_format='%.10f')
    manifest = {'source': BASE + '/klines', 'symbol': 'BTCUSDT', 'interval': '1d', 'timezone': 'UTC',
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'server_time_ms': server_ms,
                'cutoff_exclusive': str(cutoff.date()), 'requests': requests,
                'server_time_raw_sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(combined),
                'first_closed_candle': str(combined.index.min().date()), 'last_closed_candle': str(combined.index.max().date()),
                'frozen_prefix_sha256': v3.sha256(RESEARCH / 'btc_daily.csv'), 'sha256': v3.sha256(out / 'btc_daily.csv'),
                'overlap_candles_verified': 1}
    dump(out / 'manifest.json', manifest)
    return combined, manifest


def replay(source, frozen):
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['sha256'] != v3.sha256(source / 'btc_daily.csv') or manifest['frozen_prefix_sha256'] != v3.sha256(RESEARCH / 'btc_daily.csv'):
        raise ValueError('Snapshot or frozen-prefix hash mismatch')
    time_raw = (source / 'raw/time.json').read_bytes()
    if hashlib.sha256(time_raw).hexdigest() != manifest['server_time_raw_sha256'] or json.loads(time_raw)['serverTime'] != manifest['server_time_ms']:
        raise ValueError('Exchange time evidence mismatch')
    cutoff = pd.Timestamp(manifest['cutoff_exclusive'])
    if cutoff > pd.Timestamp(manifest['server_time_ms'], unit='ms').normalize():
        raise ValueError('Snapshot exceeds exchange clock')
    data = pd.read_csv(source / 'btc_daily.csv', index_col='date', parse_dates=['date'])
    validate(data, cutoff)
    if data.index.min() != frozen.index.min() or len(data) != manifest['rows'] or str(data.index.min().date()) != manifest['first_closed_candle'] or str(data.index.max().date()) != manifest['last_closed_candle']:
        raise ValueError('Manifest extent or frozen start mismatch')
    pd.testing.assert_frame_equal(data.loc[frozen.index], frozen)
    start = int(frozen.index.max().timestamp() * 1000)
    batches = []
    for request in manifest['requests']:
        path = Path(request['raw_file'])
        if not re.fullmatch(r'raw/klines-[0-9]{3,}\.json', request['raw_file']) or not (source / path).resolve().is_relative_to((source / 'raw').resolve()):
            raise ValueError('Invalid raw-file path')
        raw = (source / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != request['sha256']:
            raise ValueError('Raw response hash mismatch')
        batch = decode_batch(raw, start, int(cutoff.timestamp() * 1000))
        batches.append(batch)
        start = int(batch.index.max().timestamp() * 1000) + DAY_MS
    if not batches:
        raise ValueError('Missing raw candle evidence')
    fresh = pd.concat(batches)
    if not fresh.index.equals(data.loc[frozen.index.max():].index):
        raise ValueError('Raw candles must cover the complete appended interval')
    np.testing.assert_allclose(data.loc[fresh.index].to_numpy(), fresh.to_numpy(), rtol=0, atol=1e-8)
    if data.index.max() != cutoff - pd.Timedelta(days=1):
        raise ValueError('Incomplete requested snapshot')
    return data, manifest


def evaluate(data, config, frozen, out):
    checks = v3.run_checks(data, config)
    summaries, predictions, folds, selections = {}, [], [], []
    names = ['nested_primary', *config['selectable_candidates'], *config['diagnostic_baselines']]
    old_predictions = pd.read_csv(RESEARCH / 'v3/predictions.csv', index_col='signal_date', parse_dates=['signal_date'])
    for h in config['horizons_days']:
        history, audit = v3.monthly_predictions(v3.make_dataset(data, h, config), h, config)
        primary, selected = v3.nested_predictions(history, config)
        old = old_predictions.loc[old_predictions.horizon == h].dropna(subset=['nested_primary'])
        np.testing.assert_allclose(primary.loc[old.index, names].to_numpy(dtype=float), old[names].to_numpy(dtype=float), atol=1e-10, rtol=1e-10)
        checks[f'h{h}_frozen_predictions_preserved'] = True
        yearly = {str(year): {name: v3.metrics(group, name) for name in names} for year, group in primary.groupby(primary.index.year)}
        uncertainty = v3.paired_uncertainty(primary, config)
        evidence = all(row['accuracy_improvement_vs_prior_ci95'][0] > 0 and row['brier_improvement_vs_prior_ci95'][0] > 0 for row in uncertainty) and all(row['nested_primary']['brier'] <= row['historical_prior']['brier'] for row in yearly.values())
        # Report truly new signal dates separately; the newly matured cohort includes both old and new signals.
        new = primary.loc[primary.index > frozen.index.max()]
        matured = primary.loc[primary.label_end > frozen.index.max()]
        summaries[str(h)] = {'start': str(primary.index.min().date()), 'end': str(primary.index.max().date()),
                             'metrics': {name: v3.metrics(primary, name) for name in names}, 'yearly': yearly,
                             'uncertainty': uncertainty, 'diagnostics': v3.diagnostics(primary, h, config),
                             'new_signal_dates': {'start': str(new.index.min().date()) if len(new) else None, 'end': str(new.index.max().date()) if len(new) else None, 'metrics': {name: v3.metrics(new, name) for name in names}},
                             'newly_matured_labels': {'metrics': {name: v3.metrics(matured, name) for name in names}},
                             'evidence_over_prior': evidence}
        primary['horizon'] = h
        predictions.append(primary)
        folds.extend(audit)
        selections.extend([dict(row, horizon=h) for row in selected])
    checks['training_labels_strictly_before_refit'] = all(r['train_label_end_max'] < r['refit_date'] for r in folds)
    checks['selection_labels_strictly_before_year'] = all(r['validation_label_end_max'] < r['selection_date'] for r in selections)
    if not all(checks.values()):
        raise ValueError('Chronology audit failed')
    summary = {'status': 'staged_retrospective_refresh_not_activated', 'horizons': summaries, 'checks': checks,
               'replacement_evidence_gate_passed': all(r['evidence_over_prior'] for r in summaries.values()),
               'activation_authorized': False,
               'cost_assumptions': 'Direction-probability research only. No positions, fills, turnover, fees, slippage, funding or net-return simulation. Accuracy is not after-cost profitability.',
               'limitations': ['Historical data previously inspected; chronological out-of-sample is not pristine or prospectively logged.', 'New signal dates are reconstructed after the fact, not forecasts timestamped before outcomes.', 'Labels overlap; all nonoverlapping offsets reported without choosing one.', 'One-exchange BTCUSDT candles; one overlap candle checked, not a full historical source re-audit.'],
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}}
    staged_config = dict(config, data_sha256=v3.sha256(out / 'btc_daily.csv') if (out / 'btc_daily.csv').exists() else hashlib.sha256(data.to_csv(float_format='%.10f').encode()).hexdigest())
    staged_model = v3.web_export(data, staged_config, summary)
    staged_model['activationAuthorized'] = False
    staged_model['refreshProtocolSha256'] = v3.sha256(out / 'protocol.json')
    dump(out / 'staged_web_model.json', staged_model)
    live_folds = []
    for h in config['horizons_days']:
        _, live_audit = v3.monthly_predictions(v3.make_dataset(data, h, config, include_unmatured=True), h, config)
        live_folds.extend(live_audit)
    checks['all_forecast_labels_strictly_before_refit'] = all(r['train_label_end_max'] < r['refit_date'] for r in live_folds)
    checks['staged_parameter_prediction_parity'] = True  # asserted inside V3 web_export
    if not all(checks.values()):
        raise ValueError('Forecast audit failed')
    pd.DataFrame(live_folds).to_csv(out / 'forecast_fold_audit.csv', index=False)
    dump(out / 'summary.json', summary)
    dump(out / 'selection.json', selections)
    pd.concat(predictions).to_csv(out / 'predictions.csv', float_format='%.12g')
    pd.DataFrame(folds).to_csv(out / 'fold_audit.csv', index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New, nonexistent output directory')
    parser.add_argument('--as-of', help='UTC cutoff exclusive, YYYY-MM-DD; defaults to exchange date')
    parser.add_argument('--replay', type=Path, help='Verify existing raw snapshot offline and rerun fixed evaluation')
    args = parser.parse_args()
    config, frozen = v3.load_inputs()
    # Never permit output nested in tracked frozen source directories.
    out = args.output.resolve()
    allowed = ROOT / 'runs'
    if not out.is_relative_to(allowed.resolve()) or out == allowed.resolve():
        parser.error('output must be a new subdirectory of btc-trend-research/b02/runs')
    out.mkdir(parents=True, exist_ok=False)
    protected = {p: v3.sha256(p) for p in RESEARCH.rglob('*') if p.is_file() and ROOT not in p.parents and '__pycache__' not in p.parts}
    protocol = {'name': 'B02 append-only V3 refresh', 'v3_protocol_sha256': v3.sha256(RESEARCH / 'v3/protocol.json'),
                'refresh_script_sha256': v3.sha256(Path(__file__)), 'candidate_selection': 'Unchanged V3 four candidates, monthly purged refits, annual past-only selection',
                'new_signal_start_exclusive': str(frozen.index.max().date()), 'activation': 'No automatic publication or expiry update; requires reviewed audit, original evidence gate, forward evidence and explicit release approval.'}
    dump(out / 'protocol.json', protocol)  # Freeze before downloading or observing outcomes.
    try:
        data, manifest = replay(args.replay, frozen) if args.replay else acquire(out, frozen, args.as_of)
        if args.replay:
            shutil.copy2(args.replay / 'btc_daily.csv', out / 'btc_daily.csv')
            shutil.copy2(args.replay / 'manifest.json', out / 'manifest.json')
            (out / 'raw').mkdir()
            shutil.copy2(args.replay / 'raw/time.json', out / 'raw/time.json')
            for request in manifest['requests']:
                destination = out / request['raw_file']
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.replay / request['raw_file'], destination)
        summary = evaluate(data, config, frozen, out)
        if not all(v3.sha256(p) == expected for p, expected in protected.items()):
            raise ValueError('Protected frozen files changed')
        summary['checks']['frozen_source_files_unchanged'] = True
        summary['data'] = manifest
        summary['protocol'] = protocol
        dump(out / 'summary.json', summary)
        print(json.dumps({'output': str(out), 'last_closed_candle': manifest['last_closed_candle'], 'evidence_gate': summary['replacement_evidence_gate_passed'], 'checks': summary['checks']}, indent=2))
    except Exception as error:
        dump(out / 'FAILED.json', {'error': str(error), 'activation_authorized': False})
        raise


if __name__ == '__main__':
    main()
