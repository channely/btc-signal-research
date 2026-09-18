const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const model = require('./model.js');
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'data.js'), 'utf8'), context);
const data = JSON.parse(JSON.stringify(context.window.BTC_DATA));

test('historical research reproduces every Python primary probability and close label', () => {
  for (const horizon of [7, 30]) {
    const expected = data.v3.horizons[String(horizon)];
    let correct = 0, total = 0;
    for (const saved of expected.oos) {
      const index = data.candles.findIndex(row => row[0] === saved[0]);
      const forecast = model.researchForecast(data.candles, index, horizon, data.v3, { allowHistorical: true });
      assert.equal(forecast.probability, saved[1]);
      assert.equal(forecast.selected, saved[7]);
      assert.equal(forecast.mode, 'historical');
      assert.equal(forecast.endDate, model.addDays(saved[0], horizon));
      if (saved[6] !== null) {
        assert.ok(Math.abs(forecast.realized - saved[6]) < 1e-12);
        total++;
        correct += Number((forecast.probability >= 0.5) === (forecast.realized > 0));
      }
    }
    assert.equal(total, expected.metrics.nested_primary.total);
    assert.equal(correct, expected.metrics.nested_primary.correct);
  }
});

test('new closed candle uses frozen prior, explicit validity, and observe state', () => {
  const rows = [...data.candles, ['2026-09-17', 76000, 79000, 75000, 77500, 1000]];
  const forecast = model.researchForecast(rows, rows.length - 1, 30, data.v3);
  assert.equal(forecast.available, true);
  assert.equal(forecast.mode, 'frozen');
  assert.equal(forecast.probability, data.v3.horizons['30'].priorProbability);
  assert.equal(forecast.endDate, '2026-10-17');
  assert.equal(forecast.signal, 'observe');
  assert.equal(forecast.evidencePassed, false);
});

test('expired model and earlier CSV dates never use current parameters', () => {
  for (const date of ['2026-10-01', '2027-01-01', '2024-02-01', '2026-08-31']) {
    const rows = [[date, 100, 100, 100, 100, 0]];
    assert.equal(model.researchForecast(rows, 0, 7, data.v3).available, false);
  }
});

test('custom CSV does not silently inherit same-date historical predictions', () => {
  const index = data.candles.findIndex(row => row[0] === '2025-01-01');
  assert.equal(model.researchForecast(data.candles, index, 7, data.v3).available, false);
  assert.equal(model.researchForecast(data.candles, index, 7, data.v3, { allowHistorical: true }).available, true);
});

test('altering or truncating future candles never changes the historical probability', () => {
  const index = data.candles.findIndex(row => row[0] === '2025-01-01');
  const options = { allowHistorical: true };
  const full = model.researchForecast(data.candles, index, 30, data.v3, options);
  const truncated = model.researchForecast(data.candles.slice(0, index + 1), index, 30, data.v3, options);
  assert.equal(full.probability, truncated.probability);
  assert.equal(truncated.realized, null);
  const changed = data.candles.map((row, i) => i > index ? [row[0], 1, 1, 1, 1, 0] : row);
  assert.equal(full.probability, model.researchForecast(changed, index, 30, data.v3, options).probability);
});

test('unsupported candidate or invalid probability fails closed', () => {
  const artifact = JSON.parse(JSON.stringify(data.v3));
  const rows = [['2026-09-17', 100, 100, 100, 100, 0]];
  artifact.horizons['7'].selected = 'unreviewed-model';
  assert.equal(model.researchForecast(rows, 0, 7, artifact).available, false);
  artifact.horizons['7'].selected = 'historical_prior';
  artifact.horizons['7'].priorProbability = 1.5;
  assert.equal(model.researchForecast(rows, 0, 7, artifact).available, false);
  assert.throws(() => model.researchForecast(rows, 0, 14, data.v3), /仅支持/);
});
