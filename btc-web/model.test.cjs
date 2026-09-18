const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const model = require('./model.js');
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'data.js'), 'utf8'), context);
const data = JSON.parse(JSON.stringify(context.window.BTC_DATA));
const csv = rows => 'date,open,high,low,close,volume\n' + rows.map(row => row.join(',')).join('\n');
const example = data.candles.slice(-240);
const today = model.addDays(data.candles.at(-1)[0], 2);

const invalidCases = [
  ['empty CSV', '', /文件为空/],
  ['missing column', 'date,close\n2026-01-01,100', /缺少必需字段/],
  ['too few candles', csv(example.slice(0, 100)), /至少需要 201/],
  ['duplicate date', csv([...example, example.at(-1)]), /重复日期/],
  ['missing day', csv(example.filter((_, index) => index !== 50)), /不连续/],
  ['invalid calendar date', csv(example).replace(example[0][0], '2026-02-30'), /日期无效/],
  ['empty numeric field', csv(example).replace(String(example[0][1]) + ',', ','), /非数值或空值/],
  ['wrong OHLC', csv(example.map((row, index) => index ? row : [row[0], row[1], 1, row[3], row[4], row[5]])), /不一致/],
  ['negative volume', csv(example.map((row, index) => index ? row : [...row.slice(0, 5), -1])), /成交量不能为负数/],
  ['unclosed quote', 'date,open,high,low,close,volume\n"broken', /未闭合/],
  ['duplicate header', 'date,open,high,low,close,close\n2026-01-01,1,1,1,1,1', /重复字段/],
  ['invalid quote trailing text', 'date,open,high,low,close,volume\n"2026-01-01"BAD,1,1,1,1,1', /引号后/],
  ['unequal columns', 'date,open,high,low,close,volume\n2026-01-01,1,1,1,1', /列数/],
  ['infinity', csv(example).replace(String(example[0][1]) + ',', '1e999,'), /价格必须为正数/],
  ['nondecimal numeric literal', csv(example).replace(String(example[0][1]) + ',', '0xFF,'), /非数值/],
  ['before calibration cutoff', csv(data.candles.slice(1000, 1240)), /冻结概率模型/],
];
for (const [name, input, message] of invalidCases) {
  test('rejects ' + name, () => assert.throws(() => model.parseCSV(input, today), message));
}

test('parses and sorts valid quoted CSV with CRLF and BOM', () => {
  const input = '\uFEFFdate,open,high,low,close,volume\r\n' + [...example].reverse().map(row => row.map(value => '"' + value + '"').join(',')).join('\r\n');
  assert.deepEqual(model.parseCSV(input, today), example);
});
test('allows additional columns without interpreting text', () => {
  const input = 'date,open,high,low,close,volume,notes\n' + example.map(row => row.join(',') + ',"not a script, just text"').join('\n');
  assert.deepEqual(model.parseCSV(input, today), example);
});
test('rejects today and future daily candles', () => {
  assert.throws(() => model.parseCSV(csv(example), example.at(-1)[0]), /尚未结束/);
});
test('latest indicators match Python reference', () => {
  const signals = model.indicators(data.candles);
  const signal = signals.at(-1);
  assert.ok(Math.abs(signal.sma - 70319.4397) < 1e-7);
  assert.ok(Math.abs(signal.momentum - 0.17681265816998737) < 1e-12);
  assert.ok(Math.abs(signal.volatility - 0.4885667626935294) < 1e-10);
  assert.ok(Math.abs(signal.weight - 0.8187212691153001) < 1e-10);
  assert.equal(signal.score, 2);
});
test('future candles do not change past indicators', () => {
  const all = model.indicators(data.candles);
  const prefix = model.indicators(data.candles.slice(0, 2500));
  assert.deepEqual(all.slice(0, 2500), prefix);
});
test('forecast uses frozen calibration and executable horizon', () => {
  const signals = model.indicators(data.candles);
  const forecast = model.forecast(data.candles, signals, data.candles.length - 1, 30, data.forecast);
  assert.equal(forecast.probability, 0.5495915985997666);
  assert.equal(forecast.entryDate, '2026-09-18');
  assert.equal(forecast.endDate, '2026-10-18');
  assert.equal(forecast.realized, null);
  assert.throws(() => model.forecast(data.candles, signals, 200, 30, data.forecast), /不得早于/);
  assert.throws(() => model.forecast(data.candles, signals, 2500, 1, data.forecast), /仅支持/);
});
test('all original eligible forecast labels reproduce Python counts', () => {
  const signals = model.indicators(data.candles);
  for (const horizon of [7, 30]) {
    let total = 0, correct = 0;
    data.candles.forEach((row, index) => {
      if (row[0] < '2024-01-01' || index + 2 + horizon >= data.candles.length) return;
      const forecast = model.forecast(data.candles, signals, index, horizon, data.forecast);
      total++;
      correct += Number((forecast.probability >= 0.5) === (forecast.realized > 0));
    });
    assert.equal(total, data.forecast[String(horizon)].n_evaluation_overlapping);
    assert.equal(correct / total, data.forecast[String(horizon)].direction_accuracy);
  }
});
test('constant prices produce finite zero exposure', () => {
  const rows = example.map(row => [row[0], 100, 100, 100, 100, 0]);
  const signals = model.indicators(rows);
  assert.equal(signals.at(-1).volatility, 0);
  assert.equal(signals.at(-1).weight, 0);
});
