const assert = require('node:assert/strict');
const test = require('node:test');
const market = require('./market.js');
const DAY = 86400000;
const TODAY = Date.parse('2026-09-19T00:00:00Z');
const NOW = TODAY + 12 * 60 * 60 * 1000;

function row(time, price = 70000) {
  return [time, String(price), String(price + 100), String(price - 100), String(price + 20), '123.45', time + DAY - 1, '0', 1, '0', '0', '0'];
}
function history(count = 1200) {
  return Array.from({ length: count }, (_, index) => row(TODAY - (count - index) * DAY));
}
function service({ candles = history(), quote = { symbol: 'BTCUSDT', price: '80913.48' }, now = NOW } = {}) {
  const calls = [];
  const fetchImpl = async (input, options) => {
    const url = new URL(input);
    calls.push({ url, options });
    assert.equal(url.origin, market.BASE_URL);
    assert.equal(options.credentials, 'omit');
    assert.equal(options.method, 'GET');
    assert.equal(options.mode, 'cors');
    assert.equal(options.cache, 'no-store');
    let payload;
    if (url.pathname === '/api/v3/time') payload = { serverTime: now };
    else if (url.pathname === '/api/v3/ticker/price') payload = quote;
    else {
      assert.equal(url.pathname, '/api/v3/klines');
      assert.equal(url.searchParams.get('symbol'), 'BTCUSDT');
      assert.equal(url.searchParams.get('interval'), '1d');
      assert.equal(url.searchParams.get('timeZone'), '0');
      payload = candles.filter(item => item[0] <= Number(url.searchParams.get('endTime'))).slice(-Number(url.searchParams.get('limit')));
    }
    return { ok: true, json: async () => payload };
  };
  return { fetchImpl, calls };
}

test('uses exchange time and returns one source with 1000 fully closed UTC candles', async () => {
  const mock = service();
  const result = await market.refresh({ fetchImpl: mock.fetchImpl });
  assert.equal(result.candles.length, 1000);
  assert.equal(result.lastClosedDate, '2026-09-18');
  assert.equal(result.referenceTime, NOW);
  assert.equal(result.clockSource, 'exchange');
  assert.equal(result.currency, 'USDT');
  assert.equal(result.source, 'Binance BTC/USDT');
  assert.deepEqual(result.candles.at(-1), ['2026-09-18', 70000, 70100, 69900, 70020, 123.45]);
  assert.equal(result.quote.price, 80913.48);
  assert.equal(result.quoteError, null);
  assert.ok(Number.isSafeInteger(result.quote.receivedAt));
  assert.equal(mock.calls[0].url.pathname, '/api/v3/time');
  assert.equal(mock.calls.length, 3);
});

test('paginates without gaps, overlap, or a change of instrument', async () => {
  const mock = service();
  const result = await market.refresh({ fetchImpl: mock.fetchImpl, now: NOW, days: 1100 });
  assert.equal(result.candles.length, 1100);
  const calls = mock.calls.filter(call => call.url.pathname.endsWith('klines'));
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url.searchParams.get('limit'), '1000');
  assert.equal(calls[1].url.searchParams.get('limit'), '100');
  assert.equal(Number(calls[1].url.searchParams.get('endTime')), TODAY - 1000 * DAY - 1);
  for (let index = 1; index < result.candles.length; index++) {
    assert.equal(Date.parse(result.candles[index][0]) - Date.parse(result.candles[index - 1][0]), DAY);
  }
});

test('filters the current UTC candle until the exact following midnight', () => {
  assert.equal(market.parseKlines([row(TODAY - DAY), row(TODAY)], NOW).length, 1);
  assert.equal(market.parseKlines([row(TODAY)], TODAY + DAY - 1).length, 0);
  assert.equal(market.parseKlines([row(TODAY)], TODAY + DAY).length, 1);
});

const invalidRows = [
  ['out-of-order dates', [row(TODAY - DAY), row(TODAY - 2 * DAY)], /乱序/],
  ['duplicate dates', [row(TODAY - DAY), row(TODAY - DAY)], /重复/],
  ['missing dates', [row(TODAY - 3 * DAY), row(TODAY - DAY)], /缺口/],
  ['future candle', [row(TODAY + DAY)], /未来日期/],
  ['non-UTC daily opening', [row(TODAY - DAY + 1)], /UTC 日线/],
  ['negative volume', [[TODAY - DAY, '1', '2', '1', '1', '-1', TODAY - 1]], /价格或成交量/],
  ['inconsistent OHLC', [[TODAY - DAY, '3', '2', '1', '1', '0', TODAY - 1]], /不一致/],
  ['missing number', [[TODAY - DAY, '', '2', '1', '1', '0', TODAY - 1]], /价格或成交量/],
  ['nondecimal number', [[TODAY - DAY, '0x10', '20', '1', '1', '0', TODAY - 1]], /价格或成交量/],
  ['incorrect close timestamp', [[TODAY - DAY, '1', '2', '1', '1', '0', TODAY]], /UTC 日线/],
  ['short row', [[TODAY - DAY]], /格式不完整/],
  ['empty response', [], /没有返回日线/],
  ['nonarray response', { code: -1121 }, /没有返回日线/]
];
for (const [name, rows, pattern] of invalidRows) {
  test('rejects ' + name, () => assert.throws(() => market.parseKlines(rows, NOW), pattern));
}

test('does not silently accept fewer candles or stale daily data', async () => {
  const short = service({ candles: history(100) });
  await assert.rejects(market.refresh({ fetchImpl: short.fetchImpl, now: NOW, days: 201 }), /没有返回日线|不足/);
  const stale = service({ candles: history().slice(0, -1) });
  await assert.rejects(market.refresh({ fetchImpl: stale.fetchImpl, now: NOW, days: 201 }), /上一根完整 UTC 日线/);
});

test('rejects malformed cross-page continuity', async () => {
  const mock = service();
  const fetchImpl = async (input, options) => {
    const url = new URL(input);
    const response = await mock.fetchImpl(input, options);
    if (url.pathname.endsWith('klines') && url.searchParams.get('limit') === '100') {
      const rows = await response.json();
      return { ok: true, json: async () => rows.map(item => row(item[0] - DAY)) };
    }
    return response;
  };
  await assert.rejects(market.refresh({ fetchImpl, now: NOW, days: 1100 }), /分页日线/);
});

test('quote failures degrade independently while complete candles remain usable', async () => {
  for (const quote of [{ symbol: 'BTCUSD', price: '500' }, { symbol: 'BTCUSDT', price: 'NaN' }, null]) {
    const mock = service({ quote });
    const result = await market.refresh({ fetchImpl: mock.fetchImpl, now: NOW, days: 201 });
    assert.equal(result.candles.length, 201);
    assert.equal(result.quote, null);
    assert.match(result.quoteError, /最新报价/);
  }
});

test('invalid quote values and pairs are never interpreted as valid BTC/USDT quotes', () => {
  for (const price of ['', '0', '-1', 'Infinity', '0x10', null, true, [], {}]) {
    assert.throws(() => market.parseQuote({ symbol: 'BTCUSDT', price }), /价格无效/);
  }
  assert.throws(() => market.parseQuote({ symbol: 'BTCUSD', price: '100' }), /交易对无效/);
});

test('network and HTTP errors are readable and leave supplied datasets untouched', async () => {
  const previous = history(201);
  const original = JSON.stringify(previous);
  await assert.rejects(market.refresh({ now: NOW, fetchImpl: async () => { throw new TypeError('Failed to fetch'); } }), error => error.code === 'NETWORK' && /原有数据已保留/.test(error.message));
  for (const [status, code] of [[429, 'RATE_LIMIT'], [451, 'UNAVAILABLE'], [500, 'HTTP_ERROR']]) {
    await assert.rejects(market.refresh({ now: NOW, fetchImpl: async () => ({ ok: false, status }) }), error => error.code === code);
  }
  assert.equal(JSON.stringify(previous), original);
});

test('rejects unreadable JSON and an invalid exchange clock', async () => {
  await assert.rejects(market.refresh({ fetchImpl: async () => ({ ok: true, json: async () => { throw new SyntaxError(); } }) }), error => error.code === 'INVALID_JSON');
  for (const value of [null, {}, { serverTime: '1789750000000' }, { serverTime: NaN }]) {
    await assert.rejects(market.refresh({ fetchImpl: async () => ({ ok: true, json: async () => value }) }), /参考时间无效/);
  }
});

test('cancels before any network request when already aborted', async () => {
  const controller = new AbortController();
  controller.abort();
  let calls = 0;
  await assert.rejects(market.refresh({ now: NOW, signal: controller.signal, fetchImpl: async () => { calls++; } }), error => error.code === 'ABORTED');
  assert.equal(calls, 0);
});

test('cancels in-flight requests even when a fetch implementation ignores its signal', async () => {
  const controller = new AbortController();
  const pending = market.refresh({ now: NOW, signal: controller.signal, fetchImpl: () => new Promise(() => {}) });
  controller.abort();
  await assert.rejects(pending, error => error.code === 'ABORTED');
});

test('times out even when a fetch implementation ignores its signal', async () => {
  await assert.rejects(market.refresh({ now: NOW, timeoutMs: 10, fetchImpl: () => new Promise(() => {}) }), error => error.code === 'TIMEOUT');
});

test('a ticker timeout does not discard valid daily candles', async () => {
  const mock = service();
  const result = await market.refresh({ now: NOW, days: 201, timeoutMs: 10, fetchImpl: (input, options) => input.includes('/ticker/') ? new Promise(() => {}) : mock.fetchImpl(input, options) });
  assert.equal(result.candles.length, 201);
  assert.equal(result.quote, null);
  assert.match(result.quoteError, /超时/);
});

test('rejects unsupported request sizes and timeout settings', async () => {
  for (const options of [{ days: 200 }, { days: 5001 }, { days: 201.5 }, { timeoutMs: 0 }]) {
    await assert.rejects(market.refresh({ ...options, fetchImpl: async () => assert.fail('must not fetch') }), error => error.code === 'INVALID_OPTIONS');
  }
});
