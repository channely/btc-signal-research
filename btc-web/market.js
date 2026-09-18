(function (root) {
  'use strict';
  const DAY = 86400000;
  const BASE_URL = 'https://data-api.binance.vision';
  const SYMBOL = 'BTCUSDT';

  class MarketError extends Error {
    constructor(message, code) {
      super(message);
      this.name = 'MarketError';
      this.code = code;
    }
  }

  function fail(message, code = 'INVALID_DATA') { throw new MarketError(message, code); }
  function cancelled() { return new MarketError('行情刷新已取消，原有数据已保留。', 'ABORTED'); }
  function finiteNumber(value) {
    if (typeof value !== 'number' && (typeof value !== 'string' || !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value))) return NaN;
    return Number(value);
  }

  // REST timestamps are milliseconds; UTC 1d candles have a fixed closing time.
  // An incomplete daily candle is never an input to the prediction model.
  function parseKlines(rows, referenceTime) {
    if (!Array.isArray(rows) || !rows.length) fail('行情服务没有返回日线数据。');
    if (!Number.isSafeInteger(referenceTime) || referenceTime <= 0) fail('行情参考时间无效。');
    const today = Math.floor(referenceTime / DAY) * DAY;
    let previousTime = null;
    const candles = [];
    for (const row of rows) {
      if (!Array.isArray(row) || row.length < 7) fail('日线数据格式不完整。');
      const openTime = finiteNumber(row[0]);
      const closeTime = finiteNumber(row[6]);
      if (!Number.isSafeInteger(openTime) || openTime < 0 || openTime % DAY !== 0 || closeTime !== openTime + DAY - 1) {
        fail('日线时间格式无效，必须是完整的 UTC 日线区间。');
      }
      if (openTime > today) fail('行情服务返回了未来日期的日线。');
      if (previousTime !== null && openTime !== previousTime + DAY) fail('日线日期乱序、重复或存在缺口，请稍后重试。');
      previousTime = openTime;
      if (openTime + DAY > referenceTime) continue;
      const numbers = row.slice(1, 6).map(finiteNumber);
      const [open, high, low, close, volume] = numbers;
      if (!numbers.every(Number.isFinite) || Math.min(open, high, low, close) <= 0 || volume < 0) fail('日线价格或成交量无效。');
      if (high < Math.max(open, close, low) || low > Math.min(open, close, high)) fail('日线最高价、最低价与开收盘价不一致。');
      candles.push([new Date(openTime).toISOString().slice(0, 10), ...numbers]);
    }
    return candles;
  }

  function parseQuote(value, receivedAt = Date.now()) {
    if (!value || value.symbol !== SYMBOL) fail('最新报价交易对无效；仅支持 BTC/USDT。');
    const price = finiteNumber(value.price);
    if (!Number.isFinite(price) || price <= 0) fail('最新报价价格无效。');
    return { price, receivedAt };
  }

  async function requestJSON(path, params, options) {
    const { fetchImpl, signal, timeoutMs } = options;
    if (signal.aborted) throw cancelled();
    const controller = new AbortController();
    let timedOut = false;
    let rejectAbort;
    const aborted = new Promise((resolve, reject) => { rejectAbort = reject; });
    const onAbort = () => {
      controller.abort();
      rejectAbort(cancelled());
    };
    signal.addEventListener('abort', onAbort, { once: true });
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
      rejectAbort(new MarketError('行情服务响应超时，请检查网络后重试。', 'TIMEOUT'));
    }, timeoutMs);
    const url = new URL(path, BASE_URL);
    for (const [name, value] of Object.entries(params)) url.searchParams.set(name, String(value));
    try {
      return await Promise.race([
        (async () => {
          const response = await fetchImpl(url.href, {
            method: 'GET', mode: 'cors', credentials: 'omit', cache: 'no-store', redirect: 'error', signal: controller.signal
          });
          if (!response.ok) {
            if (response.status === 429 || response.status === 418) fail('行情请求过于频繁，请稍后再刷新。', 'RATE_LIMIT');
            if (response.status === 451 || response.status === 403) fail('当前网络或所在地区无法访问行情服务，可使用本地 CSV 导入。', 'UNAVAILABLE');
            fail(`行情服务暂时不可用（HTTP ${response.status}），请稍后重试。`, 'HTTP_ERROR');
          }
          try { return await response.json(); }
          catch { fail('行情服务返回了无法解析的数据。', 'INVALID_JSON'); }
        })(),
        aborted
      ]);
    } catch (error) {
      if (signal.aborted) throw cancelled();
      if (timedOut) fail('行情服务响应超时，请检查网络后重试。', 'TIMEOUT');
      if (error instanceof MarketError) throw error;
      fail('无法连接行情服务，请检查网络；原有数据已保留，也可导入本地 CSV。', 'NETWORK');
    } finally {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
    }
  }

  async function closedCandles(referenceTime, days, options) {
    const today = Math.floor(referenceTime / DAY) * DAY;
    let endTime = today - 1;
    let candles = [];
    // Each page comes from the same exchange and BTC/USDT instrument. Never
    // append these rows to a USD CSV or silently fall back to another provider.
    for (let page = 0; candles.length < days && page < Math.ceil(days / 1000) + 2; page++) {
      const limit = Math.min(1000, days - candles.length);
      const rows = await requestJSON('/api/v3/klines', { symbol: SYMBOL, interval: '1d', timeZone: '0', limit, endTime }, options);
      if (Array.isArray(rows) && rows.length > limit) fail('行情服务返回的日线数量超出请求范围。');
      const next = parseKlines(rows, referenceTime);
      if (!next.length) fail('行情服务没有返回足够的已完成 UTC 日线。');
      if (Date.parse(next.at(-1)[0] + 'T00:00:00Z') > endTime) fail('行情服务返回的日线超出请求时间范围。');
      if (candles.length && Date.parse(next.at(-1)[0] + 'T00:00:00Z') + DAY !== Date.parse(candles[0][0] + 'T00:00:00Z')) {
        fail('分页日线之间存在缺口或重复，请稍后重试。');
      }
      candles = next.concat(candles);
      endTime = Date.parse(next[0][0] + 'T00:00:00Z') - 1;
    }
    if (candles.length < days) fail(`已完成日线不足 ${days} 根，请稍后重试。`);
    const expectedLast = new Date(today - DAY).toISOString().slice(0, 10);
    if (candles.at(-1)[0] !== expectedLast) fail('行情服务尚未提供上一根完整 UTC 日线，请稍后重试。', 'STALE_DATA');
    return candles;
  }

  // This function is deliberately stateless. Callers replace their displayed
  // dataset only after it resolves; a failed refresh cannot erase an import.
  async function refresh(options = {}) {
    const fetchImpl = options.fetchImpl || (typeof root.fetch === 'function' ? root.fetch.bind(root) : null);
    if (!fetchImpl) fail('当前浏览器不支持在线行情，请使用本地 CSV 导入。', 'UNSUPPORTED');
    const days = options.days === undefined ? 1000 : options.days;
    const timeoutMs = options.timeoutMs === undefined ? 15000 : options.timeoutMs;
    if (!Number.isInteger(days) || days < 201 || days > 5000) fail('行情数量必须是 201 至 5000 之间的整数。', 'INVALID_OPTIONS');
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) fail('行情请求超时时间无效。', 'INVALID_OPTIONS');
    if (options.signal && options.signal.aborted) throw cancelled();
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    if (options.signal) options.signal.addEventListener('abort', onAbort, { once: true });
    const requestOptions = { fetchImpl, signal: controller.signal, timeoutMs };
    try {
      const clock = options.now === undefined ? await requestJSON('/api/v3/time', {}, requestOptions) : null;
      const referenceTime = options.now === undefined ? clock && clock.serverTime : options.now;
      if (!Number.isSafeInteger(referenceTime) || referenceTime <= 0) fail('行情服务返回的参考时间无效。');
      const quoteResult = requestJSON('/api/v3/ticker/price', { symbol: SYMBOL }, requestOptions)
        .then(value => ({ quote: parseQuote(value), quoteError: null }))
        .catch(error => ({ quote: null, quoteError: error.message }));
      const candles = await closedCandles(referenceTime, days, requestOptions);
      const quote = await quoteResult;
      if (controller.signal.aborted) throw cancelled();
      return {
        candles, ...quote, symbol: SYMBOL, currency: 'USDT', source: 'Binance BTC/USDT',
        fetchedAt: Date.now(), referenceTime, clockSource: options.now === undefined ? 'exchange' : 'provided',
        lastClosedDate: candles.at(-1)[0]
      };
    } finally {
      controller.abort();
      if (options.signal) options.signal.removeEventListener('abort', onAbort);
    }
  }

  const api = { refresh, parseKlines, parseQuote, BASE_URL, SYMBOL };
  root.BTCMarket = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
