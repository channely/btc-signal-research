(function (root) {
  'use strict';
  const DAY = 86400000;
  const REQUIRED = ['date', 'open', 'high', 'low', 'close', 'volume'];
  const addDays = (date, count) => new Date(Date.parse(date + 'T00:00:00Z') + count * DAY).toISOString().slice(0, 10);
  const mean = values => values.reduce((total, value) => total + value, 0) / values.length;

  function parseRows(text) {
    const rows = [];
    let row = [], cell = '', quoted = false, endedQuote = false;
    text = text.replace(/^\uFEFF/, '');
    for (let index = 0; index < text.length; index++) {
      const character = text[index];
      if (quoted) {
        if (character === '"' && text[index + 1] === '"') { cell += '"'; index++; }
        else if (character === '"') { quoted = false; endedQuote = true; }
        else cell += character;
      } else if (character === ',' || character === '\n' || character === '\r') {
        row.push(cell.trim()); cell = ''; endedQuote = false;
        if (character !== ',') {
          if (row.some(value => value !== '')) rows.push(row);
          row = [];
          if (character === '\r' && text[index + 1] === '\n') index++;
        }
      } else if (character === '"' && cell === '' && !endedQuote) {
        quoted = true;
      } else if (endedQuote && !/\s/.test(character)) {
        throw new Error('CSV 引号后出现了无效字符，请检查文件格式。');
      } else {
        cell += character;
      }
    }
    if (quoted) throw new Error('CSV 存在未闭合的双引号。');
    row.push(cell.trim());
    if (row.some(value => value !== '')) rows.push(row);
    return rows;
  }

  function parseCSV(text, today = new Date().toISOString().slice(0, 10)) {
    const rows = parseRows(text);
    if (rows.length < 2) throw new Error('文件为空或只有表头，请导入完整的日线数据。');
    const header = rows.shift().map(key => key.toLowerCase());
    if (new Set(header).size !== header.length) throw new Error('CSV 表头包含重复字段。');
    const missing = REQUIRED.filter(key => !header.includes(key));
    if (missing.length) throw new Error('缺少必需字段：' + missing.join(', '));
    if (rows.length > 20000) throw new Error('最多支持 20,000 根日线，请缩小文件范围。');
    const columns = REQUIRED.map(key => header.indexOf(key));
    const candles = rows.map((row, index) => {
      if (row.length !== header.length) throw new Error(`第 ${index + 2} 行的列数与表头不一致。`);
      const date = row[columns[0]];
      const timestamp = Date.parse(date + 'T00:00:00Z');
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(timestamp) || new Date(timestamp).toISOString().slice(0, 10) !== date) {
        throw new Error(`第 ${index + 2} 行日期无效，请使用 YYYY-MM-DD。`);
      }
      if (date >= today) throw new Error(`日线 ${date} 尚未结束或位于未来；仅支持已完成的 UTC 日线。`);
      const numbers = columns.slice(1).map(column => {
        const value = row[column];
        if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value)) throw new Error(`第 ${index + 2} 行包含非数值或空值。`);
        return Number(value);
      });
      const [open, high, low, close, volume] = numbers;
      if (!numbers.every(Number.isFinite) || Math.min(open, high, low, close) <= 0 || volume < 0) throw new Error(`第 ${index + 2} 行价格必须为正数，成交量不能为负数。`);
      if (high < Math.max(open, close, low) || low > Math.min(open, close, high)) throw new Error(`第 ${index + 2} 行最高价、最低价与开收盘价不一致。`);
      return [date, ...numbers];
    }).sort((a, b) => a[0].localeCompare(b[0]));
    if (candles.length < 201) throw new Error('至少需要 201 根连续日线，才能完整计算均线和波动率。');
    for (let index = 1; index < candles.length; index++) {
      if (candles[index][0] === candles[index - 1][0]) throw new Error(`存在重复日期：${candles[index][0]}。`);
      if (addDays(candles[index - 1][0], 1) !== candles[index][0]) throw new Error(`日线不连续：${candles[index - 1][0]} 之后存在缺失日期。`);
    }
    if (candles.at(-1)[0] < '2024-01-01') throw new Error('冻结概率模型适用于 2024-01-01 起的日期，请导入覆盖该日期之后的行情。');
    return candles;
  }

  function indicators(candles) {
    return candles.map((candle, index) => {
      if (index < 199) return null;
      const closes = candles.slice(index - 199, index + 1).map(row => row[4]);
      const sma = mean(closes);
      const momentum = candle[4] / candles[index - 63][4] - 1;
      const returns = candles.slice(index - 29, index + 1).map((row, offset) => Math.log(row[4]) - Math.log(candles[index - 30 + offset][4]));
      const average = mean(returns);
      const volatility = Math.sqrt(returns.reduce((sum, value) => sum + (value - average) ** 2, 0) / 29) * Math.sqrt(365);
      const score = Number(candle[4] > sma) + Number(momentum > 0);
      return { sma, momentum, volatility, score, weight: score === 2 ? Math.min(1, 0.4 / volatility) : 0 };
    });
  }

  function forecast(candles, signals, index, horizon, tables) {
    const signal = signals[index];
    if (!signal || candles[index][0] < '2024-01-01') throw new Error('当前日期不可预测：需要完成均线预热，且信号日期不得早于 2024-01-01。');
    if (![7, 30].includes(horizon)) throw new Error('仅支持 7 天或 30 天预测期限。');
    const calibrated = tables[String(horizon)].calibration.find(item => item.score === signal.score);
    const entryDate = addDays(candles[index][0], 2);
    const endDate = addDays(entryDate, horizon);
    const entry = candles[index + 2];
    const exit = candles[index + 2 + horizon];
    const realized = entry && exit ? exit[1] / entry[1] - 1 : null;
    return { ...signal, date: candles[index][0], close: candles[index][4], horizon,
      probability: calibrated.p_up, baselineProbability: tables[String(horizon)].baseline_probability,
      calibrationSamples: calibrated.n_overlapping, entryDate, endDate, realized,
      direction: calibrated.p_up >= 0.5 ? 'up' : 'not_up',
      q10: calibrated.return_q10, median: calibrated.return_median, q90: calibrated.return_q90 };
  }

  const api = { addDays, parseCSV, indicators, forecast };
  root.BTCModel = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
