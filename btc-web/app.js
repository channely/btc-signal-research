(() => {
  'use strict';
  const data = window.BTC_DATA;
  const model = window.BTCModel;
  const $ = id => document.getElementById(id);
  const numeric = (value, digits = 2) => new Intl.NumberFormat('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
  const percent = (value, signed = false) => (signed && value > 0 ? '+' : '') + numeric(value * 100) + '%';
  const compact = value => Math.abs(value) >= 1000 ? numeric(value / 1000, 0) + 'k' : numeric(value, 0);
  const strategyNames = { trend_model: 'BTC–TVM V1', buy_hold: '买入持有', sma200: '单一 200 日均线', vol_hold: '波动率控制持有' };
  const stateNames = ['弱趋势', '趋势分歧', '趋势共振'];
  let candles = data.candles;
  let signals = model.indicators(candles);
  let selectedIndex = candles.length - 1;
  let horizon = 30;
  let range = 365;
  let performanceMode = 'equity';
  let importedName = null;
  let view = 'overview';
  let currentForecast;
  let toastTimer;
  let lastImportTrigger;

  function text(id, value) { $(id).textContent = value; }
  function toast(message) {
    clearTimeout(toastTimer);
    text('toast', message);
    $('toast').hidden = false;
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4500);
  }
  function minimumIndex() {
    return candles.findIndex((row, index) => signals[index] && row[0] >= '2024-01-01');
  }
  function configureDate() {
    $('signal-date').min = candles[minimumIndex()][0];
    $('signal-date').max = candles.at(-1)[0];
    $('signal-date').value = candles[selectedIndex][0];
    $('previous-day').disabled = selectedIndex <= minimumIndex();
    $('next-day').disabled = selectedIndex >= candles.length - 1;
    $('latest-day').disabled = selectedIndex === candles.length - 1;
  }
  function setSelected(index) {
    selectedIndex = index;
    $('date-error').hidden = true;
    configureDate();
    renderOverview();
  }

  function renderOverview() {
    const candle = candles[selectedIndex];
    currentForecast = model.forecast(candles, signals, selectedIndex, horizon, data.forecast);
    const f = currentForecast;
    const daily = candle[4] / candles[selectedIndex - 1][4] - 1;
    text('close-price', numeric(candle[4]));
    text('day-change', (daily >= 0 ? '↗ ' : '↘ ') + percent(daily, true));
    $('day-change').classList.toggle('negative', daily < 0);
    text('price-date', candle[0] + ' · 已完成 UTC 日线 · 较前日');
    text('source-badge', importedName ? '已导入 · 本地计算' : '内置历史快照');
    const daysOld = Math.floor((Date.parse(new Date().toISOString().slice(0, 10)) - Date.parse(candles.at(-1)[0])) / 86400000);
    text('snapshot-label', '数据截至 ' + candles.at(-1)[0] + (daysOld > 2 ? ` · 距今 ${daysOld} 天，非实时` : ' · 非实时行情'));
    text('up-probability', numeric(f.probability * 100));
    text('direction-tag', f.probability >= 0.5 ? '轻微偏多' : '轻微偏空');
    $('probability-fill').style.width = `${f.probability * 100}%`;
    $('probability-track').setAttribute('aria-label', `历史校准上涨概率 ${percent(f.probability)}，并非已验证准确率`);
    text('forecast-explanation', f.score === 2
      ? '趋势与动量方向一致，但上涨概率仅略高于一半，不构成高确定性信号。'
      : f.score === 1 ? '长期趋势与中期动量出现分歧。规则保持现金，不代表未来必跌。'
        : '趋势条件未满足，规则保持现金。历史弱趋势中也常有反弹，不能把空仓解读为必跌。');
    text('prior-probability', percent(f.baselineProbability));
    text('calibration-count', numeric(f.calibrationSamples, 0) + ' 个重叠样本');
    text('forecast-dates', f.entryDate + ' → ' + f.endDate);
    const predictionMatches = f.realized !== null && (f.realized > 0) === (f.direction === 'up');
    text('realized-outcome', f.realized === null
      ? `尚无完整后验结果 · 需要 ${f.endDate} 开盘数据`
      : `后验观察：${percent(f.realized, true)} · 本次方向${predictionMatches ? '正确' : '错误'}（不用于计算信号）`);
    text('sma-value', numeric(f.sma));
    text('sma-distance', percent(f.close / f.sma - 1, true));
    text('trend-status', f.close > f.sma ? '高于均线' : '低于均线');
    $('trend-status').classList.toggle('caution', f.close <= f.sma);
    text('momentum-value', percent(f.momentum, true));
    text('momentum-direction', f.momentum > 0 ? '正向动量' : '非正向动量');
    text('momentum-status', f.momentum > 0 ? '动量为正' : '动量非正');
    $('momentum-status').classList.toggle('caution', f.momentum <= 0);
    text('volatility-value', percent(f.volatility));
    text('weight-value', percent(f.weight));
    $('allocation-fill').style.width = `${f.weight * 100}%`;
    text('cash-value', '现金 ' + percent(1 - f.weight));
    text('state-summary', stateNames[f.score] + ' · ' + f.score + '/2 项成立');
    renderPriceChart();
  }

  function chart(container, dates, series, options) {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height || !dates.length) return;
    const padding = { left: 5, right: 48, top: 18, bottom: 26 };
    const usableWidth = width - padding.left - padding.right;
    const usableHeight = height - padding.top - padding.bottom;
    const values = series.flatMap(item => item.values.filter(Number.isFinite));
    let low = Math.min(...values), high = Math.max(...values);
    const spread = high - low || Math.abs(high) * 0.1 || 1;
    low -= spread * 0.09; high += spread * 0.09;
    const x = index => padding.left + index / Math.max(1, dates.length - 1) * usableWidth;
    const y = value => padding.top + (high - value) / (high - low) * usableHeight;
    let svg = `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${container.id}-title ${container.id}-description"><title id="${container.id}-title">${options.title}</title><desc id="${container.id}-description">${dates[0]} 至 ${dates.at(-1)}。${options.description}</desc>`;
    for (let i = 0; i < 5; i++) {
      const value = low + (high - low) * i / 4;
      svg += `<line x1="${padding.left}" y1="${y(value)}" x2="${width - padding.right}" y2="${y(value)}" stroke="#e8ecdf" stroke-dasharray="3 4"/><text x="${width - padding.right + 8}" y="${y(value) + 3}">${options.axis(value)}</text>`;
    }
    const tickCount = width < 400 ? 3 : 5;
    for (let i = 0; i < tickCount; i++) {
      const index = Math.round((dates.length - 1) * i / (tickCount - 1));
      const label = dates[index].slice(0, 7);
      svg += `<text x="${x(index)}" y="${height - 3}" text-anchor="${i === 0 ? 'start' : i === tickCount - 1 ? 'end' : 'middle'}">${label}</text>`;
    }
    series.forEach((item, position) => {
      let path = '', started = false;
      item.values.forEach((value, index) => {
        if (!Number.isFinite(value)) { started = false; return; }
        path += `${started ? 'L' : 'M'}${x(index).toFixed(2)},${y(value).toFixed(2)} `;
        started = true;
      });
      if (position === 0 && options.fill && item.values.every(Number.isFinite)) {
        svg += `<defs><linearGradient id="${container.id}-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${item.color}" stop-opacity=".13"/><stop offset="100%" stop-color="${item.color}" stop-opacity=".015"/></linearGradient></defs><path d="${path}L${x(dates.length - 1)},${height - padding.bottom} L${x(0)},${height - padding.bottom} Z" fill="url(#${container.id}-fill)"/>`;
      }
      svg += `<path d="${path}" fill="none" stroke="${item.color}" stroke-width="${position === 0 ? 2.2 : 1.4}" stroke-linejoin="round" stroke-linecap="round" ${item.dashed ? 'stroke-dasharray="5 3"' : ''}/>`;
      const last = item.values.at(-1);
      if (position === 0 && Number.isFinite(last)) svg += `<circle cx="${x(dates.length - 1)}" cy="${y(last)}" r="3.5" fill="${item.color}" stroke="#fffefa" stroke-width="2"/>`;
    });
    svg += `<line class="crosshair" x1="0" x2="0" y1="${padding.top}" y2="${height - padding.bottom}" stroke="#829477" stroke-dasharray="3 3" visibility="hidden"/></svg>`;
    container.innerHTML = svg;
    const tooltip = document.createElement('div');
    tooltip.className = 'chart-tooltip';
    tooltip.hidden = true;
    container.append(tooltip);
    container.onpointermove = event => {
      const rectangle = container.getBoundingClientRect();
      const index = Math.max(0, Math.min(dates.length - 1, Math.round((event.clientX - rectangle.left - padding.left) / usableWidth * (dates.length - 1))));
      const crosshair = container.querySelector('.crosshair');
      crosshair.setAttribute('x1', x(index)); crosshair.setAttribute('x2', x(index));
      crosshair.setAttribute('visibility', 'visible');
      tooltip.hidden = false;
      tooltip.textContent = dates[index] + '\n' + series.map(item => item.name + '  ' + (Number.isFinite(item.values[index]) ? options.tooltip(item.values[index]) : '预热中')).join('\n');
      tooltip.style.left = `${Math.max(4, Math.min(x(index) + 12, width - tooltip.offsetWidth - 5))}px`;
    };
    container.onpointerleave = () => {
      tooltip.hidden = true;
      container.querySelector('.crosshair').setAttribute('visibility', 'hidden');
    };
  }

  function renderPriceChart() {
    const start = range === 'all' ? 0 : Math.max(0, selectedIndex - range + 1);
    const rows = candles.slice(start, selectedIndex + 1);
    text('chart-summary', rows[0][0] + ' — ' + rows.at(-1)[0]);
    chart($('price-chart'), rows.map(row => row[0]), [
      { name: '收盘价', color: '#3e7053', values: rows.map(row => row[4]) },
      { name: 'MA 200', color: '#bb9852', dashed: true, values: signals.slice(start, selectedIndex + 1).map(signal => signal?.sma ?? null) },
    ], { title: 'BTC 历史收盘价与 200 日均线', description: '绿色为收盘价，金色虚线为均线。不包含所选日期之后的数据。精确末值在图表上方和信号卡中。', axis: compact, tooltip: value => numeric(value) + ' USDT', fill: true });
  }

  function renderBacktest() {
    text('historical-accuracy', percent(data.forecast['30'].direction_accuracy));
    text('historical-cagr', percent(data.metrics.trend_model.cagr, true));
    text('historical-drawdown', percent(data.metrics.trend_model.max_drawdown));
    const tbody = $('performance-table');
    tbody.replaceChildren();
    for (const name of ['trend_model', 'buy_hold', 'sma200', 'vol_hold']) {
      const stats = data.metrics[name];
      const row = document.createElement('tr');
      if (name === 'trend_model') row.className = 'highlight-row';
      for (const value of [strategyNames[name], percent(stats.cagr, true), percent(stats.max_drawdown), numeric(stats.sharpe, 3), percent(stats.mean_position)]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      tbody.append(row);
    }
    $('accuracy-table').replaceChildren();
    for (const day of [7, 30]) {
      const stats = data.forecast[String(day)];
      const row = document.createElement('tr');
      for (const value of [day + ' 天', percent(stats.direction_accuracy), percent(stats.baseline_direction_accuracy), numeric(stats.brier, 4)]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      $('accuracy-table').append(row);
    }
    $('experiment-bars').replaceChildren();
    const experiments = [
      ['V1 / 原模型', data.forecast['30'].direction_accuracy],
      ['V2 / 逻辑回归赢家', data.v2.results['logistic_basic_expanding_l2_0.1'].accuracy],
      ['V2 / 开发期赢家', data.v2.results[data.v2.selected].accuracy],
    ];
    for (const [name, accuracy] of experiments) {
      const row = document.createElement('div'); row.className = 'experiment-row';
      const label = document.createElement('span'); label.textContent = name;
      const bar = document.createElement('span'); bar.className = 'bar';
      const fill = document.createElement('span'); fill.style.width = `${accuracy * 100}%`; bar.append(fill);
      const value = document.createElement('strong'); value.textContent = percent(accuracy);
      row.append(label, bar, value); $('experiment-bars').append(row);
    }
    renderPerformanceChart();
  }

  function renderPerformanceChart() {
    const names = ['trend_model', 'buy_hold', 'vol_hold'];
    const colors = ['#376c50', '#a2a896', '#8299a3'];
    const series = names.map((name, index) => {
      let peak = 1;
      return { name: strategyNames[name], color: colors[index], values: data.curves[name].map(row => {
        peak = Math.max(peak, row[1]);
        return performanceMode === 'equity' ? row[1] : (row[1] / peak - 1) * 100;
      }) };
    });
    chart($('performance-chart'), data.curves.trend_model.map(row => row[0]), series, {
      title: performanceMode === 'equity' ? '扣费后历史累计净值对比' : '每日开盘净值回撤对比',
      description: '初始净值为1，单边成本15bps。精确指标见下方表格。',
      axis: value => numeric(value, performanceMode === 'equity' ? 1 : 0) + (performanceMode === 'equity' ? '×' : '%'),
      tooltip: value => numeric(value) + (performanceMode === 'equity' ? '×' : '%'), fill: false,
    });
  }

  function renderData() {
    text('data-type', importedName ? '用户 CSV' : '内置历史快照');
    text('data-name', importedName || 'BTCUSDT · Binance');
    text('data-range', candles[0][0] + ' → ' + candles.at(-1)[0]);
    text('data-count', numeric(candles.length, 0) + ' 根');
    $('reset-data').disabled = !importedName;
  }

  function navigate() {
    const nextView = location.hash.slice(1);
    view = ['overview', 'backtest', 'method'].includes(nextView) ? nextView : 'overview';
    document.querySelectorAll('.view').forEach(section => { section.hidden = section.id !== 'view-' + view; });
    document.querySelectorAll('[data-nav]').forEach(link => {
      const active = link.dataset.nav === view;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
    });
    text('breadcrumb-page', { overview: '预测总览', backtest: '回测验证', method: '数据与方法' }[view]);
    if (view === 'overview') renderOverview();
    if (view === 'backtest') renderBacktest();
    if (view === 'method') renderData();
  }

  function download(filename, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = filename;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function downloadSample() {
    download('btc-daily-example.csv', 'date,open,high,low,close,volume\n' + data.candles.slice(-240).map(row => row.join(',')).join('\n') + '\n', 'text/csv;charset=utf-8');
    toast('已导出 240 根真实历史日线示例。');
  }
  function exportSignal() {
    const f = currentForecast;
    const stats = data.forecast[String(horizon)];
    const result = {
      model: 'BTC-TVM-1', exportedAt: new Date().toISOString(),
      dataSource: importedName || 'Bundled Binance BTCUSDT snapshot',
      dataEnd: candles.at(-1)[0], calibrationEndExclusive: '2024-01-01',
      signal: { date: f.date, close: f.close, sma200: f.sma, momentum63: f.momentum,
        annualVolatility30: f.volatility, score: f.score, unexecutedTargetWeight: f.weight },
      forecast: { horizonDays: horizon, upProbability: f.probability, entryDate: f.entryDate, endDate: f.endDate },
      retrospectiveOutcome: { forwardReturn: f.realized, usedForSignal: false },
      originalHistoricalEvaluation: { accuracy: stats.direction_accuracy, total: stats.n_evaluation_overlapping,
        correct: Math.round(stats.direction_accuracy * stats.n_evaluation_overlapping), coverage: 1,
        scope: 'Original frozen BTCUSDT history, not a backtest of imported data' },
      warnings: ['研究模型未达90%准确率，历史不保证未来', '目标仓位不等于交易指令', '导入数据不触发重新训练或回测', '预测标签存在重叠'],
    };
    download(`btc-signal-${f.date}-${horizon}d.json`, JSON.stringify(result, null, 2) + '\n', 'application/json');
    toast('当前信号与指标口径已导出为 JSON。');
  }

  function openImport(event) {
    lastImportTrigger = event.currentTarget;
    $('import-error').hidden = true;
    $('import-progress').hidden = true;
    $('csv-file').value = '';
    $('import-dialog').showModal();
  }
  async function importFile(file) {
    if (!file) return;
    $('import-error').hidden = true;
    $('import-progress').hidden = false;
    $('csv-file').disabled = true;
    try {
      if (file.size > 5 * 1024 * 1024) throw new Error('文件超过 5 MB，请缩小数据范围后重试。');
      if (!/\.csv$/i.test(file.name)) throw new Error('请选择 .csv 格式的行情文件。');
      const nextCandles = model.parseCSV(await file.text());
      const nextSignals = model.indicators(nextCandles);
      candles = nextCandles; signals = nextSignals;
      importedName = file.name;
      selectedIndex = candles.length - 1;
      configureDate(); renderOverview(); renderData();
      $('import-dialog').close();
      toast(`已载入 ${numeric(candles.length, 0)} 根日线；原始回测指标保持不变。`);
    } catch (error) {
      text('import-error', error.message);
      $('import-error').hidden = false;
    } finally {
      $('import-progress').hidden = true;
      $('csv-file').disabled = false;
      $('csv-file').value = '';
    }
  }

  $('signal-date').addEventListener('change', event => {
    const date = event.target.value;
    const index = candles.findIndex(row => row[0] === date);
    if (index < minimumIndex() || index < 0) {
      text('date-error', `请选择 ${candles[minimumIndex()][0]} 至 ${candles.at(-1)[0]} 之间的完整日线。`);
      $('date-error').hidden = false;
      $('signal-date').value = candles[selectedIndex][0];
      return;
    }
    setSelected(index);
  });
  $('previous-day').addEventListener('click', () => setSelected(selectedIndex - 1));
  $('next-day').addEventListener('click', () => setSelected(selectedIndex + 1));
  $('latest-day').addEventListener('click', () => setSelected(candles.length - 1));
  document.querySelectorAll('[data-horizon]').forEach(button => button.addEventListener('click', () => {
    horizon = Number(button.dataset.horizon);
    document.querySelectorAll('[data-horizon]').forEach(item => {
      const active = Number(item.dataset.horizon) === horizon;
      item.classList.toggle('selected', active); item.setAttribute('aria-pressed', active);
    });
    renderOverview();
  }));
  document.querySelectorAll('[data-range]').forEach(button => button.addEventListener('click', () => {
    range = button.dataset.range === 'all' ? 'all' : Number(button.dataset.range);
    document.querySelectorAll('[data-range]').forEach(item => {
      const active = item.dataset.range === String(range);
      item.classList.toggle('selected', active); item.setAttribute('aria-pressed', active);
    });
    renderPriceChart();
  }));
  document.querySelectorAll('[data-performance]').forEach(button => button.addEventListener('click', () => {
    performanceMode = button.dataset.performance;
    document.querySelectorAll('[data-performance]').forEach(item => {
      const active = item.dataset.performance === performanceMode;
      item.classList.toggle('selected', active); item.setAttribute('aria-pressed', active);
    });
    renderPerformanceChart();
  }));
  ['open-import', 'method-import'].forEach(id => $(id).addEventListener('click', openImport));
  ['download-sample', 'dialog-sample'].forEach(id => $(id).addEventListener('click', downloadSample));
  $('export-signal').addEventListener('click', exportSignal);
  $('csv-file').addEventListener('change', event => importFile(event.target.files[0]));
  $('reset-data').addEventListener('click', () => {
    candles = data.candles; signals = model.indicators(candles); importedName = null;
    selectedIndex = candles.length - 1; configureDate(); renderOverview(); renderData();
    toast('已恢复内置 BTCUSDT 历史快照。');
  });
  $('import-dialog').addEventListener('close', () => { if (lastImportTrigger) lastImportTrigger.focus(); });
  const zone = $('upload-zone');
  zone.addEventListener('dragover', event => { event.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', event => {
    event.preventDefault(); zone.classList.remove('dragging');
    if (!$('csv-file').disabled) importFile(event.dataTransfer.files[0]);
  });
  const observer = new ResizeObserver(() => {
    if (view === 'overview') renderPriceChart();
    if (view === 'backtest') renderPerformanceChart();
  });
  observer.observe($('price-chart')); observer.observe($('performance-chart'));
  window.addEventListener('hashchange', navigate);
  configureDate(); renderOverview(); renderData(); navigate();
})();
