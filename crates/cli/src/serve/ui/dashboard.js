/* quench6k telemetry. Charts/history stay outside Alpine's reactive proxy. */
'use strict';
document.addEventListener('alpine:init', () => {
  Alpine.data('dashboard', () => {
    const charts = {};
    let samples = [];
    let timer = null;
    let resizeObserver = null;
    let inFlight = false;
    let controller = null;
    let disposed = false;
    let since = 0;
    let serverStart = null;
    let zoomWindow = null;
    let inspectionDomain = null;
    let renderDomain = null;
    let applyingOptions = false;
    let chartGroup = null;
    const sum = values => (values || []).reduce((a, b) => a + b, 0);
    const formatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
    return {
      server: {}, totals: {}, slots: [], queued: 0, paused: false, error: '',
      lastUpdated: 0, range: 900, zoomed: false, outputRate: 0, prefillRate: 0, hasTraffic: false,
      hiddenSlots: [], colors: ['#ff966b', '#79adff', '#65d5ab', '#c3a1ff', '#ffc579', '#6ed8de', '#ef9bc1', '#b6d887'],
      ranges: [{ label: '5m', seconds: 300 }, { label: '15m', seconds: 900 }, { label: '1h', seconds: 3600 }],
      get connectionLabel() { return this.paused ? 'Paused' : this.error ? 'Reconnecting' : this.lastUpdated ? 'Connected' : 'Connecting'; },
      get active() { return this.slots.filter(s => s.phase === 'prefill' || s.phase === 'decode').length; },
      get availableSlots() { return this.slots.filter(s => s.phase !== 'unavailable').length; },
      get failedSlots() { return this.slots.some(s => s.phase === 'error'); },
      get totalKv() { return sum(this.slots.map(s => s.kv_bytes)); },
      get cacheRate() { return this.totals.prompt_tokens ? (100 * this.totals.cached_tokens / this.totals.prompt_tokens).toFixed(1) : '0.0'; },
      get uptime() {
        if (!this.server.started_at || !this.lastUpdated) return '—';
        const seconds = Math.max(0, Math.floor((this.lastUpdated - this.server.started_at) / 1000));
        if (seconds >= 86400) return Math.floor(seconds / 86400) + 'd ' + Math.floor(seconds % 86400 / 3600) + 'h';
        if (seconds >= 3600) return Math.floor(seconds / 3600) + 'h ' + Math.floor(seconds % 3600 / 60) + 'm';
        return Math.floor(seconds / 60) + 'm ' + seconds % 60 + 's';
      },
      number(value) { return formatter.format(value || 0); },
      integer(value) { return Math.round(value || 0).toLocaleString(); },
      compact(value) { return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(value || 0); },
      bytes(value) {
        if (!value) return '0 B';
        const units = ['B', 'KiB', 'MiB', 'GiB'];
        const exponent = Math.min(3, Math.floor(Math.log(value) / Math.log(1024)));
        return this.number(value / 1024 ** exponent) + ' ' + units[exponent];
      },
      clock(value) { return new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }); },
      phaseName(phase) { return ({ idle: 'Idle', prefill: 'Prefilling', decode: 'Generating', error: 'Error', unavailable: 'Unavailable' })[phase] || phase; },
      init() {
        this.$nextTick(() => {
          if (disposed) return;
          charts.output = echarts.init(this.$refs.outputChart, null, { renderer: 'canvas' });
          charts.prefill = echarts.init(this.$refs.prefillChart, null, { renderer: 'canvas' });
          charts.activity = echarts.init(this.$refs.activityChart, null, { renderer: 'canvas' });
          chartGroup = echarts.connect(Object.values(charts));
          Object.values(charts).forEach(chart => chart.on('datazoom', () => this.captureZoom(chart)));
          resizeObserver = new ResizeObserver(() => Object.values(charts).forEach(chart => chart.resize()));
          [this.$refs.outputChart, this.$refs.prefillChart, this.$refs.activityChart].forEach(el => resizeObserver.observe(el));
          this.render();
          this.poll();
        });
      },
      destroy() {
        disposed = true;
        clearTimeout(timer);
        controller?.abort();
        resizeObserver?.disconnect();
        if (chartGroup) echarts.disconnect(chartGroup);
        Object.values(charts).forEach(chart => chart.dispose());
      },
      setRange(seconds) { this.range = seconds; this.followLive(); },
      followLive() {
        zoomWindow = null;
        inspectionDomain = null;
        this.zoomed = false;
        this.render();
      },
      captureZoom(chart) {
        if (applyingOptions || !renderDomain) return;
        const option = chart.getOption();
        const zoom = option.dataZoom[0];
        if (zoom.start <= 0.001 && zoom.end >= 99.999) {
          zoomWindow = null;
          inspectionDomain = null;
          this.zoomed = false;
          return;
        }
        const [min, max] = renderDomain;
        const from = Number.isFinite(zoom.startValue) ? zoom.startValue : min + (max - min) * zoom.start / 100;
        const to = Number.isFinite(zoom.endValue) ? zoom.endValue : min + (max - min) * zoom.end / 100;
        if (!(to > from)) return;
        if (!inspectionDomain) inspectionDomain = [...renderDomain];
        zoomWindow = [from, to];
        this.zoomed = true;
      },
      toggleSlot(id) {
        this.hiddenSlots = this.hiddenSlots.includes(id) ? this.hiddenSlots.filter(i => i !== id) : [...this.hiddenSlots, id];
        this.render();
      },
      togglePause() {
        this.paused = !this.paused;
        if (!this.paused) this.retry();
      },
      retry() { clearTimeout(timer); this.poll(); },
      async poll() {
        if (disposed || inFlight) return;
        clearTimeout(timer);
        if (this.paused || document.hidden) {
          timer = setTimeout(() => this.poll(), 2000);
          return;
        }
        inFlight = true;
        controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10000);
        try {
          const response = await fetch('/stats' + (since ? '?since=' + since : ''), { cache: 'no-store', signal: controller.signal });
          if (!response.ok) throw new Error('Server returned HTTP ' + response.status + '.');
          const data = await response.json();
          if (disposed || this.paused) return;
          if (serverStart !== null && serverStart !== data.server.started_at) {
            // Incremental requests after a restart can contain only the latest samples. Re-bootstrap.
            samples = [];
            since = 0;
            zoomWindow = null;
            inspectionDomain = null;
            this.zoomed = false;
          }
          const restarted = serverStart !== null && serverStart !== data.server.started_at;
          serverStart = data.server.started_at;
          this.server = data.server;
          this.totals = data.totals;
          this.slots = data.current.slots;
          this.queued = data.current.queued;
          this.lastUpdated = data.current.timestamp;
          this.error = '';
          const cutoff = this.lastUpdated - this.server.history_seconds * 1000;
          samples = samples.filter(s => s.timestamp >= cutoff);
          for (const sample of data.samples) {
            if (!samples.length || sample.timestamp > samples[samples.length - 1].timestamp) samples.push(sample);
          }
          // Hard bound even if the host clock moves backwards.
          samples = samples.slice(-(this.server.history_seconds + 1));
          since = restarted ? 0 : samples.length ? samples[samples.length - 1].timestamp : 0;
          const latest = samples[samples.length - 1];
          this.outputRate = latest ? sum(latest.output_tps) : 0;
          this.prefillRate = latest ? sum(latest.prefill_tps) : 0;
          this.render();
        } catch (error) {
          if (!disposed) this.error = error.name === 'AbortError' ? 'Refresh timed out. Retrying automatically.' : error.message;
        } finally {
          clearTimeout(timeout);
          controller = null;
          inFlight = false;
          if (!disposed) timer = setTimeout(() => this.poll(), 2000);
        }
      },
      render() {
        const liveEnd = this.lastUpdated || Date.now();
        const windowStart = liveEnd - this.range * 1000;
        const liveStart = samples.length ? Math.min(liveEnd - 60000, Math.max(windowStart, samples[0].timestamp)) : windowStart;
        const [start, end] = inspectionDomain || [liveStart, liveEnd];
        renderDomain = [start, end];
        const visible = samples.filter(s => s.timestamp >= start && s.timestamp <= end);
        this.hasTraffic = visible.some(s => sum(s.output_tps) > 0 || sum(s.prefill_tps) > 0);
        const zoom = showSlider => ['inside', 'slider'].map(type => ({
          id: 'time-' + type, type, xAxisIndex: 0, filterMode: 'filter',
          ...(zoomWindow ? { startValue: zoomWindow[0], endValue: zoomWindow[1], rangeMode: ['value', 'value'] } : { start: 0, end: 100, rangeMode: ['percent', 'percent'] }),
          ...(type === 'inside' ? { zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false, throttle: 50 } : {
            show: showSlider, height: 18, bottom: 8, left: 58, right: 24,
            borderColor: '#39434e', backgroundColor: '#151a1f', fillerColor: '#79adff22',
            handleStyle: { color: '#68798a', borderColor: '#8d9dad' },
            textStyle: { color: '#929eab', fontSize: 9 }, labelFormatter: value => this.clock(value),
            dataBackground: { lineStyle: { color: '#516477' }, areaStyle: { color: '#354453' } },
          }),
        }));
        const base = (unit, integer = false, slider = false) => ({
          animation: false,
          textStyle: { fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif' },
          grid: { left: 58, right: 24, top: 16, bottom: slider ? 58 : 30 },
          dataZoom: zoom(slider),
          tooltip: { trigger: 'axis', confine: true, backgroundColor: '#23292f', borderColor: '#414b56', textStyle: { color: '#e0e7ef', fontSize: 11 }, axisPointer: { type: 'line', lineStyle: { color: '#647383' } }, valueFormatter: value => this.number(value) + ' ' + unit },
          xAxis: { type: 'time', min: start, max: end, boundaryGap: false, axisLine: { lineStyle: { color: '#363e47' } }, axisTick: { show: false }, axisLabel: { color: '#74818f', fontSize: 10, hideOverlap: true }, splitLine: { show: false } },
          yAxis: { type: 'value', min: 0, ...(integer ? { minInterval: 1 } : {}), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: '#74818f', fontSize: 10, formatter: value => this.compact(value) }, splitLine: { lineStyle: { color: '#2a3138', type: 'dashed' } } },
        });
        const throughput = key => this.slots.filter(slot => !this.hiddenSlots.includes(slot.id)).map(slot => ({
          id: 'slot-' + slot.id, name: 'Slot ' + (slot.id + 1), type: 'line', stack: 'tokens', showSymbol: false,
          color: this.colors[slot.id % this.colors.length], lineStyle: { width: 0 }, areaStyle: { opacity: 0.7 },
          emphasis: { focus: 'series' }, sampling: 'lttb', data: visible.map(s => [s.timestamp, s[key][slot.id] || 0]),
        }));
        applyingOptions = true;
        try {
        if (charts.output) charts.output.setOption({ ...base('tok/s', false, true), series: throughput('output_tps') }, { notMerge: true, silent: true });
        if (charts.prefill) charts.prefill.setOption({ ...base('tok/s'), series: throughput('prefill_tps') }, { notMerge: true, silent: true });
        if (charts.activity) charts.activity.setOption({ ...base('requests', true), series: [
          { name: 'Generating', key: 'decoding', color: '#65d5ab', stack: 'active' },
          { name: 'Prefilling', key: 'prefilling', color: '#79adff', stack: 'active' },
          { name: 'Queued', key: 'queued', color: '#ffc579' },
        ].map(series => ({ name: series.name, type: 'line', step: 'end', showSymbol: false, stack: series.stack,
          color: series.color, lineStyle: { width: series.stack ? 0 : 1.5, ...(series.key === 'queued' ? { type: 'dashed' } : {}) },
          areaStyle: series.stack ? { opacity: 0.35 } : undefined,
          data: visible.map(s => [s.timestamp, s[series.key]]),
        })) }, { notMerge: true, silent: true });
        } finally { applyingOptions = false; }
      },
    };
  });
});
