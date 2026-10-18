// Hourly deviation chart: for each hour and zone, how far the four 15-min
// readings sit from a reference average — against that hour's own mean,
// either the squared-error sum or the absolute-error sum (both within-hour
// dispersion/volatility measures), or the signed sum against that whole
// day's mean (a day deviation measure — unlike the hourly-mean case, this
// is NOT trivially zero, since a single hour's readings rarely average out
// to the day's mean).
// Classic script — exposes a single global `Deviation`.

const Deviation = (function () {
  const ZONE_COLOR_VAR = {
    ZONE_0: "--series-zone0",
    ZONE_1: "--series-zone1",
    ZONE_2: "--series-zone2",
    ZONE_3: "--series-zone3",
  };
  const HOURS_PER_DAY = 24;
  const MARGIN = { top: 16, right: 16, bottom: 28, left: 60 };

  let baseCanvas, overlayCanvas, baseCtx, overlayCtx, tooltipEl, wrapEl, getState;
  let cssWidth = 0, cssHeight = 0, dpr = 1;
  let layout = null;
  let resizeTimer = null;
  let dayAverages = null; // computed once Chart's hourly means are available

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function computeDayAverages() {
    const out = {};
    const hourly = Chart.getHourlyPrices();
    for (const zone of DATASET.zones) {
      const means = hourly[zone];
      const numDays = means.length / HOURS_PER_DAY;
      const days = new Float64Array(numDays);
      for (let d = 0; d < numDays; d++) {
        let sum = 0;
        const base = d * HOURS_PER_DAY;
        for (let h = 0; h < HOURS_PER_DAY; h++) sum += means[base + h];
        days[d] = sum / HOURS_PER_DAY;
      }
      out[zone] = days;
    }
    return out;
  }

  // For hourly index h (global, across the whole dataset), the 4 raw 15-min
  // values live at raw[h*4 .. h*4+3] — true because both the raw and
  // hourly-mean arrays are built in the same day-major, hour-minor order.
  function computeSeries(zone, state) {
    const raw = DATASET.prices[zone];
    const startH = state.startDayIdx * HOURS_PER_DAY;
    const endH = (state.endDayIdx + 1) * HOURS_PER_DAY;
    const out = new Float64Array(endH - startH);

    if (state.deviationMode === "squared" || state.deviationMode === "absolute") {
      const means = Chart.getHourlyPrices()[zone];
      const squared = state.deviationMode === "squared";
      for (let h = startH; h < endH; h++) {
        const mean = means[h];
        const base = h * 4;
        let sum = 0;
        for (let q = 0; q < 4; q++) {
          const d = raw[base + q] - mean;
          sum += squared ? d * d : Math.abs(d);
        }
        out[h - startH] = sum;
      }
    } else {
      const days = dayAverages[zone];
      for (let h = startH; h < endH; h++) {
        const dayAvg = days[Math.floor(h / HOURS_PER_DAY)];
        const base = h * 4;
        let sum = 0;
        for (let q = 0; q < 4; q++) sum += raw[base + q] - dayAvg;
        out[h - startH] = sum;
      }
    }
    return out;
  }

  function pad2(n) {
    return n < 10 ? "0" + n : String(n);
  }

  function indexToMeta(i, state) {
    const dayIdx = state.startDayIdx + Math.floor(i / HOURS_PER_DAY);
    const hour = (i % HOURS_PER_DAY) + 1;
    return { dayIdx, hour };
  }

  function formatLabel(meta) {
    return DATASET.dates[meta.dayIdx] + " " + pad2(meta.hour - 1) + ":00";
  }

  function niceTicks(min, max, targetCount) {
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    const rawStep = span / targetCount;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const residual = rawStep / mag;
    let step;
    if (residual > 5) step = 10 * mag;
    else if (residual > 2) step = 5 * mag;
    else if (residual > 1) step = 2 * mag;
    else step = mag;
    const niceMin = Math.floor(min / step) * step;
    const niceMax = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + step * 0.001; v += step) {
      ticks.push(Math.round(v / step) * step);
    }
    return { ticks, niceMin, niceMax };
  }

  function resizeCanvases() {
    const rect = wrapEl.getBoundingClientRect();
    cssWidth = rect.width;
    cssHeight = rect.height;
    dpr = window.devicePixelRatio || 1;
    for (const c of [baseCanvas, overlayCanvas]) {
      c.width = Math.round(cssWidth * dpr);
      c.height = Math.round(cssHeight * dpr);
    }
    baseCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function render() {
    const state = getState();
    resizeCanvases();
    baseCtx.clearRect(0, 0, cssWidth, cssHeight);

    const zones = DATASET.zones.filter((z) => state.visibleZones.has(z));
    const plot = {
      left: MARGIN.left,
      top: MARGIN.top,
      width: cssWidth - MARGIN.left - MARGIN.right,
      height: cssHeight - MARGIN.top - MARGIN.bottom,
    };

    if (zones.length === 0) {
      layout = null;
      baseCtx.fillStyle = cssVar("--text-muted");
      baseCtx.font = "13px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
      baseCtx.textAlign = "center";
      baseCtx.textBaseline = "middle";
      baseCtx.fillText("Select at least one zone.", cssWidth / 2, cssHeight / 2);
      return;
    }

    const series = {};
    let n = 0;
    let dataMin = Infinity, dataMax = -Infinity;
    for (const z of zones) {
      const s = computeSeries(z, state);
      series[z] = s;
      n = s.length;
      for (let i = 0; i < s.length; i++) {
        if (s[i] < dataMin) dataMin = s[i];
        if (s[i] > dataMax) dataMax = s[i];
      }
    }

    const pad = (dataMax - dataMin) * 0.08 || 1;
    let yMin = Math.min(0, dataMin - pad);
    let yMax = Math.max(0, dataMax + pad);
    const { ticks, niceMin, niceMax } = niceTicks(yMin, yMax, 4);
    yMin = niceMin;
    yMax = niceMax;

    const indexToX = (i) => plot.left + (n <= 1 ? 0 : (i / (n - 1)) * plot.width);
    const valueToY = (v) => plot.top + (1 - (v - yMin) / (yMax - yMin)) * plot.height;

    baseCtx.font = "11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
    baseCtx.textAlign = "right";
    baseCtx.textBaseline = "middle";
    for (const t of ticks) {
      if (t < yMin || t > yMax) continue;
      const y = Math.round(valueToY(t)) + 0.5;
      baseCtx.strokeStyle = cssVar("--grid-line");
      baseCtx.lineWidth = 1;
      baseCtx.beginPath();
      baseCtx.moveTo(plot.left, y);
      baseCtx.lineTo(plot.left + plot.width, y);
      baseCtx.stroke();
      baseCtx.fillStyle = cssVar("--text-muted");
      baseCtx.fillText(String(Math.round(t * 100) / 100), plot.left - 10, y);
    }

    if (yMin < 0 && yMax > 0) {
      const y0 = Math.round(valueToY(0)) + 0.5;
      baseCtx.strokeStyle = cssVar("--axis-line");
      baseCtx.lineWidth = 1;
      baseCtx.beginPath();
      baseCtx.moveTo(plot.left, y0);
      baseCtx.lineTo(plot.left + plot.width, y0);
      baseCtx.stroke();
    }

    const tickCount = Math.max(2, Math.min(8, Math.floor(plot.width / 110)));
    baseCtx.textAlign = "center";
    baseCtx.textBaseline = "top";
    baseCtx.fillStyle = cssVar("--text-muted");
    for (let k = 0; k <= tickCount; k++) {
      const i = Math.round((k / tickCount) * (n - 1));
      const meta = indexToMeta(i, state);
      const date = DATASET.dates[meta.dayIdx];
      const x = indexToX(i);
      baseCtx.fillText(date, Math.min(Math.max(x, plot.left + 24), plot.left + plot.width - 24), plot.top + plot.height + 8);
    }

    baseCtx.strokeStyle = cssVar("--axis-line");
    baseCtx.beginPath();
    baseCtx.moveTo(plot.left, plot.top + plot.height + 0.5);
    baseCtx.lineTo(plot.left + plot.width, plot.top + plot.height + 0.5);
    baseCtx.stroke();

    baseCtx.lineJoin = "round";
    baseCtx.lineCap = "round";
    baseCtx.lineWidth = 2;
    for (const z of zones) {
      const s = series[z];
      baseCtx.strokeStyle = cssVar(ZONE_COLOR_VAR[z]);
      baseCtx.beginPath();
      baseCtx.moveTo(indexToX(0), valueToY(s[0]));
      for (let i = 1; i < s.length; i++) baseCtx.lineTo(indexToX(i), valueToY(s[i]));
      baseCtx.stroke();
    }

    layout = {
      plot, n, yMin, yMax, indexToX, valueToY, zones, series,
      state: { ...state, visibleZones: new Set(state.visibleZones) },
    };
  }

  function clearOverlay() {
    overlayCtx.clearRect(0, 0, cssWidth, cssHeight);
    tooltipEl.hidden = true;
  }

  function nearestIndexFromPx(px) {
    const { plot, n } = layout;
    const t = (px - plot.left) / plot.width;
    return Math.min(n - 1, Math.max(0, Math.round(t * (n - 1))));
  }

  function showTooltip(pxX, pxY, timeLabel, rows) {
    tooltipEl.innerHTML = "";
    const time = document.createElement("div");
    time.className = "tooltip-time";
    time.textContent = timeLabel;
    tooltipEl.appendChild(time);

    for (const r of rows) {
      const row = document.createElement("div");
      row.className = "tooltip-row";
      const key = document.createElement("span");
      key.className = "key";
      key.style.background = r.color;
      const name = document.createElement("span");
      name.className = "series-name";
      name.textContent = r.zone.replace("ZONE_", "Zone ") + ":";
      const val = document.createElement("span");
      val.className = "value";
      val.textContent = r.value.toFixed(2);
      row.appendChild(key);
      row.appendChild(name);
      row.appendChild(val);
      tooltipEl.appendChild(row);
    }

    tooltipEl.hidden = false;
    const wrapRect = wrapEl.getBoundingClientRect();
    let left = pxX + 14;
    let top = pxY + 14;
    const ttRect = tooltipEl.getBoundingClientRect();
    if (left + ttRect.width > wrapRect.width) left = pxX - ttRect.width - 14;
    if (top + ttRect.height > wrapRect.height) top = pxY - ttRect.height - 14;
    tooltipEl.style.left = left + "px";
    tooltipEl.style.top = top + "px";
  }

  function renderOverlay(pxX, pxY) {
    if (!layout) return;
    const { plot, zones, series, indexToX, valueToY, state } = layout;
    if (pxX < plot.left || pxX > plot.left + plot.width || pxY < plot.top || pxY > plot.top + plot.height) {
      clearOverlay();
      return;
    }
    const i = nearestIndexFromPx(pxX);
    const x = indexToX(i);

    overlayCtx.clearRect(0, 0, cssWidth, cssHeight);
    overlayCtx.strokeStyle = cssVar("--axis-line");
    overlayCtx.lineWidth = 1;
    overlayCtx.beginPath();
    overlayCtx.moveTo(Math.round(x) + 0.5, plot.top);
    overlayCtx.lineTo(Math.round(x) + 0.5, plot.top + plot.height);
    overlayCtx.stroke();

    const surface = cssVar("--surface-1");
    const rows = [];
    for (const z of zones) {
      const v = series[z][i];
      const y = valueToY(v);
      overlayCtx.beginPath();
      overlayCtx.fillStyle = surface;
      overlayCtx.arc(x, y, 6, 0, Math.PI * 2);
      overlayCtx.fill();
      overlayCtx.beginPath();
      overlayCtx.fillStyle = cssVar(ZONE_COLOR_VAR[z]);
      overlayCtx.arc(x, y, 4, 0, Math.PI * 2);
      overlayCtx.fill();
      rows.push({ zone: z, value: v, color: cssVar(ZONE_COLOR_VAR[z]) });
    }

    const meta = indexToMeta(i, state);
    showTooltip(pxX, pxY, formatLabel(meta), rows);
  }

  function handleMove(evt) {
    const rect = overlayCanvas.getBoundingClientRect();
    renderOverlay(evt.clientX - rect.left, evt.clientY - rect.top);
  }

  function handleLeave() {
    clearOverlay();
  }

  function init(opts) {
    baseCanvas = opts.baseCanvas;
    overlayCanvas = opts.overlayCanvas;
    tooltipEl = opts.tooltipEl;
    wrapEl = opts.wrapEl;
    getState = opts.getState;

    baseCtx = baseCanvas.getContext("2d");
    overlayCtx = overlayCanvas.getContext("2d");
    dayAverages = computeDayAverages();

    overlayCanvas.addEventListener("mousemove", handleMove);
    overlayCanvas.addEventListener("mouseleave", handleLeave);

    const ro = new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(render, 150);
    });
    ro.observe(wrapEl);

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  }

  return { init, render };
})();
