// Canvas rendering + hit-testing for the price line chart.
// Classic script (no modules) — exposes a single global `Chart`.

const Chart = (function () {
  const ZONE_COLOR_VAR = {
    ZONE_0: "--series-zone0",
    ZONE_1: "--series-zone1",
    ZONE_2: "--series-zone2",
    ZONE_3: "--series-zone3",
  };

  const MARGIN = { top: 16, right: 16, bottom: 32, left: 60 };
  const QUARTERS_PER_DAY = 96;
  const HOURS_PER_DAY = 24;

  let baseCanvas, overlayCanvas, baseCtx, overlayCtx, tooltipEl, emptyStateEl;
  let getState, onPointClick;
  let hourlyPrices = null; // computed once DATASET is available
  let cssWidth = 0, cssHeight = 0, dpr = 1;
  let layout = null; // last computed render layout, reused by hover/click
  let resizeTimer = null;

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function computeHourlyMeans() {
    const out = {};
    for (const zone of DATASET.zones) {
      const raw = DATASET.prices[zone];
      const n = raw.length / 4;
      const means = new Float32Array(n);
      for (let h = 0; h < n; h++) {
        const base = h * 4;
        means[h] = (raw[base] + raw[base + 1] + raw[base + 2] + raw[base + 3]) / 4;
      }
      out[zone] = means;
    }
    return out;
  }

  // The chart always overlays both resolutions, so the x-domain is always
  // "point index at 15-min resolution" — the hourly line is plotted by
  // sampling hourlyPrices at the center of each hour's 4 quarters.
  function getRawSlice(zone, state) {
    return DATASET.prices[zone].subarray(state.startDayIdx * QUARTERS_PER_DAY, (state.endDayIdx + 1) * QUARTERS_PER_DAY);
  }

  function getHourlySlice(zone, state) {
    return hourlyPrices[zone].subarray(state.startDayIdx * HOURS_PER_DAY, (state.endDayIdx + 1) * HOURS_PER_DAY);
  }

  function pad2(n) {
    return n < 10 ? "0" + n : String(n);
  }

  // Convert a raw (15-min) point index within the current slice to a
  // { date, hour (1..24), quarter (1..4) } label and to the flat dataset index.
  function indexToMeta(i, state) {
    const dayIdx = state.startDayIdx + Math.floor(i / QUARTERS_PER_DAY);
    const rem = i % QUARTERS_PER_DAY;
    const hour = Math.floor(rem / 4) + 1;
    const quarter = (rem % 4) + 1;
    const rawFlatIndex = dayIdx * QUARTERS_PER_DAY + (hour - 1) * 4 + (quarter - 1);
    return { dayIdx, hour, quarter, rawFlatIndex };
  }

  function formatLabel(meta) {
    const date = DATASET.dates[meta.dayIdx];
    const time = pad2(meta.hour - 1) + ":" + pad2((meta.quarter - 1) * 15);
    return date + " " + time;
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
    const rect = baseCanvas.parentElement.getBoundingClientRect();
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

  function renderBase() {
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
      emptyStateEl.hidden = false;
      layout = null;
      return;
    }
    emptyStateEl.hidden = true;

    const rawSlices = {};
    const hourlySlices = {};
    let n = 0;
    let dataMin = Infinity, dataMax = -Infinity;
    for (const z of zones) {
      const s = getRawSlice(z, state);
      rawSlices[z] = s;
      hourlySlices[z] = getHourlySlice(z, state);
      n = s.length;
      // raw always encompasses the hourly-mean range, so scanning it alone
      // is sufficient to bound both series' y-domain.
      for (let i = 0; i < s.length; i++) {
        if (s[i] < dataMin) dataMin = s[i];
        if (s[i] > dataMax) dataMax = s[i];
      }
    }

    const pad = (dataMax - dataMin) * 0.08 || 10;
    let yMin = Math.min(0, dataMin - pad);
    let yMax = Math.max(0, dataMax + pad);
    const { ticks, niceMin, niceMax } = niceTicks(yMin, yMax, 5);
    yMin = niceMin;
    yMax = niceMax;

    const indexToX = (i) => plot.left + (n <= 1 ? 0 : (i / (n - 1)) * plot.width);
    const priceToY = (v) => plot.top + (1 - (v - yMin) / (yMax - yMin)) * plot.height;

    // gridlines + y labels
    baseCtx.strokeStyle = cssVar("--grid-line");
    baseCtx.fillStyle = cssVar("--text-muted");
    baseCtx.font = "11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
    baseCtx.textAlign = "right";
    baseCtx.textBaseline = "middle";
    baseCtx.lineWidth = 1;

    for (const t of ticks) {
      if (t < yMin || t > yMax) continue;
      const y = Math.round(priceToY(t)) + 0.5;
      baseCtx.beginPath();
      baseCtx.moveTo(plot.left, y);
      baseCtx.lineTo(plot.left + plot.width, y);
      baseCtx.strokeStyle = cssVar("--grid-line");
      baseCtx.stroke();
      baseCtx.fillStyle = cssVar("--text-muted");
      baseCtx.fillText(String(Math.round(t)), plot.left - 10, y);
    }

    // zero reference line — solid, axis-weight (never dashed: a dash reads as
    // a threshold/projection, this is just "nothing")
    if (yMin < 0 && yMax > 0) {
      const y0 = Math.round(priceToY(0)) + 0.5;
      baseCtx.strokeStyle = cssVar("--axis-line");
      baseCtx.lineWidth = 1;
      baseCtx.beginPath();
      baseCtx.moveTo(plot.left, y0);
      baseCtx.lineTo(plot.left + plot.width, y0);
      baseCtx.stroke();
    }

    // x-axis date labels (sparse)
    const tickCount = Math.max(2, Math.min(8, Math.floor(plot.width / 110)));
    baseCtx.textAlign = "center";
    baseCtx.textBaseline = "top";
    baseCtx.fillStyle = cssVar("--text-muted");
    for (let k = 0; k <= tickCount; k++) {
      const i = Math.round((k / tickCount) * (n - 1));
      const meta = indexToMeta(i, state);
      const date = DATASET.dates[meta.dayIdx];
      const x = indexToX(i);
      baseCtx.fillText(date, Math.min(Math.max(x, plot.left + 24), plot.left + plot.width - 24), plot.top + plot.height + 10);
    }

    // axis baseline
    baseCtx.strokeStyle = cssVar("--axis-line");
    baseCtx.beginPath();
    baseCtx.moveTo(plot.left, plot.top + plot.height + 0.5);
    baseCtx.lineTo(plot.left + plot.width, plot.top + plot.height + 0.5);
    baseCtx.stroke();

    // series lines — two layers per zone, same hue, different weight/style:
    // hourly average underneath as a thin, dotted, lower-alpha line; 15-min
    // (raw) on top as the bold solid line that carries the story.
    baseCtx.lineJoin = "round";
    baseCtx.lineCap = "round";

    baseCtx.lineWidth = 1.5;
    baseCtx.setLineDash([1, 4]);
    baseCtx.globalAlpha = 0.55;
    for (const z of zones) {
      const s = hourlySlices[z];
      baseCtx.strokeStyle = cssVar(ZONE_COLOR_VAR[z]);
      baseCtx.beginPath();
      baseCtx.moveTo(indexToX(1.5), priceToY(s[0]));
      for (let h = 1; h < s.length; h++) {
        baseCtx.lineTo(indexToX(h * 4 + 1.5), priceToY(s[h]));
      }
      baseCtx.stroke();
    }
    baseCtx.setLineDash([]);
    baseCtx.globalAlpha = 1;

    baseCtx.lineWidth = 2;
    for (const z of zones) {
      const s = rawSlices[z];
      baseCtx.strokeStyle = cssVar(ZONE_COLOR_VAR[z]);
      baseCtx.beginPath();
      baseCtx.moveTo(indexToX(0), priceToY(s[0]));
      for (let i = 1; i < s.length; i++) {
        baseCtx.lineTo(indexToX(i), priceToY(s[i]));
      }
      baseCtx.stroke();
    }

    layout = {
      plot, n, yMin, yMax, indexToX, priceToY, zones, rawSlices, hourlySlices,
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

  function renderOverlay(pxX, pxY) {
    if (!layout) return;
    const { plot, zones, rawSlices, hourlySlices, indexToX, priceToY, state } = layout;
    if (pxX < plot.left || pxX > plot.left + plot.width || pxY < plot.top || pxY > plot.top + plot.height) {
      clearOverlay();
      return;
    }
    const i = nearestIndexFromPx(pxX);
    const x = indexToX(i);
    const hourIdx = Math.floor(i / 4);

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
      const color = cssVar(ZONE_COLOR_VAR[z]);
      const rawVal = rawSlices[z][i];
      const hourlyVal = hourlySlices[z][hourIdx];

      // smaller, translucent dot for the hourly average
      overlayCtx.globalAlpha = 0.55;
      overlayCtx.beginPath();
      overlayCtx.fillStyle = surface;
      overlayCtx.arc(x, priceToY(hourlyVal), 5, 0, Math.PI * 2);
      overlayCtx.fill();
      overlayCtx.beginPath();
      overlayCtx.fillStyle = color;
      overlayCtx.arc(x, priceToY(hourlyVal), 3, 0, Math.PI * 2);
      overlayCtx.fill();
      overlayCtx.globalAlpha = 1;

      // bold, opaque dot for the 15-min reading
      overlayCtx.beginPath();
      overlayCtx.fillStyle = surface;
      overlayCtx.arc(x, priceToY(rawVal), 6, 0, Math.PI * 2);
      overlayCtx.fill();
      overlayCtx.beginPath();
      overlayCtx.fillStyle = color;
      overlayCtx.arc(x, priceToY(rawVal), 4, 0, Math.PI * 2);
      overlayCtx.fill();

      rows.push({ zone: z, hourlyVal, rawVal, color });
    }

    const meta = indexToMeta(i, state);
    showTooltip(pxX, pxY, formatLabel(meta), rows);
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
      val.textContent = r.rawVal.toFixed(2);
      const sub = document.createElement("span");
      sub.className = "value-sub";
      sub.textContent = "avg " + r.hourlyVal.toFixed(2);
      row.appendChild(key);
      row.appendChild(name);
      row.appendChild(val);
      row.appendChild(sub);
      tooltipEl.appendChild(row);
    }

    tooltipEl.hidden = false;
    const wrapRect = baseCanvas.parentElement.getBoundingClientRect();
    let left = pxX + 14;
    let top = pxY + 14;
    const ttRect = tooltipEl.getBoundingClientRect();
    if (left + ttRect.width > wrapRect.width) left = pxX - ttRect.width - 14;
    if (top + ttRect.height > wrapRect.height) top = pxY - ttRect.height - 14;
    tooltipEl.style.left = left + "px";
    tooltipEl.style.top = top + "px";
  }

  function handleMove(evt) {
    const rect = overlayCanvas.getBoundingClientRect();
    renderOverlay(evt.clientX - rect.left, evt.clientY - rect.top);
  }

  function handleLeave() {
    clearOverlay();
  }

  function handleClick(evt) {
    if (!layout) return;
    const rect = overlayCanvas.getBoundingClientRect();
    const pxX = evt.clientX - rect.left;
    const { plot } = layout;
    if (pxX < plot.left || pxX > plot.left + plot.width) return;
    const i = nearestIndexFromPx(pxX);
    onPointClick(layout.state.startDayIdx * QUARTERS_PER_DAY + i);
  }

  function init(opts) {
    baseCanvas = opts.baseCanvas;
    overlayCanvas = opts.overlayCanvas;
    tooltipEl = opts.tooltipEl;
    emptyStateEl = opts.emptyStateEl;
    getState = opts.getState;
    onPointClick = opts.onPointClick;

    baseCtx = baseCanvas.getContext("2d");
    overlayCtx = overlayCanvas.getContext("2d");
    hourlyPrices = computeHourlyMeans();

    overlayCanvas.addEventListener("mousemove", handleMove);
    overlayCanvas.addEventListener("mouseleave", handleLeave);
    overlayCanvas.addEventListener("click", handleClick);

    const ro = new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(renderBase, 150);
    });
    ro.observe(baseCanvas.parentElement);

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderBase);
  }

  return { init, renderBase, clearOverlay, getHourlyPrices: () => hourlyPrices };
})();
