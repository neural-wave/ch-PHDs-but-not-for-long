// "Typical day" chart: average price by hour-of-day (0..23), averaged across
// the days in the currently selected date range, one line per visible zone.
// Classic script — exposes a single global `Profile`.

const Profile = (function () {
  const ZONE_COLOR_VAR = {
    ZONE_0: "--series-zone0",
    ZONE_1: "--series-zone1",
    ZONE_2: "--series-zone2",
    ZONE_3: "--series-zone3",
  };
  const HOURS_PER_DAY = 24;
  const MARGIN = { top: 14, right: 16, bottom: 26, left: 56 };

  let canvas, ctx, tooltipEl, wrapEl, getState;
  let cssWidth = 0, cssHeight = 0, dpr = 1;
  let layout = null;
  let resizeTimer = null;

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function pad2(n) {
    return n < 10 ? "0" + n : String(n);
  }

  // Average price for each hour-of-day (0..23) across the days currently
  // selected, for one zone. Reuses Chart's cached per-day hourly-mean series.
  function computeProfile(zone, state) {
    const means = Chart.getHourlyPrices()[zone];
    const startIdx = state.startDayIdx * HOURS_PER_DAY;
    const endIdx = (state.endDayIdx + 1) * HOURS_PER_DAY;
    const sums = new Float64Array(HOURS_PER_DAY);
    const counts = new Float64Array(HOURS_PER_DAY);
    for (let i = startIdx; i < endIdx; i++) {
      const h = i % HOURS_PER_DAY;
      sums[h] += means[i];
      counts[h]++;
    }
    const profile = new Float64Array(HOURS_PER_DAY);
    for (let h = 0; h < HOURS_PER_DAY; h++) profile[h] = sums[h] / counts[h];
    return profile;
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

  function resizeCanvas() {
    const rect = wrapEl.getBoundingClientRect();
    cssWidth = rect.width;
    cssHeight = rect.height;
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function render() {
    const state = getState();
    resizeCanvas();
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const zones = DATASET.zones.filter((z) => state.visibleZones.has(z));
    const plot = {
      left: MARGIN.left,
      top: MARGIN.top,
      width: cssWidth - MARGIN.left - MARGIN.right,
      height: cssHeight - MARGIN.top - MARGIN.bottom,
    };

    if (zones.length === 0) {
      layout = null;
      ctx.fillStyle = cssVar("--text-muted");
      ctx.font = "13px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Select at least one zone.", cssWidth / 2, cssHeight / 2);
      return;
    }

    const profiles = {};
    let dataMin = Infinity, dataMax = -Infinity;
    for (const z of zones) {
      const p = computeProfile(z, state);
      profiles[z] = p;
      for (let h = 0; h < HOURS_PER_DAY; h++) {
        if (p[h] < dataMin) dataMin = p[h];
        if (p[h] > dataMax) dataMax = p[h];
      }
    }

    const pad = (dataMax - dataMin) * 0.12 || 10;
    let yMin = Math.min(0, dataMin - pad);
    let yMax = Math.max(0, dataMax + pad);
    const { ticks, niceMin, niceMax } = niceTicks(yMin, yMax, 3);
    yMin = niceMin;
    yMax = niceMax;

    const hourToX = (h) => plot.left + (h / (HOURS_PER_DAY - 1)) * plot.width;
    const priceToY = (v) => plot.top + (1 - (v - yMin) / (yMax - yMin)) * plot.height;

    ctx.font = "11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of ticks) {
      if (t < yMin || t > yMax) continue;
      const y = Math.round(priceToY(t)) + 0.5;
      ctx.strokeStyle = cssVar("--grid-line");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.left + plot.width, y);
      ctx.stroke();
      ctx.fillStyle = cssVar("--text-muted");
      ctx.fillText(String(Math.round(t)), plot.left - 10, y);
    }

    if (yMin < 0 && yMax > 0) {
      const y0 = Math.round(priceToY(0)) + 0.5;
      ctx.strokeStyle = cssVar("--axis-line");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plot.left, y0);
      ctx.lineTo(plot.left + plot.width, y0);
      ctx.stroke();
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = cssVar("--text-muted");
    for (let h = 0; h <= HOURS_PER_DAY - 1; h += 6) {
      ctx.fillText(pad2(h) + ":00", hourToX(h), plot.top + plot.height + 8);
    }

    ctx.strokeStyle = cssVar("--axis-line");
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top + plot.height + 0.5);
    ctx.lineTo(plot.left + plot.width, plot.top + plot.height + 0.5);
    ctx.stroke();

    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = 2;
    for (const z of zones) {
      const p = profiles[z];
      ctx.strokeStyle = cssVar(ZONE_COLOR_VAR[z]);
      ctx.beginPath();
      ctx.moveTo(hourToX(0), priceToY(p[0]));
      for (let h = 1; h < HOURS_PER_DAY; h++) ctx.lineTo(hourToX(h), priceToY(p[h]));
      ctx.stroke();
    }

    layout = { plot, yMin, yMax, hourToX, priceToY, zones, profiles };
  }

  function showTooltip(pxX, pxY, hour, rows) {
    tooltipEl.innerHTML = "";
    const time = document.createElement("div");
    time.className = "tooltip-time";
    time.textContent = pad2(hour) + ":00 average";
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

  function handleMove(evt) {
    if (!layout) return;
    const rect = canvas.getBoundingClientRect();
    const pxX = evt.clientX - rect.left;
    const pxY = evt.clientY - rect.top;
    const { plot, zones, profiles, hourToX, priceToY } = layout;
    if (pxX < plot.left || pxX > plot.left + plot.width || pxY < plot.top || pxY > plot.top + plot.height) {
      tooltipEl.hidden = true;
      render();
      return;
    }
    const hour = Math.min(HOURS_PER_DAY - 1, Math.max(0, Math.round(((pxX - plot.left) / plot.width) * (HOURS_PER_DAY - 1))));

    render();
    const x = hourToX(hour);
    ctx.strokeStyle = cssVar("--axis-line");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, plot.top);
    ctx.lineTo(Math.round(x) + 0.5, plot.top + plot.height);
    ctx.stroke();

    const surface = cssVar("--surface-1");
    const rows = [];
    for (const z of zones) {
      const v = profiles[z][hour];
      const y = priceToY(v);
      ctx.beginPath();
      ctx.fillStyle = surface;
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.fillStyle = cssVar(ZONE_COLOR_VAR[z]);
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      rows.push({ zone: z, value: v, color: cssVar(ZONE_COLOR_VAR[z]) });
    }
    showTooltip(pxX, pxY, hour, rows);
  }

  function handleLeave() {
    tooltipEl.hidden = true;
    render();
  }

  function init(opts) {
    canvas = opts.canvas;
    tooltipEl = opts.tooltipEl;
    wrapEl = opts.wrapEl;
    getState = opts.getState;
    ctx = canvas.getContext("2d");

    canvas.addEventListener("mousemove", handleMove);
    canvas.addEventListener("mouseleave", handleLeave);

    const ro = new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(render, 150);
    });
    ro.observe(wrapEl);

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  }

  return { init, render };
})();
