// Window comparison chart: the raw 15-min series for two independently
// chosen time windows of equal length, overlaid on one plot (x-axis =
// elapsed offset from each window's own start, so the two lines line up
// position-by-position regardless of their actual calendar dates).
// Classic script — exposes a single global `Compare`.

const Compare = (function () {
  const ZONE_COLOR_VAR = {
    ZONE_0: "--series-zone0",
    ZONE_1: "--series-zone1",
    ZONE_2: "--series-zone2",
    ZONE_3: "--series-zone3",
  };
  const QUARTERS_PER_DAY = 96;
  const MARGIN = { top: 16, right: 16, bottom: 28, left: 60 };

  let canvas, ctx, tooltipEl, wrapEl, hintEl, getState;
  let cssWidth = 0, cssHeight = 0, dpr = 1;
  let layout = null;
  let resizeTimer = null;

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function pad2(n) {
    return n < 10 ? "0" + n : String(n);
  }

  function flatIndexToLabel(idx) {
    const dayIdx = Math.floor(idx / QUARTERS_PER_DAY);
    const rem = idx % QUARTERS_PER_DAY;
    const hour = Math.floor(rem / 4);
    const minute = (rem % 4) * 15;
    return DATASET.dates[dayIdx] + " " + pad2(hour) + ":" + pad2(minute);
  }

  function formatOffset(q) {
    const totalMin = q * 15;
    const days = Math.floor(totalMin / 1440);
    const hours = Math.floor((totalMin % 1440) / 60);
    const mins = totalMin % 60;
    if (days > 0) return "+" + days + "d" + (hours ? " " + hours + "h" : "");
    if (hours > 0) return "+" + hours + "h" + (mins ? " " + mins + "m" : "");
    return "+" + mins + "m";
  }

  // Clamp a window start so the full duration always fits inside the dataset.
  function clampStart(startIdx, durationQuarters) {
    const totalLen = DATASET.dates.length * QUARTERS_PER_DAY;
    return Math.min(Math.max(startIdx, 0), Math.max(0, totalLen - durationQuarters));
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

  function updateHint(clampedA, clampedB, durationQuarters) {
    if (!hintEl) return;
    const endA = flatIndexToLabel(clampedA + durationQuarters - 1);
    const endB = flatIndexToLabel(clampedB + durationQuarters - 1);
    hintEl.textContent =
      "A: " + flatIndexToLabel(clampedA) + " → " + endA +
      "   ·   B: " + flatIndexToLabel(clampedB) + " → " + endB;
  }

  function render() {
    const state = getState();
    resizeCanvas();
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const zones = DATASET.zones.filter((z) => state.visibleZones.has(z));
    const duration = state.compareDurationQuarters;
    const clampedA = clampStart(state.compareStartA, duration);
    const clampedB = clampStart(state.compareStartB, duration);
    updateHint(clampedA, clampedB, duration);

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

    const slicesA = {};
    const slicesB = {};
    let dataMin = Infinity, dataMax = -Infinity;
    for (const z of zones) {
      const raw = DATASET.prices[z];
      const a = raw.subarray(clampedA, clampedA + duration);
      const b = raw.subarray(clampedB, clampedB + duration);
      slicesA[z] = a;
      slicesB[z] = b;
      for (let i = 0; i < duration; i++) {
        if (a[i] < dataMin) dataMin = a[i];
        if (a[i] > dataMax) dataMax = a[i];
        if (b[i] < dataMin) dataMin = b[i];
        if (b[i] > dataMax) dataMax = b[i];
      }
    }

    const pad = (dataMax - dataMin) * 0.08 || 10;
    let yMin = Math.min(0, dataMin - pad);
    let yMax = Math.max(0, dataMax + pad);
    const { ticks, niceMin, niceMax } = niceTicks(yMin, yMax, 5);
    yMin = niceMin;
    yMax = niceMax;

    const n = duration;
    const indexToX = (i) => plot.left + (n <= 1 ? 0 : (i / (n - 1)) * plot.width);
    const valueToY = (v) => plot.top + (1 - (v - yMin) / (yMax - yMin)) * plot.height;

    ctx.font = "11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of ticks) {
      if (t < yMin || t > yMax) continue;
      const y = Math.round(valueToY(t)) + 0.5;
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
      const y0 = Math.round(valueToY(0)) + 0.5;
      ctx.strokeStyle = cssVar("--axis-line");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plot.left, y0);
      ctx.lineTo(plot.left + plot.width, y0);
      ctx.stroke();
    }

    const tickCount = Math.max(2, Math.min(8, Math.floor(plot.width / 90)));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = cssVar("--text-muted");
    for (let k = 0; k <= tickCount; k++) {
      const i = Math.round((k / tickCount) * (n - 1));
      const x = indexToX(i);
      ctx.fillText(formatOffset(i), Math.min(Math.max(x, plot.left + 20), plot.left + plot.width - 20), plot.top + plot.height + 8);
    }

    ctx.strokeStyle = cssVar("--axis-line");
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top + plot.height + 0.5);
    ctx.lineTo(plot.left + plot.width, plot.top + plot.height + 0.5);
    ctx.stroke();

    // Window B underneath, thin/dotted/translucent; Window A on top, bold
    // solid — the same hue-identity + line-style convention used by the
    // main chart's hourly-vs-15-min overlay.
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    ctx.lineWidth = 1.5;
    ctx.setLineDash([1, 4]);
    ctx.globalAlpha = 0.55;
    for (const z of zones) {
      const s = slicesB[z];
      ctx.strokeStyle = cssVar(ZONE_COLOR_VAR[z]);
      ctx.beginPath();
      ctx.moveTo(indexToX(0), valueToY(s[0]));
      for (let i = 1; i < s.length; i++) ctx.lineTo(indexToX(i), valueToY(s[i]));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    ctx.lineWidth = 2;
    for (const z of zones) {
      const s = slicesA[z];
      ctx.strokeStyle = cssVar(ZONE_COLOR_VAR[z]);
      ctx.beginPath();
      ctx.moveTo(indexToX(0), valueToY(s[0]));
      for (let i = 1; i < s.length; i++) ctx.lineTo(indexToX(i), valueToY(s[i]));
      ctx.stroke();
    }

    layout = { plot, n, indexToX, valueToY, zones, slicesA, slicesB, clampedA, clampedB };
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
      val.textContent = r.aVal.toFixed(2);
      const sub = document.createElement("span");
      sub.className = "value-sub";
      sub.textContent = "B " + r.bVal.toFixed(2);
      row.appendChild(key);
      row.appendChild(name);
      row.appendChild(val);
      row.appendChild(sub);
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
    const { plot, zones, slicesA, slicesB, indexToX, valueToY, clampedA, clampedB } = layout;
    if (pxX < plot.left || pxX > plot.left + plot.width || pxY < plot.top || pxY > plot.top + plot.height) {
      tooltipEl.hidden = true;
      render();
      return;
    }
    const i = nearestIndexFromPx(pxX);

    render();
    const x = indexToX(i);
    ctx.strokeStyle = cssVar("--axis-line");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, plot.top);
    ctx.lineTo(Math.round(x) + 0.5, plot.top + plot.height);
    ctx.stroke();

    const surface = cssVar("--surface-1");
    const rows = [];
    for (const z of zones) {
      const color = cssVar(ZONE_COLOR_VAR[z]);
      const aVal = slicesA[z][i];
      const bVal = slicesB[z][i];

      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      ctx.fillStyle = surface;
      ctx.arc(x, valueToY(bVal), 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(x, valueToY(bVal), 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      ctx.beginPath();
      ctx.fillStyle = surface;
      ctx.arc(x, valueToY(aVal), 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.arc(x, valueToY(aVal), 4, 0, Math.PI * 2);
      ctx.fill();

      rows.push({ zone: z, aVal, bVal, color });
    }

    const timeLabel = formatOffset(i) + "  —  A " + flatIndexToLabel(clampedA + i) + "  /  B " + flatIndexToLabel(clampedB + i);
    showTooltip(pxX, pxY, timeLabel, rows);
  }

  function handleLeave() {
    tooltipEl.hidden = true;
    render();
  }

  function init(opts) {
    canvas = opts.canvas;
    tooltipEl = opts.tooltipEl;
    wrapEl = opts.wrapEl;
    hintEl = opts.hintEl;
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
