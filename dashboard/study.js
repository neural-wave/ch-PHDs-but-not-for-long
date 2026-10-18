// "Time series study" tab: static charts of the measured properties in
// ../TIMESERIES_INSIGHTS.md. The figures below are transcribed from that
// document (they are whole-dataset statistics, not computed here), so they
// don't depend on the dashboard filters - update them when the document
// changes. Classic script - exposes a single global `Study`.

const Study = (function () {
  const ZONE_COLOR_VAR = {
    ZONE_0: "--series-zone0",
    ZONE_1: "--series-zone1",
    ZONE_2: "--series-zone2",
    ZONE_3: "--series-zone3",
  };
  // Figures pooled across zones are not any one zone, so they use a neutral
  // ink instead of borrowing a zone hue.
  const POOLED_COLOR_VAR = "--series-pooled";
  const FONT = "11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";

  // Tab §8 (doc §6) - day-to-day autocorrelation of the 96-point deviation vector.
  const ACF = {
    lags: [1, 2, 3, 7, 14, 21, 28, 56],
    byZone: {
      ZONE_0: [0.65, 0.56, 0.53, 0.57, 0.53, 0.51, 0.49, 0.43],
      ZONE_1: [0.65, 0.57, 0.55, 0.59, 0.55, 0.54, 0.52, 0.46],
      ZONE_2: [0.63, 0.55, 0.53, 0.58, 0.55, 0.54, 0.52, 0.48],
      ZONE_3: [0.58, 0.52, 0.51, 0.51, 0.48, 0.47, 0.46, 0.42],
    },
  };

  // Tab §1 / §2 (doc §7 / §8) - mean |d| (EUR/MWh), averaged over zones.
  const AMP_BY_QUARTER = [10.86, 4.28, 4.20, 10.97];
  const AMP_BY_WEEKDAY = [8.37, 8.34, 8.37, 7.93, 7.54, 6.37, 6.05];
  const AMP_BY_HOUR = [
    6.4, 3.7, 2.9, 2.1, 2.5, 5.6,
    7.8, 7.9, 11.7, 12.5, 9.0, 5.6,
    4.1, 3.6, 5.5, 8.3, 11.9, 14.1,
    13.3, 11.8, 9.1, 8.6, 7.2, 6.6,
  ];
  const AMP_BY_MONTH = {
    labels: ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
    years: ["2025", "2025", "2025", "2026", "2026", "2026", "2026", "2026", "2026", "2026", "2026", "2026"],
    values: [8.94, 6.06, 4.46, 5.58, 5.05, 8.96, 8.25, 8.07, 8.40, 7.62, 8.77, 10.70],
  };
  const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  let charts = [];

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function zoneLabel(zone) {
    return zone.replace("ZONE_", "Zone ");
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
    return { ticks, niceMin, niceMax, step };
  }

  // Shared canvas plumbing for one chart: DPR-aware sizing from its wrap,
  // tooltip placement, and a debounced redraw on resize.
  function makeSurface(ids) {
    const wrapEl = document.getElementById(ids.wrap);
    const canvas = document.getElementById(ids.canvas);
    const tooltipEl = document.getElementById(ids.tooltip);
    const ctx = canvas.getContext("2d");
    const surface = { wrapEl, canvas, tooltipEl, ctx, width: 0, height: 0 };

    surface.resize = function () {
      const rect = wrapEl.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      surface.width = rect.width;
      surface.height = rect.height;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
    };

    surface.showTooltip = function (pxX, pxY, title, rows) {
      tooltipEl.innerHTML = "";
      const head = document.createElement("div");
      head.className = "tooltip-time";
      head.textContent = title;
      tooltipEl.appendChild(head);
      for (const r of rows) {
        const row = document.createElement("div");
        row.className = "tooltip-row";
        const key = document.createElement("span");
        key.className = "key";
        key.style.background = r.color;
        const name = document.createElement("span");
        name.className = "series-name";
        name.textContent = r.name + ":";
        const val = document.createElement("span");
        val.className = "value";
        val.textContent = r.value;
        row.appendChild(key);
        row.appendChild(name);
        row.appendChild(val);
        tooltipEl.appendChild(row);
      }
      tooltipEl.hidden = false;
      const wrapRect = wrapEl.getBoundingClientRect();
      const ttRect = tooltipEl.getBoundingClientRect();
      let left = pxX + 14;
      let top = pxY + 14;
      if (left + ttRect.width > wrapRect.width) left = pxX - ttRect.width - 14;
      if (top + ttRect.height > wrapRect.height) top = pxY - ttRect.height - 14;
      tooltipEl.style.left = Math.max(0, left) + "px";
      tooltipEl.style.top = Math.max(0, top) + "px";
    };

    surface.hideTooltip = function () {
      tooltipEl.hidden = true;
    };

    surface.pointer = function (evt) {
      const rect = canvas.getBoundingClientRect();
      return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
    };

    return surface;
  }

  function drawYGrid(ctx, plot, ticks, valueToY, decimals) {
    ctx.font = FONT;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of ticks) {
      const y = Math.round(valueToY(t)) + 0.5;
      ctx.strokeStyle = cssVar("--grid-line");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.left + plot.width, y);
      ctx.stroke();
      ctx.fillStyle = cssVar("--text-muted");
      ctx.fillText(t.toFixed(decimals), plot.left - 10, y);
    }
  }

  function drawBaseline(ctx, plot) {
    ctx.strokeStyle = cssVar("--axis-line");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top + plot.height + 0.5);
    ctx.lineTo(plot.left + plot.width, plot.top + plot.height + 0.5);
    ctx.stroke();
  }

  // Single-series column chart of a pooled (all-zone) measure. Columns grow
  // from a zero baseline, are capped at 24px wide with a 4px rounded cap, and
  // only the indices in `labelIdx` get a direct value label.
  function columnChart(spec) {
    const s = makeSurface(spec.ids);
    const MARGIN = { top: 18, right: 12, bottom: 26, left: 40 };
    let layout = null;
    let hoverIdx = null;

    function render() {
      s.resize();
      const { ctx } = s;
      const n = spec.values.length;
      const plot = {
        left: MARGIN.left,
        top: MARGIN.top,
        width: s.width - MARGIN.left - MARGIN.right,
        height: s.height - MARGIN.top - MARGIN.bottom,
      };
      if (plot.width <= 0 || plot.height <= 0) { layout = null; return; }

      const { ticks, niceMax } = niceTicks(0, Math.max(...spec.values), 3);
      const valueToY = (v) => plot.top + (1 - v / niceMax) * plot.height;
      drawYGrid(ctx, plot, ticks, valueToY, 0);

      const band = plot.width / n;
      const barW = Math.max(2, Math.min(24, band - 2));
      const color = cssVar(POOLED_COLOR_VAR);
      const baseY = plot.top + plot.height;

      for (let i = 0; i < n; i++) {
        const cx = plot.left + band * (i + 0.5);
        const y = valueToY(spec.values[i]);
        ctx.fillStyle = color;
        ctx.globalAlpha = hoverIdx === null || hoverIdx === i ? 1 : 0.45;
        ctx.beginPath();
        ctx.roundRect(cx - barW / 2, y, barW, baseY - y, [4, 4, 0, 0]);
        ctx.fill();
        ctx.globalAlpha = 1;

        if (spec.labelIdx.includes(i)) {
          ctx.font = "600 " + FONT;
          ctx.fillStyle = cssVar("--text-secondary");
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText(spec.values[i].toFixed(spec.decimals), cx, y - 3);
        }

        const tick = spec.tickLabel(i);
        if (tick !== null) {
          ctx.font = FONT;
          ctx.fillStyle = cssVar("--text-muted");
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(tick, cx, baseY + 8);
        }
      }
      drawBaseline(ctx, plot);
      layout = { plot, band };
    }

    function handleMove(evt) {
      if (!layout) return;
      const p = s.pointer(evt);
      const { plot, band } = layout;
      const inside = p.x >= plot.left && p.x <= plot.left + plot.width &&
                     p.y >= plot.top && p.y <= plot.top + plot.height;
      const idx = inside ? Math.min(spec.values.length - 1, Math.floor((p.x - plot.left) / band)) : null;
      if (idx !== hoverIdx) {
        hoverIdx = idx;
        render();
      }
      if (idx === null) { s.hideTooltip(); return; }
      s.showTooltip(p.x, p.y, spec.tipTitle(idx), [{
        name: spec.seriesName,
        value: spec.values[idx].toFixed(spec.decimals),
        color: cssVar(POOLED_COLOR_VAR),
      }]);
    }

    function handleLeave() {
      hoverIdx = null;
      s.hideTooltip();
      render();
    }

    s.canvas.addEventListener("mousemove", handleMove);
    s.canvas.addEventListener("mouseleave", handleLeave);
    return { render, wrapEl: s.wrapEl };
  }

  // §8 autocorrelation: one line per zone over a linear lag axis (days), so
  // the uneven lag spacing is shown honestly. Lines end in direct labels,
  // nudged apart vertically where the zones finish close together.
  function acfChart(ids) {
    const s = makeSurface(ids);
    const MARGIN = { top: 14, right: 64, bottom: 26, left: 44 };
    const zones = DATASET.zones;
    let layout = null;

    function render(hoverLagIdx) {
      s.resize();
      const { ctx } = s;
      const plot = {
        left: MARGIN.left,
        top: MARGIN.top,
        width: s.width - MARGIN.left - MARGIN.right,
        height: s.height - MARGIN.top - MARGIN.bottom,
      };
      if (plot.width <= 0 || plot.height <= 0) { layout = null; return; }

      let lo = Infinity, hi = -Infinity;
      for (const z of zones) for (const v of ACF.byZone[z]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const { ticks, niceMin, niceMax } = niceTicks(lo, hi, 3);
      const maxLag = ACF.lags[ACF.lags.length - 1];
      const lagToX = (lag) => plot.left + (lag / maxLag) * plot.width;
      const valueToY = (v) => plot.top + (1 - (v - niceMin) / (niceMax - niceMin)) * plot.height;

      drawYGrid(ctx, plot, ticks, valueToY, 2);
      ctx.font = FONT;
      ctx.fillStyle = cssVar("--text-muted");
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const lag of [0, 7, 14, 21, 28, 35, 42, 49, 56]) {
        ctx.fillText(lag === 0 ? "0 d" : String(lag), lagToX(lag), plot.top + plot.height + 8);
      }
      drawBaseline(ctx, plot);

      if (hoverLagIdx !== undefined && hoverLagIdx !== null) {
        const x = Math.round(lagToX(ACF.lags[hoverLagIdx])) + 0.5;
        ctx.strokeStyle = cssVar("--axis-line");
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.top + plot.height);
        ctx.stroke();
      }

      const surfaceColor = cssVar("--surface-1");
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      for (const z of zones) {
        const vals = ACF.byZone[z];
        const color = cssVar(ZONE_COLOR_VAR[z]);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ACF.lags.forEach((lag, i) => {
          const x = lagToX(lag), y = valueToY(vals[i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        // Markers at the measured lags, ringed in the surface colour so
        // overlapping zones stay separable.
        ACF.lags.forEach((lag, i) => {
          const x = lagToX(lag), y = valueToY(vals[i]);
          const r = i === hoverLagIdx ? 5 : 4;
          ctx.beginPath();
          ctx.fillStyle = surfaceColor;
          ctx.arc(x, y, r + 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.fillStyle = color;
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      // Direct end labels, de-collided top to bottom with a 13px minimum gap.
      const ends = zones
        .map((z) => ({ z, y: valueToY(ACF.byZone[z][ACF.lags.length - 1]) }))
        .sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) {
        if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
      }
      ctx.font = FONT;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = cssVar("--text-secondary");
      for (const e of ends) ctx.fillText(zoneLabel(e.z), plot.left + plot.width + 10, e.y);

      layout = { plot, lagToX };
    }

    function handleMove(evt) {
      if (!layout) return;
      const p = s.pointer(evt);
      const { plot, lagToX } = layout;
      if (p.x < plot.left - 8 || p.x > plot.left + plot.width + 8 || p.y < plot.top || p.y > plot.top + plot.height) {
        s.hideTooltip();
        render(null);
        return;
      }
      let best = 0;
      ACF.lags.forEach((lag, i) => {
        if (Math.abs(lagToX(lag) - p.x) < Math.abs(lagToX(ACF.lags[best]) - p.x)) best = i;
      });
      render(best);
      const lag = ACF.lags[best];
      s.showTooltip(p.x, p.y, "Lag " + lag + (lag === 1 ? " day" : " days"), zones.map((z) => ({
        name: zoneLabel(z),
        value: ACF.byZone[z][best].toFixed(2),
        color: cssVar(ZONE_COLOR_VAR[z]),
      })));
    }

    function handleLeave() {
      s.hideTooltip();
      render(null);
    }

    s.canvas.addEventListener("mousemove", handleMove);
    s.canvas.addEventListener("mouseleave", handleLeave);
    return { render: () => render(null), wrapEl: s.wrapEl };
  }

  function buildLegend(el) {
    el.innerHTML = "";
    for (const zone of DATASET.zones) {
      const item = document.createElement("span");
      item.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = `var(${ZONE_COLOR_VAR[zone]})`;
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(zoneLabel(zone)));
      el.appendChild(item);
    }
  }

  function render() {
    for (const c of charts) c.render();
  }

  function init(opts) {
    buildLegend(opts.root.querySelector("#study-acf-legend"));

    const amp = "Mean |d|";
    charts = [
      acfChart({ wrap: "study-acf-wrap", canvas: "study-acf-canvas", tooltip: "study-acf-tooltip" }),
      columnChart({
        ids: { wrap: "study-quarter-wrap", canvas: "study-quarter-canvas", tooltip: "study-quarter-tooltip" },
        values: AMP_BY_QUARTER,
        decimals: 2,
        labelIdx: [0, 1, 2, 3],
        seriesName: amp,
        tickLabel: (i) => "q" + (i + 1),
        tipTitle: (i) => "Quarter " + (i + 1) + " (:" + String(i * 15).padStart(2, "0") + "–:" + String(i * 15 + 15).padStart(2, "0") + ")",
      }),
      columnChart({
        ids: { wrap: "study-weekday-wrap", canvas: "study-weekday-canvas", tooltip: "study-weekday-tooltip" },
        values: AMP_BY_WEEKDAY,
        decimals: 2,
        labelIdx: [0, 6],
        seriesName: amp,
        tickLabel: (i) => WEEKDAYS[i],
        tipTitle: (i) => WEEKDAYS[i],
      }),
      columnChart({
        ids: { wrap: "study-hour-wrap", canvas: "study-hour-canvas", tooltip: "study-hour-tooltip" },
        values: AMP_BY_HOUR,
        decimals: 1,
        labelIdx: [3, 9, 17],
        seriesName: amp,
        tickLabel: (i) => (i === 0 || (i + 1) % 3 === 0 ? "h" + (i + 1) : null),
        tipTitle: (i) => "Hour " + (i + 1) + " (" + String(i).padStart(2, "0") + ":00–" + String(i + 1).padStart(2, "0") + ":00)",
      }),
      columnChart({
        ids: { wrap: "study-month-wrap", canvas: "study-month-canvas", tooltip: "study-month-tooltip" },
        values: AMP_BY_MONTH.values,
        decimals: 2,
        labelIdx: [2, 11],
        seriesName: amp,
        tickLabel: (i) => AMP_BY_MONTH.labels[i],
        tipTitle: (i) => AMP_BY_MONTH.labels[i] + " " + AMP_BY_MONTH.years[i],
      }),
    ];

    let resizeTimer = null;
    const ro = new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(render, 150);
    });
    for (const c of charts) ro.observe(c.wrapEl);

    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  }

  return { init, render };
})();
