// "Raw-hourly CNN" tab: static report of the experiment in
// ../ch-PHDs-but-not-for-long/. The figures below are transcribed from that
// repo's committed results (experiments/*/run_info.json and *.csv), not
// computed here, so they don't depend on the dashboard filters - update them
// when the experiment is rerun. Classic script - exposes a single global
// `Experiment`.

const Experiment = (function () {
  const ZONE_COLOR_VAR = {
    ZONE_0: "--series-zone0",
    ZONE_1: "--series-zone1",
    ZONE_2: "--series-zone2",
    ZONE_3: "--series-zone3",
  };
  const FONT = "11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace";

  // raw_hourly_only_zonal_cnn/validation_metrics.csv - Aug-Sep MAE by quarter.
  const QUARTER_MAE = {
    labels: ["q1", "q2", "q3", "q4"],
    byZone: {
      ZONE_0: [4.487, 2.926, 2.677, 4.870],
      ZONE_1: [7.885, 4.830, 4.539, 8.202],
      ZONE_2: [6.843, 4.628, 4.559, 7.037],
      ZONE_3: [5.638, 3.435, 3.449, 5.804],
    },
  };

  // month_difficulty_no_month_embedding_aug_sep/monthly_oof_difficulty.csv -
  // q-weighted MAE of an 11-month CNN on each held-out month (MAE objective).
  // Aug and Sep (the last two) are diagnostic only: they never set a weight.
  const MONTH_DIFFICULTY = {
    labels: ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
    years: ["2025", "2025", "2025", "2026", "2026", "2026", "2026", "2026", "2026", "2026", "2026", "2026"],
    contextFrom: 10,
    byZone: {
      ZONE_0: [4.65, 3.82, 2.71, 2.74, 2.25, 4.41, 3.91, 3.66, 3.80, 3.04, 3.52, 4.68],
      ZONE_1: [6.08, 3.91, 3.35, 5.33, 3.24, 5.94, 5.22, 4.96, 4.64, 4.76, 5.58, 8.13],
      ZONE_2: [3.49, 2.74, 2.14, 2.88, 2.08, 3.40, 3.94, 4.01, 3.78, 4.59, 5.37, 5.96],
      ZONE_3: [5.33, 4.22, 3.53, 3.21, 5.07, 5.65, 5.57, 4.57, 4.90, 4.27, 4.69, 4.81],
    },
  };

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
  // tooltip placement, and pointer coordinates. Same as study.js's.
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

  // One line per zone over an ordered categorical x axis, in the zone's fixed
  // hue. Points from `contextFrom` on are contextual (not part of the fit):
  // that stretch is drawn dotted and faded, the same-hue secondary style.
  // Lines end in direct labels, de-collided with a 13px minimum gap.
  function zoneLineChart(spec) {
    const s = makeSurface(spec.ids);
    const MARGIN = { top: 14, right: 64, bottom: 26, left: 44 };
    const zones = DATASET.zones;
    const n = spec.labels.length;
    const contextFrom = spec.contextFrom === undefined ? n : spec.contextFrom;
    let layout = null;

    function render(hoverIdx) {
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
      for (const z of zones) for (const v of spec.byZone[z]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const { ticks, niceMin, niceMax } = niceTicks(lo, hi, 4);
      const band = plot.width / n;
      const idxToX = (i) => plot.left + band * (i + 0.5);
      const valueToY = (v) => plot.top + (1 - (v - niceMin) / (niceMax - niceMin)) * plot.height;

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
        ctx.fillText(t.toFixed(spec.tickDecimals), plot.left - 10, y);
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i < n; i++) {
        ctx.fillStyle = cssVar("--text-muted");
        ctx.fillText(spec.labels[i], idxToX(i), plot.top + plot.height + 8);
      }
      ctx.strokeStyle = cssVar("--axis-line");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plot.left, plot.top + plot.height + 0.5);
      ctx.lineTo(plot.left + plot.width, plot.top + plot.height + 0.5);
      ctx.stroke();

      if (hoverIdx !== null && hoverIdx !== undefined) {
        const x = Math.round(idxToX(hoverIdx)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.top + plot.height);
        ctx.stroke();
      }

      const surfaceColor = cssVar("--surface-1");
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      for (const z of zones) {
        const vals = spec.byZone[z];
        const color = cssVar(ZONE_COLOR_VAR[z]);
        ctx.strokeStyle = color;

        // Primary stretch: solid 2px.
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < contextFrom; i++) {
          if (i === 0) ctx.moveTo(idxToX(i), valueToY(vals[i]));
          else ctx.lineTo(idxToX(i), valueToY(vals[i]));
        }
        ctx.stroke();

        // Contextual stretch: dotted 1.5px, faded, continuing from the last
        // primary point.
        if (contextFrom < n) {
          ctx.save();
          ctx.lineWidth = 1.5;
          ctx.setLineDash([1, 4]);
          ctx.globalAlpha = 0.55;
          ctx.beginPath();
          const start = Math.max(0, contextFrom - 1);
          ctx.moveTo(idxToX(start), valueToY(vals[start]));
          for (let i = start + 1; i < n; i++) ctx.lineTo(idxToX(i), valueToY(vals[i]));
          ctx.stroke();
          ctx.restore();
        }

        // Markers, ringed in the surface colour so overlapping zones stay
        // separable.
        for (let i = 0; i < n; i++) {
          const x = idxToX(i), y = valueToY(vals[i]);
          const r = i === hoverIdx ? 5 : 4;
          ctx.globalAlpha = i >= contextFrom ? 0.55 : 1;
          ctx.beginPath();
          ctx.fillStyle = surfaceColor;
          ctx.arc(x, y, r + 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.fillStyle = color;
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }

      const ends = zones
        .map((z) => ({ z, y: valueToY(spec.byZone[z][n - 1]) }))
        .sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) {
        if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
      }
      ctx.font = FONT;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = cssVar("--text-secondary");
      for (const e of ends) ctx.fillText(zoneLabel(e.z), idxToX(n - 1) + 14, e.y);

      layout = { plot, band };
    }

    function handleMove(evt) {
      if (!layout) return;
      const p = s.pointer(evt);
      const { plot, band } = layout;
      if (p.x < plot.left || p.x > plot.left + plot.width || p.y < plot.top || p.y > plot.top + plot.height) {
        s.hideTooltip();
        render(null);
        return;
      }
      const idx = Math.min(n - 1, Math.floor((p.x - plot.left) / band));
      render(idx);
      s.showTooltip(p.x, p.y, spec.tipTitle(idx), zones.map((z) => ({
        name: zoneLabel(z),
        value: spec.byZone[z][idx].toFixed(2),
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
    opts.root.querySelectorAll(".cnn-zone-legend").forEach(buildLegend);

    charts = [
      zoneLineChart({
        ids: { wrap: "cnn-quarter-wrap", canvas: "cnn-quarter-canvas", tooltip: "cnn-quarter-tooltip" },
        labels: QUARTER_MAE.labels,
        byZone: QUARTER_MAE.byZone,
        tickDecimals: 0,
        tipTitle: (i) => "Quarter " + (i + 1) + " (:" + String(i * 15).padStart(2, "0") + "–:" + String(i * 15 + 15).padStart(2, "0") + ")",
      }),
      zoneLineChart({
        ids: { wrap: "cnn-month-wrap", canvas: "cnn-month-canvas", tooltip: "cnn-month-tooltip" },
        labels: MONTH_DIFFICULTY.labels,
        byZone: MONTH_DIFFICULTY.byZone,
        contextFrom: MONTH_DIFFICULTY.contextFrom,
        tickDecimals: 0,
        tipTitle: (i) => MONTH_DIFFICULTY.labels[i] + " " + MONTH_DIFFICULTY.years[i] +
          (i >= MONTH_DIFFICULTY.contextFrom ? " (diagnostic only)" : ""),
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
