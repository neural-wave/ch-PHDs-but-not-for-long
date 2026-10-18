// Click-to-table rendering: shows a window of 15-minute readings around a
// clicked timestamp. Classic script — exposes a single global `TableView`.

const TableView = (function () {
  const ZONE_COLOR_VAR = {
    ZONE_0: "--series-zone0",
    ZONE_1: "--series-zone1",
    ZONE_2: "--series-zone2",
    ZONE_3: "--series-zone3",
  };
  const QUARTERS_PER_DAY = 96;
  const WINDOW = 4; // +/- 4 quarters (~1 hour either side), 9 rows total

  let cardEl, headRowEl, bodyEl, timeEl, closeBtn;

  function pad2(n) {
    return n < 10 ? "0" + n : String(n);
  }

  function flatIndexToMeta(idx) {
    const dayIdx = Math.floor(idx / QUARTERS_PER_DAY);
    const rem = idx % QUARTERS_PER_DAY;
    const hour = Math.floor(rem / 4) + 1;
    const quarter = (rem % 4) + 1;
    return { dayIdx, hour, quarter };
  }

  function formatTime(meta) {
    return pad2(meta.hour - 1) + ":" + pad2((meta.quarter - 1) * 15);
  }

  function formatDateTime(meta) {
    return DATASET.dates[meta.dayIdx] + " " + formatTime(meta);
  }

  function buildHeader(zones) {
    headRowEl.innerHTML = "";
    const timeTh = document.createElement("th");
    timeTh.textContent = "Time";
    headRowEl.appendChild(timeTh);
    for (const z of zones) {
      const th = document.createElement("th");
      const swatch = document.createElement("span");
      swatch.style.display = "inline-block";
      swatch.style.width = "8px";
      swatch.style.height = "8px";
      swatch.style.borderRadius = "50%";
      swatch.style.marginRight = "6px";
      swatch.style.background = getComputedStyle(document.documentElement)
        .getPropertyValue(ZONE_COLOR_VAR[z]).trim();
      th.appendChild(swatch);
      th.appendChild(document.createTextNode(z.replace("ZONE_", "Zone ")));
      headRowEl.appendChild(th);
    }
  }

  function render(centerIdx, visibleZones) {
    const zones = DATASET.zones.filter((z) => visibleZones.has(z));
    const N = DATASET.dates.length * QUARTERS_PER_DAY;
    const centerMeta = flatIndexToMeta(centerIdx);

    timeEl.textContent = formatDateTime(centerMeta);
    buildHeader(zones);
    bodyEl.innerHTML = "";

    if (zones.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 1;
      td.textContent = "No zones selected.";
      td.style.color = "var(--text-muted)";
      tr.appendChild(td);
      bodyEl.appendChild(tr);
      cardEl.hidden = false;
      return;
    }

    const lo = Math.max(0, centerIdx - WINDOW);
    const hi = Math.min(N - 1, centerIdx + WINDOW);

    for (let idx = lo; idx <= hi; idx++) {
      const meta = flatIndexToMeta(idx);
      const tr = document.createElement("tr");
      if (idx === centerIdx) tr.className = "is-selected";

      const timeTd = document.createElement("td");
      timeTd.textContent = DATASET.dates[meta.dayIdx] + " " + formatTime(meta);
      tr.appendChild(timeTd);

      for (const z of zones) {
        const td = document.createElement("td");
        td.textContent = DATASET.prices[z][idx].toFixed(2);
        tr.appendChild(td);
      }
      bodyEl.appendChild(tr);
    }

    cardEl.hidden = false;
  }

  function hide() {
    cardEl.hidden = true;
  }

  function init(opts) {
    cardEl = opts.cardEl;
    headRowEl = opts.headRowEl;
    bodyEl = opts.bodyEl;
    timeEl = opts.timeEl;
    closeBtn = opts.closeBtn;
    closeBtn.addEventListener("click", () => {
      hide();
      if (opts.onClose) opts.onClose();
    });
  }

  return { init, render, hide };
})();
