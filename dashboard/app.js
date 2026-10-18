// State, filter wiring, and bootstrap. Classic script, runs after
// data.js / chart.js / table.js have defined DATASET / Chart / TableView.

(function () {
  const ZONE_COLOR_VAR = {
    ZONE_0: "--series-zone0",
    ZONE_1: "--series-zone1",
    ZONE_2: "--series-zone2",
    ZONE_3: "--series-zone3",
  };

  const dateIndexMap = new Map(DATASET.dates.map((d, i) => [d, i]));
  const lastDayIdx = DATASET.dates.length - 1;

  // Calendar months present in the dataset, in order, each with the first and
  // last *present* day index for that month. Derived by scanning
  // DATASET.dates rather than from calendar arithmetic, so the two missing
  // DST days (2025-10-26, 2026-03-29) simply never become bounds.
  const MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  const months = (function () {
    const out = [];
    let current = null;
    DATASET.dates.forEach((date, i) => {
      const key = date.slice(0, 7); // "YYYY-MM"
      if (!current || current.key !== key) {
        current = { key: key, startDayIdx: i, endDayIdx: i };
        out.push(current);
      } else {
        current.endDayIdx = i;
      }
    });
    return out;
  })();

  const monthByKey = new Map(months.map((m) => [m.key, m]));

  function monthLabel(key) {
    const [year, month] = key.split("-");
    return MONTH_ABBR[Number(month) - 1] + " " + year;
  }
  const QUARTERS_PER_DAY = 96;
  const totalQuarters = DATASET.dates.length * QUARTERS_PER_DAY;
  const lastFlatIndex = totalQuarters - 1;

  const defaultDuration = 96; // 24h
  const defaultStartA = Math.max(0, lastFlatIndex - defaultDuration + 1);
  const defaultStartB = Math.max(0, defaultStartA - 7 * QUARTERS_PER_DAY);

  // Tab id -> extra work to do when that tab becomes visible. Canvas modules
  // size themselves from getBoundingClientRect(), which is 0x0 inside a
  // hidden panel, so a tab with charts must re-render them on activation
  // (the ResizeObservers would too, but only after a visible 150ms blank).
  const TAB_ON_SHOW = {
    data: () => rerenderChart(),
    study: () => Study.render(),
    // History holds the CNN charts and the RevIN charts, which Ensemble draws.
    history: () => {
      Experiment.render();
      Ensemble.render();
    },
    final: () => Ensemble.render(),
  };
  const TAB_IDS = Object.keys(TAB_ON_SHOW);

  const state = {
    activeTab: "data",
    visibleZones: new Set(DATASET.zones),
    startDayIdx: 0,
    endDayIdx: lastDayIdx,
    selectedFlatIndex: null,
    deviationMode: "squared",
    compareStartA: defaultStartA,
    compareStartB: defaultStartB,
    compareDurationQuarters: defaultDuration,
  };

  function pad2(n) {
    return n < 10 ? "0" + n : String(n);
  }

  // Snap an arbitrary ISO date string (which may not exist in DATASET.dates,
  // e.g. one of the 2 DST gap days) to the nearest present date.
  // mode "up" -> first present date >= iso (for range start)
  // mode "down" -> last present date <= iso (for range end)
  function resolveBoundaryDate(iso, mode) {
    if (dateIndexMap.has(iso)) return dateIndexMap.get(iso);
    const dates = DATASET.dates;
    if (mode === "up") {
      for (let i = 0; i < dates.length; i++) if (dates[i] >= iso) return i;
      return lastDayIdx;
    }
    for (let i = dates.length - 1; i >= 0; i--) if (dates[i] <= iso) return i;
    return 0;
  }

  // Convert a flat 15-min dataset index to a "YYYY-MM-DDTHH:MM" string for
  // a datetime-local input's value.
  function flatIndexToDatetimeLocal(idx) {
    const dayIdx = Math.floor(idx / QUARTERS_PER_DAY);
    const rem = idx % QUARTERS_PER_DAY;
    const hour = Math.floor(rem / 4);
    const minute = (rem % 4) * 15;
    return DATASET.dates[dayIdx] + "T" + pad2(hour) + ":" + pad2(minute);
  }

  // Convert a datetime-local value back to a flat 15-min dataset index,
  // snapping the date part to the nearest present date (see
  // resolveBoundaryDate) and rounding the time down to its 15-min quarter.
  function datetimeLocalToFlatIndex(value) {
    const [datePart, timePart] = value.split("T");
    const dayIdx = resolveBoundaryDate(datePart, "up");
    const [hh, mm] = (timePart || "00:00").split(":").map(Number);
    const quarter = Math.floor(mm / 15);
    const idx = dayIdx * QUARTERS_PER_DAY + hh * 4 + quarter;
    return Math.min(Math.max(idx, 0), lastFlatIndex);
  }

  let els = {};

  function clearPresetActive() {
    els.presetButtons.querySelectorAll("button").forEach((b) => b.classList.remove("is-active"));
  }

  function syncDateInputs() {
    els.dateStart.value = DATASET.dates[state.startDayIdx];
    els.dateEnd.value = DATASET.dates[state.endDayIdx];
  }

  // Show a month in the dropdown only when the active range is exactly that
  // month; any other range (preset or hand-picked) falls back to the "—"
  // placeholder rather than claiming a month it only partly covers.
  function syncMonthSelect() {
    const match = months.find(
      (m) => m.startDayIdx === state.startDayIdx && m.endDayIdx === state.endDayIdx
    );
    els.monthSelect.value = match ? match.key : "";
  }

  function applyRange(startDayIdx, endDayIdx) {
    if (startDayIdx > endDayIdx) [startDayIdx, endDayIdx] = [endDayIdx, startDayIdx];
    state.startDayIdx = startDayIdx;
    state.endDayIdx = endDayIdx;
    syncDateInputs();
    syncMonthSelect();
    rerenderChart();
  }

  function rerenderChart() {
    Chart.renderBase();
    Profile.render();
    Deviation.render();
    Compare.render();
  }

  function resyncTable() {
    if (state.selectedFlatIndex !== null) {
      TableView.render(state.selectedFlatIndex, state.visibleZones);
    }
  }

  function buildLegend() {
    els.zoneLegend.innerHTML = "";
    for (const zone of DATASET.zones) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.dataset.zone = zone;

      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = `var(${ZONE_COLOR_VAR[zone]})`;

      btn.appendChild(swatch);
      btn.appendChild(document.createTextNode(zone.replace("ZONE_", "Zone ")));

      btn.addEventListener("click", () => {
        if (state.visibleZones.has(zone)) {
          state.visibleZones.delete(zone);
          btn.classList.add("is-off");
        } else {
          state.visibleZones.add(zone);
          btn.classList.remove("is-off");
        }
        rerenderChart();
        resyncTable();
      });

      els.zoneLegend.appendChild(btn);
    }
  }

  function wirePresets() {
    els.presetButtons.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        clearPresetActive();
        btn.classList.add("is-active");
        const preset = btn.dataset.preset;
        if (preset === "full") applyRange(0, lastDayIdx);
        else if (preset === "30d") applyRange(Math.max(0, lastDayIdx - 29), lastDayIdx);
        else if (preset === "7d") applyRange(Math.max(0, lastDayIdx - 6), lastDayIdx);
      });
    });
  }

  function wireDeviationMode() {
    els.deviationModeGroup.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        els.deviationModeGroup.querySelectorAll("button").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        state.deviationMode = btn.dataset.mode;
        Deviation.render();
      });
    });
  }

  function wireCompareControls() {
    els.compareStartA.value = flatIndexToDatetimeLocal(state.compareStartA);
    els.compareStartB.value = flatIndexToDatetimeLocal(state.compareStartB);
    const minVal = flatIndexToDatetimeLocal(0);
    const maxVal = flatIndexToDatetimeLocal(lastFlatIndex);
    for (const el of [els.compareStartA, els.compareStartB]) {
      el.min = minVal;
      el.max = maxVal;
    }

    els.compareStartA.addEventListener("change", () => {
      state.compareStartA = datetimeLocalToFlatIndex(els.compareStartA.value);
      Compare.render();
    });
    els.compareStartB.addEventListener("change", () => {
      state.compareStartB = datetimeLocalToFlatIndex(els.compareStartB.value);
      Compare.render();
    });

    els.compareDurationGroup.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        els.compareDurationGroup.querySelectorAll("button").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        state.compareDurationQuarters = Number(btn.dataset.hours) * 4;
        Compare.render();
      });
    });
  }

  // A hash is either a tab id (#study) or an anchor inside a tab panel
  // (#study-zones); resolve it to the tab that must be shown. Anything else
  // falls back to the first tab.
  function tabForHash(hash) {
    if (TAB_ON_SHOW[hash]) return hash;
    const target = hash ? document.getElementById(hash) : null;
    const panel = target ? target.closest(".tab-panel") : null;
    return panel ? panel.id.replace(/^tab-/, "") : TAB_IDS[0];
  }

  function showTab(tabId) {
    if (!TAB_ON_SHOW[tabId]) tabId = TAB_IDS[0];
    state.activeTab = tabId;
    els.tabBar.querySelectorAll("button[data-tab]").forEach((btn) => {
      const active = btn.dataset.tab === tabId;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    for (const id of TAB_IDS) {
      document.getElementById("tab-" + id).hidden = id !== tabId;
    }
    TAB_ON_SHOW[tabId]();
  }

  // The active tab lives in location.hash (#data, #study, or an anchor
  // inside a tab) so a reload or a shared link keeps it; hash changes work
  // over file://.
  function wireTabs() {
    els.tabBar.querySelectorAll("button[data-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (location.hash !== "#" + btn.dataset.tab) location.hash = btn.dataset.tab;
        else showTab(btn.dataset.tab);
      });
    });
    window.addEventListener("hashchange", () => openHash(location.hash.slice(1)));
  }

  // Show the tab a hash belongs to; for an in-panel anchor, scroll to it once
  // its panel is visible (the browser's own anchor jump can't reach a
  // hidden panel).
  function openHash(hash) {
    const tabId = tabForHash(hash);
    showTab(tabId);
    if (hash && hash !== tabId) {
      const target = document.getElementById(hash);
      if (target) target.scrollIntoView();
    }
  }

  function wireMonthSelect() {
    for (const month of months) {
      const opt = document.createElement("option");
      opt.value = month.key;
      opt.textContent = monthLabel(month.key);
      els.monthSelect.appendChild(opt);
    }
    syncMonthSelect();

    els.monthSelect.addEventListener("change", () => {
      const month = monthByKey.get(els.monthSelect.value);
      if (!month) {
        // Placeholder re-picked: leave the range alone, just restore the label.
        syncMonthSelect();
        return;
      }
      clearPresetActive();
      applyRange(month.startDayIdx, month.endDayIdx);
    });
  }

  function wireDateInputs() {
    els.dateStart.min = DATASET.dates[0];
    els.dateStart.max = DATASET.dates[lastDayIdx];
    els.dateEnd.min = DATASET.dates[0];
    els.dateEnd.max = DATASET.dates[lastDayIdx];
    syncDateInputs();

    els.dateStart.addEventListener("change", () => {
      clearPresetActive();
      const idx = resolveBoundaryDate(els.dateStart.value, "up");
      applyRange(idx, state.endDayIdx);
    });
    els.dateEnd.addEventListener("change", () => {
      clearPresetActive();
      const idx = resolveBoundaryDate(els.dateEnd.value, "down");
      applyRange(state.startDayIdx, idx);
    });
  }

  function init() {
    els = {
      tabBar: document.getElementById("tab-bar"),
      presetButtons: document.getElementById("preset-buttons"),
      zoneLegend: document.getElementById("zone-legend"),
      monthSelect: document.getElementById("month-select"),
      dateStart: document.getElementById("date-start"),
      dateEnd: document.getElementById("date-end"),
      chartWrap: document.getElementById("chart-wrap"),
      baseCanvas: document.getElementById("chart-base"),
      overlayCanvas: document.getElementById("chart-overlay"),
      tooltip: document.getElementById("tooltip"),
      emptyState: document.getElementById("empty-state"),
      tableCard: document.getElementById("table-card"),
      tableHeadRow: document.getElementById("table-head-row"),
      tableBody: document.getElementById("table-body"),
      tableTime: document.getElementById("table-time"),
      tableClose: document.getElementById("table-close"),
      profileWrap: document.getElementById("profile-wrap"),
      profileCanvas: document.getElementById("profile-canvas"),
      profileTooltip: document.getElementById("profile-tooltip"),
      deviationModeGroup: document.getElementById("deviation-mode-group"),
      deviationWrap: document.getElementById("deviation-wrap"),
      deviationBase: document.getElementById("deviation-base"),
      deviationOverlay: document.getElementById("deviation-overlay"),
      deviationTooltip: document.getElementById("deviation-tooltip"),
      compareStartA: document.getElementById("compare-start-a"),
      compareStartB: document.getElementById("compare-start-b"),
      compareDurationGroup: document.getElementById("compare-duration-group"),
      compareWrap: document.getElementById("compare-wrap"),
      compareCanvas: document.getElementById("compare-canvas"),
      compareTooltip: document.getElementById("compare-tooltip"),
      compareHint: document.getElementById("compare-hint"),
    };

    wireTabs();
    buildLegend();
    wirePresets();
    wireDeviationMode();
    wireCompareControls();
    wireMonthSelect();
    wireDateInputs();

    TableView.init({
      cardEl: els.tableCard,
      headRowEl: els.tableHeadRow,
      bodyEl: els.tableBody,
      timeEl: els.tableTime,
      closeBtn: els.tableClose,
      onClose: () => { state.selectedFlatIndex = null; },
    });

    Chart.init({
      baseCanvas: els.baseCanvas,
      overlayCanvas: els.overlayCanvas,
      tooltipEl: els.tooltip,
      emptyStateEl: els.emptyState,
      getState: () => state,
      onPointClick: (rawFlatIndex) => {
        state.selectedFlatIndex = rawFlatIndex;
        TableView.render(rawFlatIndex, state.visibleZones);
      },
    });

    Profile.init({
      canvas: els.profileCanvas,
      tooltipEl: els.profileTooltip,
      wrapEl: els.profileWrap,
      getState: () => state,
    });

    Deviation.init({
      baseCanvas: els.deviationBase,
      overlayCanvas: els.deviationOverlay,
      tooltipEl: els.deviationTooltip,
      wrapEl: els.deviationWrap,
      getState: () => state,
    });

    Compare.init({
      canvas: els.compareCanvas,
      tooltipEl: els.compareTooltip,
      wrapEl: els.compareWrap,
      hintEl: els.compareHint,
      getState: () => state,
    });

    Study.init({ root: document.getElementById("tab-study") });
    Experiment.init({ root: document.getElementById("tab-history") });
    Ensemble.init({ root: document.getElementById("tab-final") });

    openHash(location.hash.slice(1));
  }

  document.addEventListener("DOMContentLoaded", init);
})();
