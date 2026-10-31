const heroEl = document.getElementById("hero");
const racksEl = document.getElementById("racks");
const countEl = document.getElementById("rack-count");
const keyEl = document.getElementById("map-key");
const noteEl = document.getElementById("note");
const updatedEl = document.getElementById("updated");
const refreshIco = document.getElementById("refresh-ico");
const mapEl = document.getElementById("map");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const twoPanes = window.matchMedia("(min-width: 1024px)"); // as in user.css

const icon = (paths, cls = "ico") => `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${paths}</svg>`;
const ARROW_IN = '<path d="M21 12H5M10 7l-5 5 5 5"/>';
const ICONS = {
  search: icon('<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.4-4.4"/>'),
  locate: icon('<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2" fill="currentColor" stroke="none"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3"/>'),
  go: icon('<path d="M5 12h14M13 6l6 6-6 6"/>', "ico btn__go"),
  chev: icon('<path d="M6 9l6 6 6-6"/>'),
  entry: icon(ARROW_IN, "ico plan__entry"),
  entryKey: icon(ARROW_IN),
  plus: icon('<path d="M12 6v12M6 12h12"/>'),
  minus: icon('<path d="M6 12h12"/>'),
};
/* Every chip has its own icon shape, so the state never depends on colour */
const CHIP_ICONS = {
  live: '<circle class="ring" cx="6" cy="6" r="3" fill="currentColor"/><circle cx="6" cy="6" r="5.5" fill="currentColor" opacity="0.18"/><circle cx="6" cy="6" r="3" fill="currentColor"/>',
  full: '<circle cx="6" cy="6" r="4.9" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M3.6 6h4.8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
  starting: '<circle cx="6" cy="6" r="4.6" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"/><path class="arc" d="M6 1.4a4.6 4.6 0 0 1 4.6 4.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
  stale: '<circle cx="6" cy="6" r="5.6" fill="currentColor"/><path class="hands" d="M6 3.3V6l1.9 1.2" fill="none" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>',
  none: '<circle cx="6" cy="6" r="5" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M4.6 4.7a1.45 1.45 0 1 1 2 1.35c-.42.18-.6.5-.6.95" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/><circle cx="6" cy="8.7" r="0.75" fill="currentColor"/>',
  demo: '<path d="M4.4 1.3h3.2M5 1.3v3.3L2.3 9.3a.9.9 0 0 0 .8 1.4h5.8a.9.9 0 0 0 .8-1.4L7 4.6V1.3" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>',
  real: '<path d="M1.2 3.6h2l.9-1.4h3.8l.9 1.4h2v6.2H1.2z" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/><circle cx="6" cy="6.6" r="1.7" fill="none" stroke="currentColor" stroke-width="1.2"/>',
};
const chip = (kind, label) => `<span class="chip chip--${kind}"><svg class="chip__i" viewBox="0 0 12 12" aria-hidden="true">${CHIP_ICONS[kind]}</svg>${esc(label)}</span>`;
const DEMO_CHIP = chip("demo", "Demo data");
const REAL_CHIP = chip("real", "Real camera");
const CAMERA_MARK = `<svg class="pin__cam" viewBox="0 0 12 12" aria-hidden="true">${CHIP_ICONS.real}</svg>`;
let anyDemo = false; // "Real camera" only means something next to demo racks

function sourceChip(rack) {
  return rack.demo ? DEMO_CHIP : anyDemo ? REAL_CHIP : "";
}

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function agoText(seconds) {
  return seconds < 60 ? `${Math.round(seconds)} s ago` : `${Math.round(seconds / 60)} min ago`;
}

function ageText(rack) {
  const seen = known(rack.age_s);
  if (rack.status === "ok") return seen ? `Updated ${agoText(rack.age_s)}` : "Updated";
  if (rack.status === "starting") return "Waiting for first count";
  if (rack.status === "stale" && seen) return `Last update ${agoText(rack.age_s)}`;
  return "No count received yet";
}

function chipFor(rack) {
  if (rack.status === "ok" && rack.free === 0) return ["full", "Full"];
  if (rack.status === "ok") return ["live", "Live"];
  if (rack.status === "starting") return ["starting", "Starting"];
  if (rack.status === "stale") return ["stale", rack.camera_too_hot ? "Camera too hot" : "No recent data"];
  return ["none", "No data"];
}

/* Rack plan as a grid of cells: one grid row per rack row with an aisle between rows, each space at its place
   from the layout, the entrance arrow on its end. Without a layout the spaces simply wrap. */
function planHtml(rack, changed = [], small = false) {
  const live = rack.status === "ok";
  const { layout } = rack;
  const seen = new Map(rack.slots.map((s, i) => [s.id, [s.state, changed[i]]]));
  const spaces = layout ? layout.spaces : Array.from({ length: rack.capacity }, (_, i) => ({ id: i + 1 }));
  const cells = spaces.map((s) => {
    const [state, flipped] = (live && seen.get(s.id)) || ["unknown", false];
    const place = layout ? ` style="grid-area: ${2 * Math.round(s.y) + 1} / ${Math.round(s.x) + 1} / span ${2 * Math.round(s.h) - 1} / span ${Math.round(s.w)}"` : "";
    return `<span class="cell cell--${state}${flipped ? " is-changed" : ""}"${place} title="Space ${s.id}: ${state}"><span>${s.id}</span></span>`;
  });
  const rows = layout ? Math.ceil(layout.height) : 0;
  for (let y = 0; y < rows - 1; y++) {
    cells.push(`<span class="plan__aisle" style="grid-row: ${2 * y + 2}">${y === Math.floor((rows - 2) / 2) ? '<span class="plan__door"></span>' : ""}</span>`);
  }
  const cols = layout ? Math.ceil(layout.width) : Math.min(spaces.length, 12);
  const wrap = `plan-wrap${small ? " plan-wrap--sm" : ""}${!layout ? " plan-wrap--free" : layout.entrance === "start" ? " plan-wrap--start" : ""}`;
  const label = live ? `${rack.free} of ${rack.capacity} spaces free` : "Rack plan, no live data";
  return `<div class="${wrap}" role="img" aria-label="${label}"><div class="plan${small ? " plan--sm" : ""}${cols > 10 ? " plan--dense" : ""}" style="grid-template-columns: repeat(${cols}, minmax(0, 1fr))">${cells.join("")}</div>${layout ? ICONS.entry : ""}</div>`;
}

function freeSpacesText(rack) {
  const runs = [];
  rack.slots.filter((s) => s.state === "free").forEach((s) => {
    const run = runs[runs.length - 1];
    if (run && s.id === run[1] + 1) run[1] = s.id;
    else runs.push([s.id, s.id]);
  });
  if (!runs.length) return "No free space right now";
  return `Free spaces: <b>${runs.map(([a, b]) => (a === b ? `${a}` : b === a + 1 ? `${a}, ${b}` : `${a} to ${b}`)).join(", ")}</b>`;
}

const PLAN_KEY = `<div class="plan-key" aria-hidden="true"><span class="plan-key__i"><span class="cell cell--free"></span>Free</span><span class="plan-key__i"><span class="cell cell--taken"></span>Taken</span>`;

/* The open tile's body: the plan, the free spaces and the key (the predictions are added below by placeForecast) */
function bodyHtml(rack, changed) {
  const live = rack.status === "ok";
  const plan = rack.layout || (live && rack.slots.length) ? planHtml(rack, changed) : "";
  const line = live && rack.slots.length
    ? `<p class="plan-free">${freeSpacesText(rack)}</p>`
    : `<p class="plan-free plan-free--quiet">${live ? "This rack reports a count only." : "No live data for the spaces right now."}</p>`;
  const key = plan ? `${PLAN_KEY}${rack.layout ? `<span class="plan-key__i">${ICONS.entryKey}Entrance</span>` : ""}</div>` : "";
  return `${plan}${line}${key}`;
}

const BIKE_M_PER_MIN = 250; // 15 km/h
const WALK_M_PER_MIN = 5000 / 60;
const EARTH_M = 6371000;
let targetPos = null; // {lat, lng}: the chosen destination or the browser's position; never sent anywhere

function distanceM(a, b) {
  const rad = Math.PI / 180;
  const h = Math.sin(((b.lat - a.lat) * rad) / 2) ** 2
    + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(((b.lng - a.lng) * rad) / 2) ** 2;
  return 2 * EARTH_M * Math.asin(Math.sqrt(h));
}

function distanceText(metres) {
  const rounded = Math.max(10, Math.round(metres / 10) * 10);
  return rounded < 1000 ? `${rounded} m` : `${(metres / 1000).toFixed(1)} km`;
}

function metaHtml(rack) {
  const extra = [];
  if (targetPos && rack.location) extra.push(distanceText(distanceM(targetPos, rack.location)));
  if (rack.forecast && !rack.forecast.learning && rack.forecast.usually_full_by) extra.push(`full by ${rack.forecast.usually_full_by}`);
  return `${esc(ageText(rack))}${extra.length ? `<span class="tile__extra">${esc(extra.join(", "))}</span>` : ""}`;
}

/* Small DOM helpers: content is only replaced when it changed, so focus and running transitions survive the 5 s refresh */
function setHtml(el, html) {
  if (el.dataset.html === html) return;
  el.innerHTML = html;
  el.dataset.html = html;
}

function fromHtml(html) {
  const box = document.createElement("template");
  box.innerHTML = html.trim();
  return box.content.firstElementChild;
}

function restart(el, className) {
  el.classList.remove(className);
  void el.getBoundingClientRect(); // a reflow, so the animation runs again
  el.classList.add(className);
}

/* A number that changes slides in from below when it rises and from above when it drops */
function roll(el, value) {
  const text = String(value);
  if (el.textContent === text) return;
  const before = Number(el.textContent);
  const first = el.textContent === "";
  el.textContent = text;
  el.classList.remove("num--up", "num--down");
  if (first || !Number.isFinite(before) || !Number.isFinite(Number(value))) return;
  restart(el, Number(value) > before ? "num--up" : "num--down");
}

document.addEventListener("animationend", (event) => {
  const el = event.target;
  if (!el.classList) return;
  if (el.classList.contains("is-entering")) el.classList.remove("is-entering"); // so the hover lift can move the pin again
  else if (el.classList.contains("enter")) el.classList.remove("enter");
  else if (el.classList.contains("ring")) el.closest(".is-ping")?.classList.remove("is-ping");
});

/* Rack tiles: one per rack, built once and then updated part by part. The open tile is the selected rack. */
const tiles = new Map();
const lastSlotStates = new Map();
let selectedId = null;
let userClosed = false; // the open tile was closed on purpose: do not open the default one again
let tilesBuilt = false;

function makeTile(rack, index) {
  const id = `tile-${String(rack.id).replace(/\W/g, "_")}`;
  const li = fromHtml(`<li><article class="tile enter" style="--i: ${Math.min(index + 3, 6)}" data-rack="${esc(rack.id)}">
    <button class="tile__head" type="button" aria-expanded="false" aria-controls="${id}">
      <span class="tile__title"></span><span class="chip"></span>
      <span class="count"><span class="count__n"></span><span class="count__of"></span></span>
      <span class="tile__chev" aria-hidden="true">${ICONS.chev}</span>
      <span class="bar" aria-hidden="true"><span class="bar__fill" style="width: 0%"></span></span>
      <span class="tile__meta"></span>
    </button>
    <div class="tile__body" id="${id}" inert><div class="tile__inner"><div class="tile__plan"><div class="tile__live"></div></div></div></div>
  </article></li>`);
  const part = (selector) => li.querySelector(selector);
  return {
    li, article: part(".tile"), head: part(".tile__head"), title: part(".tile__title"), chip: part(".tile__head > .chip"),
    count: part(".count"), countN: part(".count__n"), countOf: part(".count__of"), fill: part(".bar__fill"),
    meta: part(".tile__meta"), body: part(".tile__body"), plan: part(".tile__plan"), live: part(".tile__live"),
  };
}

function updateTile(tile, rack, changed) {
  const live = rack.status === "ok";
  const open = rack.id === selectedId;
  const [kind, label] = chipFor(rack);
  ["full", "live", "starting", "stale", "none"].forEach((k) => tile.article.classList.toggle(`tile--${k}`, k === kind));
  tile.article.classList.toggle("is-open", open);
  tile.head.setAttribute("aria-expanded", String(open));
  tile.body.inert = !open;
  setHtml(tile.title, `<span class="tile__name">${esc(rack.name)}</span>${sourceChip(rack)}`);
  const chipNow = chip(kind, label);
  if (tile.chipHtml !== chipNow) {
    const fresh = fromHtml(chipNow);
    tile.chip.replaceWith(fresh);
    tile.chip = fresh;
    tile.chipHtml = chipNow;
  }
  if (kind === "live" && known(tile.lastAge) && known(rack.age_s) && rack.age_s < tile.lastAge) restart(tile.chip, "is-ping"); // a fresh reading
  tile.lastAge = rack.age_s;
  tile.count.classList.toggle("count--unknown", !live);
  roll(tile.countN, live ? rack.free : "?");
  tile.countOf.textContent = `free of ${rack.capacity}`;
  tile.fill.style.width = live ? `${Math.round(((rack.capacity - rack.free) / Math.max(1, rack.capacity)) * 100)}%` : "0%";
  setHtml(tile.meta, metaHtml(rack));
  setHtml(tile.live, bodyHtml(rack, changed));
}

function renderMeta() {
  lastRacks.forEach((rack) => tiles.has(rack.id) && setHtml(tiles.get(rack.id).meta, metaHtml(rack)));
}

/* Predictions live in the open tile only; a new selection moves them and loads them again */
let forecastData = null;
let forecastHtml = "";

function placeForecast() {
  const tile = tiles.get(selectedId);
  const current = document.getElementById("forecast");
  if (current && tile && tile.plan.contains(current)) return false;
  current?.remove();
  forecastData = null;
  forecastHtml = "";
  if (!tile) return false;
  tile.plan.insertAdjacentHTML("beforeend", '<section class="forecast" id="forecast" aria-label="Predictions" data-state="loading"><p class="forecast-quiet">Loading predictions</p></section>');
  return true;
}

function defaultRackId(racks) {
  const live = racks.filter((r) => r.status === "ok");
  return live.length ? live.reduce((a, b) => (b.free > a.free ? b : a)).id : racks[0].id;
}

function renderRacks(racks) {
  if (!tilesBuilt) {
    racksEl.innerHTML = "";
    racksEl.removeAttribute("aria-busy");
    tilesBuilt = true;
  }
  countEl.textContent = racks.length;
  const ids = new Set(racks.map((r) => r.id));
  tiles.forEach((tile, id) => {
    if (ids.has(id)) return;
    tile.li.remove();
    tiles.delete(id);
  });
  if (!racks.length) {
    racksEl.innerHTML = '<li class="list-empty">No racks configured.</li>';
    selectedId = null;
    return;
  }
  racksEl.querySelector(".list-empty")?.remove();
  if (selectedId !== null && !ids.has(selectedId)) selectedId = null;
  if (selectedId === null && !userClosed) selectedId = defaultRackId(racks);
  racks.forEach((rack, index) => {
    const before = lastSlotStates.get(rack.id) || [];
    const changed = rack.slots.map((s, i) => before.length > 0 && before[i] !== s.state);
    lastSlotStates.set(rack.id, rack.slots.map((s) => s.state));
    let tile = tiles.get(rack.id);
    if (!tile) {
      tile = makeTile(rack, index);
      racksEl.appendChild(tile.li);
      tiles.set(rack.id, tile);
    }
    updateTile(tile, rack, changed);
  });
  if (placeForecast()) loadForecast();
}

/* Every way of picking a rack (tile, pin, card, "Most space", destination) ends here */
function selectRack(id, fly = true) {
  const rack = lastRacks.find((r) => r.id === id);
  if (!rack) return;
  selectedId = id;
  userClosed = false;
  if (fly && map && rack.location) flyToRack(rack, Math.max(map.getZoom(), DETAIL_ZOOM + 1));
  renderRacks(lastRacks);
  updateMarkers(lastRacks);
  if (twoPanes.matches) tiles.get(id)?.article.scrollIntoView({ block: "nearest", behavior: reduceMotion.matches ? "auto" : "smooth" });
}

racksEl.addEventListener("click", (event) => {
  const head = event.target.closest(".tile__head");
  if (!head) return;
  const id = head.closest(".tile").dataset.rack;
  if (id !== selectedId) return selectRack(id);
  selectedId = null; // the open tile closes, and with it the selection
  userClosed = true;
  renderRacks(lastRacks);
  updateMarkers(lastRacks);
});

/* Predictions for the selected rack from its history: fetched on selection and every 30 s. A missing or
   failing endpoint only gives a quiet line. */
const CHART_FROM = 6 * 60;
const CHART_TO = 20 * 60;
let forecastToken = 0;

function known(value) {
  return value !== null && value !== undefined;
}

function minutesOf(t) {
  const [h, m] = String(t).split(":").map(Number);
  return h * 60 + m;
}

function chartSvg(data, width, day, animate) {
  const height = 160;
  const [left, right, top, bottom] = [30, 12, 18, 24];
  const cap = Math.max(1, Number(data.capacity) || 1);
  const x = (t) => (left + ((minutesOf(t) - CHART_FROM) / (CHART_TO - CHART_FROM)) * (width - left - right)).toFixed(1);
  const y = (free) => (top + (1 - Math.min(cap, Math.max(0, free)) / cap) * (height - top - bottom)).toFixed(1);
  const inDay = (p) => p && known(p.t) && minutesOf(p.t) >= CHART_FROM && minutesOf(p.t) <= CHART_TO;
  const base = y(0);
  const runs = [[]]; // the typical day split where there is no data, so gaps stay gaps
  (Array.isArray(data.typical) ? data.typical : []).filter(inDay).forEach((p) => {
    if (known(p.free)) runs[runs.length - 1].push(p);
    else if (runs[runs.length - 1].length) runs.push([]);
  });
  const typical = runs
    .filter((run) => run.length > 1)
    .map((run) => `M${x(run[0].t)} ${base}${run.map((p) => `L${x(p.t)} ${y(p.free)}`).join("")}L${x(run[run.length - 1].t)} ${base}Z`)
    .join("");
  const today = (Array.isArray(data.today) ? data.today : []).filter((p) => inDay(p) && known(p.free));
  const line = today.length > 1 ? `M${today.map((p) => `${x(p.t)} ${y(p.free)}`).join("L")}` : "";
  const ticks = [];
  for (let hour = 6; hour <= 20; hour += 2) {
    const t = `${String(hour).padStart(2, "0")}:00`;
    ticks.push(`<line class="chart-grid" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${base}"/><text class="chart-tick" x="${x(t)}" y="${height - 6}" text-anchor="middle">${t}</text>`);
  }
  let now = "";
  if (inDay({ t: data.now })) {
    now = `<line class="chart-now" x1="${x(data.now)}" x2="${x(data.now)}" y1="${top - 4}" y2="${base}"/><text class="chart-now-label" x="${x(data.now)}" y="${top - 8}" text-anchor="middle">Now</text>${
      known(data.free_now) ? `<circle class="chart-now-dot" cx="${x(data.now)}" cy="${y(data.free_now)}" r="4"/>` : ""
    }`;
  }
  const label = `Free spaces from 06:00 to 20:00, out of ${cap}: today as a line, a typical ${day} as a shaded area`;
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}">${ticks.join("")}
    <line class="chart-grid" x1="${left}" x2="${width - right}" y1="${y(cap)}" y2="${y(cap)}"/>
    <line class="chart-base" x1="${left}" x2="${width - right}" y1="${base}" y2="${base}"/>
    <text class="chart-tick" x="${left - 6}" y="${y(cap)}" text-anchor="end" dominant-baseline="middle">${cap}</text>
    <text class="chart-tick" x="${left - 6}" y="${base}" text-anchor="end" dominant-baseline="middle">0</text>
    ${typical ? `<path class="chart-typical" d="${typical}"/>` : ""}
    ${line ? `<path class="chart-today${animate ? " draw" : ""}" pathLength="1" d="${line}"/>` : ""}${now}</svg>`;
}

function forecastLines(data) {
  const insights = data.insights;
  if (insights.learning) {
    const days = insights.days || 0;
    return [`Still learning: ${days} day${days === 1 ? "" : "s"} so far`];
  }
  const when = data.day_type === "weekend" ? "on weekends" : "on weekdays";
  const lines = [];
  if (insights.usually_full_by) lines.push(`Usually full by ${insights.usually_full_by} ${when}`);
  if (insights.best_before) lines.push(`Best chance before ${insights.best_before}`);
  if (known(insights.usual_free_now)) {
    const usual = Math.round(insights.usual_free_now);
    lines.push(known(data.free_now) ? `Now ${Math.round(data.free_now)} free, usually ${usual} at this time` : `Usually ${usual} free at this time`);
  }
  return lines;
}

function renderForecast(data, animate) {
  const el = document.getElementById("forecast");
  if (!el) return;
  forecastData = data;
  const rack = lastRacks.find((r) => r.id === selectedId);
  let state = "error";
  let html = '<p class="forecast-quiet">Predictions are not available right now.</p>';
  if (data && Array.isArray(data.typical) && data.insights) {
    state = data.insights.learning ? "learning" : "ready";
    const day = data.day_type === "weekend" ? "weekend day" : "weekday";
    html = `<h3 class="forecast-title">Free spaces through the day</h3>
      ${chartSvg(data, Math.round(el.clientWidth) || 320, day, animate)}
      <p class="chart-key"><span class="key key-today"></span>Today <span class="key key-typical"></span>Typical ${day}</p>
      <ul class="forecast-lines">${forecastLines(data).map((line) => `<li>${esc(line)}</li>`).join("")}</ul>
      ${(rack && rack.demo) || data.clock === "demo" ? `<p class="forecast-note">${DEMO_CHIP}This history is made up.</p>` : ""}`;
  }
  el.dataset.state = state;
  if (html !== forecastHtml) {
    el.innerHTML = html;
    forecastHtml = html;
  }
}

async function loadForecast() {
  const id = selectedId;
  if (id === null) return;
  const token = ++forecastToken;
  const data = await fetch(`/api/racks/${encodeURIComponent(id)}/history`, { cache: "no-store" })
    .then((response) => (response.ok ? response.json() : null))
    .catch(() => null);
  if (token !== forecastToken || id !== selectedId) return; // a newer request or another rack took over
  const el = document.getElementById("forecast");
  renderForecast(data, Boolean(el) && !el.querySelector(".chart"));
}

let resizeTimer = 0;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => forecastData && renderForecast(forecastData, false), 150); // the chart is drawn at the panel's width
});

/* Summary: built once on the first data (replacing the skeleton), then only its parts change */
let heroParts = null;

function renderHero(racks) {
  if (!heroParts) {
    heroEl.removeAttribute("aria-busy");
    heroEl.innerHTML = `<h2 class="eyebrow" id="sum-h">Free now</h2><p class="summary__n"><span class="num"></span></p>
      <p class="summary__label" aria-live="polite"></p>
      <button class="btn btn--primary" type="button" hidden>${ICONS.locate}<span class="btn__label"></span>${ICONS.go}</button>`;
    restart(heroEl, "enter");
    heroParts = { value: heroEl.querySelector(".summary__n"), label: heroEl.querySelector(".summary__label"), best: heroEl.querySelector(".btn") };
  }
  const { value, label, best } = heroParts;
  const live = racks.filter((r) => r.status === "ok");
  const total = live.reduce((sum, r) => sum + r.free, 0);
  roll(value.firstElementChild, live.length ? total : "?");
  value.classList.toggle("is-quiet", !live.length || total === 0);
  const where = live.length > 1 ? ` across ${live.length} racks` : live.length ? ` at ${live[0].name}` : "";
  const text = !live.length
    ? `No live data yet. ${racks.length} rack${racks.length === 1 ? "" : "s"} configured.`
    : total === 0 ? "No free spaces right now" : `free space${total === 1 ? "" : "s"}${where}`;
  if (label.textContent !== text) label.textContent = text;
  const top = live.length ? live.reduce((a, b) => (b.free > a.free ? b : a)) : null;
  best.hidden = !(live.length > 1 && top.free > 0);
  if (!best.hidden) {
    best.dataset.rack = top.id;
    setHtml(best.querySelector(".btn__label"), `<span class="btn__mute">Most space:</span> ${esc(top.name)}`);
  }
}

/* Map with three levels of detail: a count dot when zoomed out, count plus word at street level,
   the rack card with its spaces when zoomed in. A pin re-renders (and animates) only when its content changes. */
const DETAIL_ZOOM = 18;
const COUNT_ZOOM = 16;
let map = null;
const markers = new Map();
let lastRacks = [];

function levelFor(zoom) {
  return zoom >= DETAIL_ZOOM ? "detail" : zoom >= COUNT_ZOOM ? "count" : "dot";
}


/* One pin per rack: the count when zoomed out, count and word at street level, the rack card with its plan when
   zoomed in. enter >= 0 adds the entrance animation with that stagger step. */
function pinHtml(rack, level, enter = -1) {
  const live = rack.status === "ok";
  const real = !rack.demo && anyDemo;
  const selected = rack.id === selectedId ? " is-selected" : "";
  const entering = enter >= 0 ? ` is-entering" style="--i: ${enter}` : "";
  if (level === "detail") {
    const [kind, label] = chipFor(rack);
    const source = sourceChip(rack);
    const plan = rack.layout || (live && rack.slots.length) ? planHtml(rack, [], true) : "";
    return `<div class="mcard${selected}${entering}">
      <div class="mcard__head"><span class="mcard__name">${esc(rack.name)}</span>${chip(kind, label)}</div>
      <p class="count${live ? "" : " count--unknown"}"><span class="count__n">${live ? rack.free : "?"}</span>free of ${rack.capacity}${source ? `<span class="mcard__src">${source}</span>` : ""}</p>
      ${plan}<span class="mcard__stem"></span></div>`;
  }
  const state = !live ? " pin--none" : rack.free === 0 ? " pin--full" : "";
  const word = level === "count" && live ? '<span class="pin__l">free</span>' : "";
  const tip = `${rack.name}${rack.demo ? ", demo data" : real ? ", real camera" : ""}`;
  return `<span class="pin${state}${rack.demo ? " pin--demo" : ""}${selected}${entering}">${real ? CAMERA_MARK : ""}<span class="pin__n">${live ? rack.free : "?"}</span>${word}</span><span class="pin-tip">${esc(tip)}</span>`;
}

/* Rack plans on the map at real size: each space is a polygon in metres (space_m along a row and across it,
   row_gap_m between rows) around the rack's location, turned to its bearing, so Leaflet scales it with the map.
   Where a space would be a smudge only the rack's outline is drawn. A refresh only changes classes. */
const PLAN_ZOOM = 17;
const SPACE_MIN_PX = 3; // a 0.7 m space passes this from zoom 19, where a pin click lands
const ENTRANCE_M = [0.3, 2.3]; // tip and tail of the entrance arrow, off the end of the rack
const plans = new Map();

function flyToRack(rack, zoom) {
  // the rack lands below the middle of the map, so the lifted card above it and the plan under it both fit
  const point = map.project([rack.location.lat, rack.location.lng], zoom).subtract([0, map.getSize().y * 0.18]);
  map.flyTo(map.unproject(point, zoom), zoom, { duration: 0.9, animate: !reduceMotion.matches });
}

function metresPerPx(lat, zoom) {
  return (2 * Math.PI * EARTH_M * Math.cos((lat * Math.PI) / 180)) / (256 * 2 ** zoom);
}

function planGeometry(rack) {
  const { spaces, width, height, bearing, entrance, space_m: [along, depth], row_gap_m: gap } = rack.layout;
  const rad = (bearing * Math.PI) / 180;
  const deg = 180 / Math.PI;
  const pitch = depth + gap;
  const length = width * along;
  const across = height * pitch - gap;
  const { lat, lng } = rack.location;
  // u metres along the row from space 1, v metres across from row 1 (row 1 is on the left looking along the bearing)
  const at = (u, v) => {
    const a = u - length / 2;
    const c = v - across / 2;
    const north = a * Math.cos(rad) - c * Math.sin(rad);
    const east = a * Math.sin(rad) + c * Math.cos(rad);
    return [lat + (north / EARTH_M) * deg, lng + (east / (EARTH_M * Math.cos(lat / deg))) * deg];
  };
  const box = (u0, v0, u1, v1) => [at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)];
  const atStart = entrance === "start";
  const out = atStart ? -1 : 1; // the arrow sits off the entrance end and points into the rack
  const [tip, tail] = ENTRANCE_M.map((m) => (atStart ? 0 : length) + out * m);
  const barb = tip + out * 0.5;
  const mid = across / 2;
  return {
    outline: box(0, 0, length, across),
    spaces: spaces.map((s) => [s.id, box(s.x * along, s.y * pitch, (s.x + s.w) * along, (s.y + s.h) * pitch - gap)]),
    entrance: [[at(tail, mid), at(tip, mid)], [at(barb, mid - 0.45), at(tip, mid), at(barb, mid + 0.45)]],
  };
}

function planFor(rack) {
  if (!plans.has(rack.id)) {
    const geo = planGeometry(rack);
    const shape = (ring, className) => L.polygon(ring, { className, interactive: false });
    const spaces = new Map(geo.spaces.map(([id, ring]) => [id, shape(ring, "plan-space")]));
    const entrance = L.polyline(geo.entrance, { className: "plan-entrance", interactive: false });
    plans.set(rack.id, {
      points: [...geo.outline, ...geo.entrance.flat()],
      outline: shape(geo.outline, "plan-outline"),
      spaces,
      detail: L.layerGroup([...spaces.values(), entrance]),
      mode: "",
    });
  }
  return plans.get(rack.id);
}

function setClass(layer, className) {
  if (layer.options.className === className) return;
  layer.options.className = className; // Leaflet reads it when it draws the path again
  layer.getElement()?.setAttribute("class", className);
}

/* One hatch pattern in the SVG Leaflet draws the polygons in, so taken spaces read without colour too */
function ensureHatch() {
  const svg = mapEl.querySelector(".leaflet-overlay-pane svg");
  if (!svg || svg.querySelector("#plan-hatch")) return;
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = '<pattern id="plan-hatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><rect class="plan-hatch-fill" width="6" height="6"/><line class="plan-hatch-line" x1="3" y1="0" x2="3" y2="6"/></pattern>';
  svg.prepend(defs);
}

/* Draws the rack's plan for this zoom and returns how far above the rack a pin must sit to clear it */
function updatePlan(rack, zoom) {
  const plan = planFor(rack);
  const live = rack.status === "ok";
  const seen = new Map(rack.slots.map((s) => [s.id, s.state]));
  plan.spaces.forEach((layer, id) => setClass(layer, `plan-space plan-space-${(live && seen.get(id)) || "unknown"}`));
  setClass(plan.outline, `plan-outline plan-outline-${!live ? "unknown" : rack.free === 0 ? "full" : "ok"}`);
  const spacePx = Math.min(...rack.layout.space_m) / metresPerPx(rack.location.lat, zoom);
  const mode = zoom < PLAN_ZOOM ? "" : spacePx < SPACE_MIN_PX ? "outline" : "spaces";
  if (mode !== plan.mode) {
    plan.outline.remove();
    plan.detail.remove();
    if (mode) (mode === "outline" ? plan.outline : plan.detail).addTo(map);
    plan.mode = mode;
    ensureHatch();
  }
  if (!mode) return 0;
  const centre = map.project([rack.location.lat, rack.location.lng], zoom).y;
  return Math.ceil(centre - Math.min(...plan.points.map((p) => map.project(p, zoom).y))) + 6;
}

function updateMarkers(racks) {
  if (!map) return;
  const zoom = map.getZoom();
  const level = levelFor(zoom);
  racks.filter((r) => r.location).forEach((rack, index) => {
    const html = pinHtml(rack, level);
    const lift = rack.layout ? updatePlan(rack, zoom) : 0;
    let marker = markers.get(rack.id);
    const enter = !marker ? Math.min(index, 6) : marker.level !== level ? 0 : -1; // new pin or a new level: it enters
    const icon = () => L.divIcon({ className: "pin-anchor", html: enter >= 0 ? pinHtml(rack, level, enter) : html, iconSize: [0, 0] });
    if (!marker) {
      marker = L.marker([rack.location.lat, rack.location.lng], { icon: icon(), keyboard: true, title: rack.name });
      marker.on("click", () => selectRack(rack.id));
      marker.addTo(map);
      markers.set(rack.id, marker);
    } else if (marker.pinHtml !== html || enter >= 0) {
      marker.setIcon(icon());
    }
    marker.pinHtml = html;
    marker.level = level;
    marker.setZIndexOffset(rack.id === selectedId ? 1000 : 0);
    const el = marker.getElement();
    el.style.setProperty("--lift", `${lift}px`); // moved in place: a new icon would replay its entrance
    el.classList.toggle("is-lifted", lift > 0);
    el.classList.toggle("is-none", rack.status !== "ok");
  });
}

function initMap(racks) {
  const placed = racks.filter((r) => r.location);
  if (!mapEl || typeof L === "undefined" || !placed.length) {
    if (mapEl) mapEl.innerHTML = '<p class="map-note">Map unavailable. The list below has the same information.</p>';
    return;
  }
  map = L.map(mapEl, { zoomControl: false, attributionControl: true, maxZoom: 21 });
  map.attributionControl.setPrefix(false);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 21,
    maxNativeZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  })
    .on("tileerror", () => {
      if (!mapEl.querySelector(".map-note")) {
        const note = document.createElement("p");
        note.className = "map-note";
        note.textContent = "Map tiles are not loading (no internet?). Pins and the list below still work.";
        mapEl.appendChild(note);
      }
    })
    .addTo(map);
  L.control.zoom({ position: "bottomright", zoomInText: ICONS.plus, zoomOutText: ICONS.minus }).addTo(map); // above the attribution
  keyEl.hidden = false;
  addDestControl();
  const bounds = L.latLngBounds(placed.map((r) => [r.location.lat, r.location.lng]));
  if (placed.length === 1) map.setView(bounds.getCenter(), 17);
  else map.fitBounds(bounds, { ...destPadding(48), maxZoom: 17 });
  map.on("zoomend", () => updateMarkers(lastRacks));
  updateMarkers(racks);
}

/* "Where are you going?": the live rack with a free space nearest to a destination, or to the browser's position.
   A place from the racks file, a tap on the map and the position stay in this browser; a typed place is looked up
   through this server (/api/geocode). The answer is worked out again on every refresh. */
const GEO_ERRORS = {
  1: "Location is blocked for this page. Allow it in the browser, or pick a place.",
  2: "Your position is not available right now.",
  3: "Finding your position took too long. Please try again.",
};
const FAR_M = 3000;
const narrow = window.matchMedia("(max-width: 639px)"); // the control folds into one button, as in style.css
let destEl = null;
let destQueryEl = null;
let destResultEl = null;
let places = [];
let placesJson = "";
let route = [];
let dest = null; // {lat, lng, how, name}
let destRackId; // the rack the current answer points to
let searchToken = 0;

function say(text) {
  if (destResultEl.textContent !== text) destResultEl.textContent = text; // the same line again is not announced again
  destResultEl.hidden = false;
}

/* Fit padding that keeps pins and the line clear of the control: beside it on a wide map, below it on a narrow one */
function destPadding(edge) {
  const box = mapEl.querySelector(".dest");
  if (mapEl.clientWidth > box.offsetWidth * 2 + 160) return { paddingTopLeft: [edge, edge], paddingBottomRight: [box.offsetWidth + edge + 16, edge] };
  return { paddingTopLeft: [edge, box.offsetHeight + edge + 16], paddingBottomRight: [edge, edge] };
}

function setOpen(open) {
  destEl.classList.toggle("open", open);
  destEl.querySelector("#dest-open").setAttribute("aria-expanded", String(open));
}

function addDestControl() {
  const control = L.control({ position: "topright" });
  control.onAdd = () => {
    const box = L.DomUtil.create("div", "dest");
    box.innerHTML = `<button type="button" class="dest-open" id="dest-open" aria-expanded="false" aria-controls="dest-body">${ICONS.search}Where are you going?</button>
      <div class="dest-body" id="dest-body">
      <form class="dest-search" id="dest-form" role="search">
        <label class="dest-title" for="dest-q">Where are you going?</label>
        <div class="dest-field"><input class="dest-q" id="dest-q" type="search" list="dest-places" autocomplete="off" enterkeyhint="search" maxlength="120" placeholder="A place or an address">
        <button type="submit" class="dest-go" id="dest-go" aria-label="Search">${ICONS.search}</button></div>
        <datalist id="dest-places"></datalist>
      </form>
      <div class="dest-more"><span class="dest-hint">or tap the map</span>
      <button type="button" class="dest-near" id="nearest">${ICONS.locate}Nearest to me</button></div>
      <p class="dest-result" id="dest-result" aria-live="polite" hidden></p>
      <p class="dest-note">A typed place is looked up with OpenStreetMap's search through this server and is not stored. A place from the list, a tap on the map and your position stay in this browser.</p></div>`;
    L.DomEvent.disableClickPropagation(box);
    L.DomEvent.disableScrollPropagation(box);
    return box;
  };
  control.addTo(map);
  destEl = mapEl.querySelector(".dest");
  destQueryEl = mapEl.querySelector("#dest-q");
  destResultEl = mapEl.querySelector("#dest-result");
  mapEl.querySelector("#dest-open").addEventListener("click", () => {
    setOpen(!destEl.classList.contains("open"));
    if (destEl.classList.contains("open")) destQueryEl.focus();
  });
  mapEl.querySelector("#dest-form").addEventListener("submit", (event) => {
    event.preventDefault();
    search(destQueryEl.value);
  });
  destQueryEl.addEventListener("input", (event) => {
    // picking a suggestion (not typing) answers at once; a configured place needs no lookup
    if (event.inputType && event.inputType !== "insertReplacementText") return;
    const place = placeNamed(destQueryEl.value);
    if (place) goTo(place.location, "walk", place.name);
  });
  mapEl.querySelector("#nearest").addEventListener("click", nearMe);
  map.on("click", (event) => {
    destQueryEl.value = "";
    goTo(event.latlng, "walk");
  });
}

function renderPlaces(list) {
  const json = JSON.stringify(list || []);
  if (!destEl || json === placesJson) return;
  placesJson = json;
  places = list || [];
  mapEl.querySelector("#dest-places").innerHTML = places.map((p) => `<option value="${esc(p.name)}"></option>`).join("");
}

function placeNamed(text) {
  const key = text.trim().replace(/\s+/g, " ").toLowerCase();
  return places.find((p) => p.name.toLowerCase() === key);
}

async function search(raw) {
  const text = raw.trim().replace(/\s+/g, " ");
  if (!text) return say("Type a place or an address, or tap the map.");
  const place = placeNamed(text);
  if (place) return goTo(place.location, "walk", place.name);
  const token = ++searchToken;
  say(`Looking up ${text}`);
  let found = null;
  let problem = "";
  try {
    const response = await fetch(`/api/geocode?q=${encodeURIComponent(text)}`, { cache: "no-store" });
    if (response.status === 404) problem = "Searching for any place is off on this server. Pick a place from the list or tap the map.";
    else if (!response.ok) problem = "The place search is not answering right now. Pick a place from the list or tap the map.";
    else found = await response.json();
  } catch {
    problem = "The place search cannot be reached right now. Pick a place from the list or tap the map.";
  }
  if (token !== searchToken) return; // a newer search, a tap or a pick took over
  if (problem) return say(problem);
  const top = Array.isArray(found) ? found[0] : null;
  if (!top || !Number.isFinite(Number(top.lat)) || !Number.isFinite(Number(top.lng))) {
    return say(`Nothing found for "${text}". Try another name or tap the map.`);
  }
  goTo({ lat: Number(top.lat), lng: Number(top.lng) }, "walk", String(top.name || text));
}

function goTo(point, how, name = "") {
  searchToken++;
  dest = { lat: point.lat, lng: point.lng, how, name };
  destRackId = undefined;
  if (narrow.matches) setOpen(false); // folded before the fit, so the line and the rack card stay in view
  answer();
}

/* The map is fitted and the rack selected only when the answer moves to another rack, so a refresh that keeps
   the same answer leaves the user's panning and selection alone */
function answer() {
  targetPos = { lat: dest.lat, lng: dest.lng };
  const bike = dest.how === "bike";
  const placed = lastRacks.filter((r) => r.location).map((r) => [r, distanceM(targetPos, r.location)]).sort((a, b) => a[1] - b[1]);
  const pick = placed.find(([r]) => r.status === "ok" && r.free > 0);
  const to = [targetPos.lat, targetPos.lng];
  const name = dest.name ? `<span class="dest-name">${esc(dest.name)}</span>` : "";
  route.forEach((layer) => layer.remove());
  route = [bike
    ? L.circleMarker(to, { radius: 7, className: "you-dot", interactive: false })
    : L.marker(to, { icon: L.divIcon({ className: "pin-wrap", html: `<span class="dest-mark"></span>${name}`, iconSize: null }), interactive: false, keyboard: false })];
  if (!pick) {
    route[0].addTo(map);
    destRackId = null;
    say("No rack with a known place has a free space right now.");
    renderMeta(); // the tiles show each rack's distance
    return;
  }
  const [rack, metres] = pick;
  const minutes = Math.max(1, Math.ceil(metres / (bike ? BIKE_M_PER_MIN : WALK_M_PER_MIN)));
  const reply = `${rack.name}: ${rack.free} free, about ${minutes} min ${bike ? "by bike" : "walk"}`;
  const [closest, nearestM] = placed[0];
  const far = nearestM > FAR_M ? `. ${dest.name || "This place"} is ${distanceText(nearestM)} from the nearest rack` : "";
  say((closest === rack ? reply : `${closest.name} ${closest.status === "ok" ? "is full" : "has no live data"}, next: ${reply}`) + far);
  const from = [rack.location.lat, rack.location.lng];
  const middle = [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2];
  route.push(
    L.polyline([from, to], { className: "dest-line", interactive: false }),
    L.marker(middle, { icon: L.divIcon({ className: "pin-wrap", html: `<span class="dest-label">${distanceText(metres)}</span>`, iconSize: null }), interactive: false, keyboard: false }),
  );
  route.forEach((layer) => layer.addTo(map));
  renderMeta();
  if (rack.id === destRackId) return;
  destRackId = rack.id;
  map.fitBounds(L.latLngBounds([from, to]), { ...destPadding(40), maxZoom: PLAN_ZOOM, animate: !reduceMotion.matches });
  selectRack(rack.id, false);
}

function nearMe() {
  if (!navigator.geolocation) {
    say("This browser cannot share your position.");
  } else if (!window.isSecureContext) {
    say("Browsers only share your position with https pages, and this one is plain http. Pick a place or tap the map.");
  } else {
    say("Finding your position");
    navigator.geolocation.getCurrentPosition(
      (position) => goTo({ lat: position.coords.latitude, lng: position.coords.longitude }, "bike"),
      (error) => say(GEO_ERRORS[error.code] || GEO_ERRORS[2]),
      { timeout: 10000, maximumAge: 60000 },
    );
  }
}


heroEl.addEventListener("click", (event) => {
  const button = event.target.closest(".btn");
  if (button) selectRack(button.dataset.rack);
});

async function refresh() {
  try {
    const response = await fetch("/api/racks", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    lastRacks = data.racks;
    anyDemo = data.racks.some((r) => r.demo);
    renderHero(data.racks);
    renderRacks(data.racks); // before the map, so the pins know the selection
    if (!map && mapEl && !mapEl.dataset.ready) {
      mapEl.dataset.ready = "1";
      initMap(data.racks);
    } else {
      updateMarkers(data.racks);
    }
    renderPlaces(data.places);
    if (dest && map) answer(); // a rack that filled up hands over to the next one
    noteEl.hidden = !data.racks.some((r) => r.demo);
    updatedEl.innerHTML = `Last refresh <span class="num">${esc(new Date().toLocaleTimeString())}</span>`;
    restart(refreshIco, "is-spinning");
  } catch (err) {
    updatedEl.textContent = `Server unreachable (${err.message}). Retrying.`;
  }
}

refresh();
setInterval(refresh, 5000);
setInterval(loadForecast, 30000);
