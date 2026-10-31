const el = (id) => document.getElementById(id);

const ICONS = {
  dot: '<svg viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="5"/></svg>',
  half: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 0a5 5 0 0 1 0 10z"/></svg>',
  tri: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 0l5 10H0z"/></svg>',
  ring: '<svg viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="4"/></svg>',
  square: '<svg viewBox="0 0 10 10" aria-hidden="true"><rect width="10" height="10" rx="1"/></svg>',
  dashed: '<svg viewBox="0 0 10 10" aria-hidden="true"><rect x="1" y="1" width="8" height="8" rx="1" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="2 1.5"/></svg>',
  camera: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M1 3h2l1-1.5h2L7 3h2v5.5H1z" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/><circle cx="5" cy="5.6" r="1.6" stroke="currentColor" stroke-width="1.2"/></svg>',
};

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function ago(seconds) {
  if (seconds === null || seconds === undefined) return "never";
  if (seconds < 60) return `${Math.round(seconds)} s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  return `${(seconds / 3600).toFixed(1)} h ago`;
}

function chip(kind, icon, label) {
  return `<span class="chip chip-${kind}">${ICONS[icon]}${esc(label)}</span>`;
}

function cameraChip(c) {
  if (c.too_hot) return chip("warning", "tri", "Too hot");
  if (c.online) return chip("good", "dot", "Online");
  if (c.last_alive_at === null) return chip("muted", "ring", "Never seen");
  return chip("critical", "square", "Offline");
}

/* Next to demo racks, the real ones get a chip too, so the real camera stands out */
const anyDemo = (status) => status.racks.some((k) => k.demo) || status.cameras.some((c) => c.demo);

function demoChip(item, status) {
  if (item.demo) return ` ${chip("demo", "dashed", "Demo")}`;
  return anyDemo(status) ? ` ${chip("real", "camera", "Real camera")}` : "";
}

/* What the node reports about itself (pull mode, or a pushed alive message with the same keys) */
const LOW_BLOCK_BYTES = 100 * 1024;
const REBOOT_SHOWN_S = 600;

const signal = (dbm) => (dbm >= -60 ? "excellent" : dbm >= -75 ? "ok" : "weak");

function wifi(node) {
  if (node.wifi_rssi_dbm === null) return "n/a";
  return `${node.wifi_rssi_dbm} dBm, ${signal(node.wifi_rssi_dbm)}`;
}

function memory(node) {
  if (node.heap_largest_block === null) return "n/a";
  const block = `${Math.round(node.heap_largest_block / 1024)} KB`;
  return node.heap_largest_block < LOW_BLOCK_BYTES ? `${block} ${chip("warning", "tri", "Low")}` : block;
}

function rackChip(k) {
  if (k.status === "ok") return chip("good", "dot", "Live");
  if (k.status === "starting") return chip("warning", "half", "Starting");
  if (k.status === "stale") return chip("serious", "tri", "Stale");
  return chip("muted", "ring", "No data");
}

function kv(pairs) {
  return pairs.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("");
}

/* Data-flow diagram: node states, details, the travelling packet and the health pulse */
let selectedNode = "camera";
let lastStatus = null;
let lastPictureTotal = null;
let lastHealthKey = null;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const PACKET_MS = 3000; // the dur of #packet-anim
const PULSE_MS = 1200; // the dur of #pulse-anim
/* Defaults of the app and of a rack's camera block; the status does not report them */
const PULL_HEALTH_S = 5;
const PULL_PICTURE_S = 10;
const HOT_LIMIT_C = 75;
const DETECTOR_TIMEOUT_S = 15;
const DEMO_EVERY_S = 5;
const VISION_POLL_S = 3; // VISION_POLL_S of the app, the status does not report it

/* Lanes inside each step, parts in the order they happen: [id, label], a third item true marks a planned part.
   The first lane is the picture lane on the line the arrows use; a quieter second lane runs below it. joins: the side
   where a lane runs on into the next step; loop: drawn as a loop; clock: the step has a clock rule as a side note.
   flowNodes() supplies the values and the labels that depend on the mode. */
const FLOW_PARTS = {
  camera: {
    lanes: [
      { tag: "picture", parts: [["heat", "heat guard on the board"], ["capture", "take picture"], ["send", "picture out"]] },
      { tag: "health", joins: "right", parts: [["sensors", "temperature, WiFi, memory, uptime"], ["alive", "health answer"]] },
    ],
  },
  receive: {
    clock: true,
    lanes: [
      { tag: "picture", parts: [["endpoint", "picture in, counted"], ["memory", "size noted, held for this request"], ["forward", "to the detector"]] },
      { tag: "health", joins: "left", parts: [["alive", "health or alive in"], ["fields", "fields mapped"], ["hot", "too hot if no flag"], ["online", "camera online"]] },
    ],
  },
  detector: {
    http: {
      lanes: [
        { tag: "picture", parts: [["analyze", `POST /analyze, ${DETECTOR_TIMEOUT_S} s timeout`], ["blur", "blur people", true], ["bikes", "find bikes, pre-trained model"], ["match", "match slot boxes"], ["answer", "answer per space"]] },
        { tag: "health check", parts: [["health", "health check"], ["model", "model"]] },
      ],
    },
    fake: {
      lanes: [
        { tag: "picture", parts: [["standin", "built-in stand-in"], ["made", "made-up answer"], ["answer", "answer per space"]] },
        { tag: "health check", parts: [["health", "health check"], ["model", "model"]] },
      ],
    },
  },
  decide: {
    clock: true,
    lanes: [
      { tag: "picture", parts: [["discard", "picture deleted"], ["confidence", "confidence rule"], ["window", "picture rule"], ["counts", "free count per rack"], ["history", "history sample"]] },
    ],
  },
  pages: {
    lanes: [
      { loop: true, parts: [["racks", "user page asks for the racks"], ["history", "history of the selected rack"], ["user", "shows free spaces"]] },
      { loop: true, parts: [["status", "admin page asks for the status"], ["admin", "shows each part's health"]] },
    ],
  },
};

function short(text, max = 24) {
  text = String(text);
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function flowNodes(status) {
  const cams = status.cameras;
  const real = cams.filter((c) => !c.demo);
  const demo = cams.filter((c) => c.demo).length;
  const online = cams.filter((c) => c.online).length;
  const hot = cams.filter((c) => c.too_hot).length;
  const seen = cams.filter((c) => c.last_alive_at !== null).length;
  const pictures = cams.reduce((sum, c) => sum + c.pictures, 0);
  const youngest = (key) => cams.reduce((best, c) => (c[key] !== null && (best === null || c[key] < best) ? c[key] : best), null);
  const lastPicture = youngest("picture_age_s");
  const lastAlive = youngest("alive_age_s");
  const temps = cams.filter((c) => c.temp_c !== null).map((c) => c.temp_c);
  const signals = cams.map((c) => c.node.wifi_rssi_dbm).filter((dbm) => dbm !== null);
  const blocks = cams.map((c) => c.node.heap_largest_block).filter((bytes) => bytes !== null);
  const reporting = cams.filter((c) => c.temp_c !== null || c.node.uptime_s !== null || c.node.wifi_rssi_dbm !== null).length;
  const derived = cams.filter((c) => c.too_hot_derived).length;
  const lastKb = cams.some((c) => c.last_picture_bytes) ? `${Math.round(Math.max(...cams.map((c) => c.last_picture_bytes)) / 1024)} KB` : null;
  const d = status.detector;
  const r = status.detector_results;
  const fake = d.mode === "fake";
  const app = status.app;
  const live = status.racks.filter((k) => k.status === "ok");
  const staleRacks = status.racks.filter((k) => k.status === "stale").length;
  const totalFree = live.reduce((sum, k) => sum + k.free, 0);
  const slots = status.racks.flatMap((k) => k.slots);
  const stale = app.stale_after_s;
  const receiving = lastPicture !== null && lastPicture <= stale;
  const deciding = r.ok_age_s !== null && r.ok_age_s <= stale;
  const decideState = deciding ? "good" : r.last_ok_at ? "warning" : "muted";
  const healthModel = d.info.model ?? "unknown";
  const model = r.model ?? healthModel;
  const sources = new Set(real.map((c) => c.source).filter(Boolean));
  const mode = sources.size >= 2 ? "mixed" : sources.has("pull") ? "pull" : sources.has("push") ? "push" : sources.has("vision") ? "vision" : null;
  const visionRacks = cams.filter((c) => c.vision);
  const readings = [
    temps.length ? `${Math.round(Math.max(...temps))} C` : "",
    signals.length ? `${Math.min(...signals)} dBm` : "",
    blocks.length ? `${Math.round(Math.min(...blocks) / 1024)} KB block` : "",
  ].filter(Boolean).join(", ");
  const realWords = {
    pull: `pull: the app asks (health ${PULL_HEALTH_S} s, picture ${PULL_PICTURE_S} s)`,
    push: "push: the node posts by itself",
    vision: "vision service: the laptop keeps the camera, the app reads the decided spaces",
    mixed: real.filter((c) => c.source).map((c) => `${c.rack} ${c.source}`).join(", "),
  };

  const camera = hot
    ? ["warning", `${hot} too hot`]
    : online === cams.length && cams.length
      ? ["good", `${online} online`]
      : online
        ? ["warning", `${online} of ${cams.length} online`]
        : seen
          ? ["critical", "offline"]
          : ["muted", "no camera yet"];
  const pagesState = live.length === status.racks.length && live.length ? "good" : live.length ? "warning" : "muted";
  const howCamera = {
    pull: `The real camera is asked by the app (pull): every ${PULL_HEALTH_S} s for its health (chip temperature, WiFi signal, memory, uptime) and every ${PULL_PICTURE_S} s for a picture, the picture only after a good health answer in the same round.`,
    push: "The real camera posts a picture and an alive message to the app by itself (push).",
    vision: `The real camera is kept by the vision service on the vision laptop: the service fetches the pictures itself, decides free or taken per hook (slot classifier, smoothed over a few frames) and the app only reads the decided spaces, every ${VISION_POLL_S} s or pushed by the service. No picture reaches the app on this path.`,
    mixed: `Some real cameras are asked by the app (pull: health every ${PULL_HEALTH_S} s, a picture every ${PULL_PICTURE_S} s after a good health answer), others post a picture and an alive message by themselves (push).`,
  };
  return {
    edges: [
      mode === "pull"
        ? ["app asks", [`picture every ${PULL_PICTURE_S} s`, `health every ${PULL_HEALTH_S} s`], ["", "app asks: health", `${PULL_HEALTH_S} s, picture ${PULL_PICTURE_S} s`]]
        : mode === "push"
          ? ["node posts", ["picture and alive", ""], ["", "node posts a", "picture and alive"]]
          : mode === "mixed"
            ? ["pull and push", ["picture and health", ""], ["", "pull: app asks", "push: node posts"]]
            : mode === "vision"
              ? ["decided spaces", ["from the vision service,", "no picture"], ["", "vision service:", "decided spaces"]]
              : ["picture, health", ["", ""], ["", "picture, health", ""]],
      fake
        ? ["picture", ["to the built-in", "stand-in"], ["", "picture in memory,", "to the stand-in"]]
        : ["picture", ["real racks only,", "held in memory"], ["", "real racks only,", "held in memory"]],
      ["answer per space", ["taken, free, unsure", "with a confidence"], ["", "taken, free or", "unsure per space"]],
      ["free per rack", ["user page asks 5 s", "admin page 3 s"], ["free per rack", "user page 5 s,", "admin page 3 s"]],
    ],
    camera: {
      state: camera[0],
      label: camera[1],
      sub: { pull: "ESP32-S3, app asks", push: "ESP32-S3, it posts", vision: "via vision service", mixed: "pull and push" }[mode] ?? "ESP32-S3 board",
      title: "Camera node (ESP32-S3)",
      text: `${howCamera[mode] ?? "No real camera has been heard from yet."} The heat guard runs on the board: above the hot limit it stops taking pictures. When the node sends no too-hot flag, the app works it out from the temperature (${HOT_LIMIT_C} C).${demo ? ` Demo racks have no camera: the demo driver posts a made-up picture and an alive message every ${DEMO_EVERY_S} s (push).` : ""}`,
      live: [
        ["Cameras online", `${online} of ${cams.length}${demo ? ` (${demo} demo)` : ""}`],
        ["Real camera", real.map((c) => `${esc(c.rack)}, ${esc(status.racks.find((k) => k.id === c.rack)?.name ?? "")}: ${c.source ?? "not heard from yet"}`).join("; ") || "none"],
        ...(demo ? [["Demo racks", `${cams.filter((c) => c.demo).map((c) => esc(c.rack)).join(", ")}: fed by the demo driver`]] : []),
        ["Too hot", hot ? `${hot}` : "none"],
        ["Last picture", lastPicture === null ? "never" : ago(lastPicture)],
        ["Chip temperatures", cams.filter((c) => c.temp_c !== null).map((c) => `${esc(c.rack)}: ${c.temp_c} C`).join(", ") || "n/a"],
        ["WiFi signal", cams.filter((c) => c.node.wifi_rssi_dbm !== null).map((c) => `${esc(c.rack)}: ${wifi(c.node)}`).join(", ") || "n/a"],
        ["Largest free memory block", cams.filter((c) => c.node.heap_largest_block !== null).map((c) => `${esc(c.rack)}: ${memory(c.node)}`).join(", ") || "n/a"],
      ],
      labels: {
        capture: mode === "pull" ? "take picture if health ok" : "take picture",
        send: { pull: "app fetches it", push: "node posts it", vision: "vision service decides" }[mode] ?? "picture out",
        alive: { pull: "health answer", push: "alive message", vision: "status answer" }[mode] ?? "health or alive",
      },
      notes: {
        mode: real.length ? `real camera, ${realWords[mode] ?? "not heard from yet"}` : "no real camera",
        foot: demo ? `demo racks: no camera, the demo driver posts made-up pictures every ${DEMO_EVERY_S} s` : "",
      },
      parts: {
        heat: [hot ? "warning" : temps.length ? "good" : "muted", hot ? `${hot} too hot: no pictures` : temps.length ? `hottest ${Math.round(Math.max(...temps))} C, none too hot` : "no reading"],
        capture: [receiving ? "good" : pictures ? "warning" : "muted", lastPicture === null ? "no picture yet" : ago(lastPicture)],
        send: [pictures ? "good" : "muted", pictures ? `${pictures} sent${lastKb ? `, last ${lastKb}` : ""}` : "none yet"],
        sensors: [readings ? "good" : "muted", readings || "no reading yet"],
        alive: [camera[0], `${online} of ${cams.length} online`],
      },
    },
    receive: {
      state: receiving ? "good" : pictures ? "warning" : "muted",
      label: receiving ? "receiving" : pictures ? "idle" : "waiting",
      sub: "pictures and health",
      title: "App receives the picture and the health answers",
      text: `Two separate paths. Health: a health answer or alive message is mapped onto the camera's fields; when the node sends no too-hot flag, the app compares the temperature with ${HOT_LIMIT_C} C. A camera counts as online while answers arrive and as offline after ${stale} s without one. Pictures: each one is counted and its size noted, then held in memory for this one request only. It is never written to disk and never logged.`,
      live: [
        ["Pictures received", `${pictures}`],
        ["Last picture", lastPicture === null ? "never" : ago(lastPicture)],
        ["Last size", lastKb ?? "n/a"],
        ["Last health or alive", ago(lastAlive)],
        ["Cameras online", `${online} of ${cams.length}`],
        ["Too hot worked out by the app", `${derived}`],
      ],
      notes: { foot: `clock rule: offline after ${stale} s without an answer` },
      parts: {
        endpoint: [receiving ? "good" : pictures ? "warning" : "muted", `${pictures} received`],
        memory: [pictures ? "good" : "muted", lastKb ? `last ${lastKb}, never stored` : "nothing yet"],
        forward: fake ? ["good", "built-in stand-in"] : [d.reachable ? "good" : "critical", `${d.reachable ? "service reachable" : "service down"}${demo ? ", demo: stand-in" : ""}`],
        alive: [lastAlive !== null && lastAlive <= stale ? "good" : seen ? "critical" : "muted", lastAlive === null ? "none yet" : `last ${ago(lastAlive)}`],
        fields: [reporting ? "good" : "muted", `${reporting} of ${cams.length} with readings`],
        hot: [hot ? "warning" : temps.length ? "good" : "muted", `${HOT_LIMIT_C} C limit, ${hot} hot`],
        online: [online ? "good" : seen ? "critical" : "muted", `${online} of ${cams.length} online`],
      },
    },
    detector: {
      state: d.reachable ? "good" : "critical",
      label: d.reachable ? (fake ? "stand-in" : "reachable") : "down",
      sub: fake ? "built-in stand-in" : "bikes per slot box",
      title: "Detector (recognition service)",
      text: `${fake
        ? "Built-in stand-in, no recognition: the app is set to use its stand-in for every rack (detector mode fake), so nothing is recognised. The stand-in makes up taken or free per space from the picture's bytes."
        : `A real rack's picture goes to the recognition service (POST /analyze, ${DETECTOR_TIMEOUT_S} s timeout). It finds bikes with a pre-trained model, matches them to the slot boxes drawn once per camera and answers taken, free or unsure per space with a confidence. People are never reported; blurring them before recognition is being built, not running yet. Demo racks never reach it: the app's built-in stand-in reads their made-up pictures.`} The health check is separate: it runs only when this admin page asks, at most every 5 s. The last error is a status, not a step.${visionRacks.length ? ` Racks with a vision block (${visionRacks.map((c) => c.rack).join(", ")}) never come here either: the vision service on the laptop decides their spaces and the app reads them (see the camera cards).` : ""}`,
      live: [
        ["Mode", fake ? "fake (built-in stand-in, no recognition)" : esc(d.url)],
        ["Reachable", d.reachable ? "yes" : `<span class="error">no: ${esc(d.error ?? "")}</span>`],
        ["Model", fake ? "none" : esc(model)],
        ["Last latency", r.last_latency_ms === null ? "n/a" : `${r.last_latency_ms} ms`],
        ["Last error", r.last_error ? `<span class="error">${esc(r.last_error)}</span>` : "none"],
      ],
      notes: {
        mode: fake ? "built-in stand-in, no recognition" : `model: ${short(healthModel, 28)}`,
        foot: fake ? "every rack, demo or real, gets the stand-in" : "demo racks: the stand-in, never sent here",
        status: `last error: ${r.last_error ? short(r.last_error, 18) : "none"}`,
      },
      parts: {
        standin: [d.reachable ? "good" : "critical", "no recognition"],
        made: [decideState, "from the bytes, not the image"],
        analyze: [decideState, r.last_latency_ms === null ? "no call yet" : `last ${r.last_latency_ms} ms`],
        blur: ["muted", "planned, being built"],
        bikes: [decideState, short(model)],
        match: [slots.length ? "good" : "muted", "boxes drawn once per camera"],
        answer: [decideState, fake ? (r.ok_age_s === null ? "none yet" : `last ${ago(r.ok_age_s)}`) : "taken, free or unsure, no people"],
        health: [d.reachable ? "good" : "critical", d.reachable ? "when this page asks, 5 s at most" : `down: ${short(d.error ?? "", 20)}`],
        model: [fake ? "muted" : d.reachable ? "good" : "critical", fake ? "none, stand-in" : short(healthModel)],
      },
    },
    decide: {
      state: decideState,
      label: deciding ? "deciding" : r.last_ok_at ? "idle" : "waiting",
      sub: "drops picture, rules",
      title: "App drops the picture, then applies the rules",
      text: `In this order. The picture is deleted right after the detector answers (or fails). Then the confidence rule: a free below ${app.min_confidence} confidence, or an unsure, counts as a bike. Then the ${app.free_after_clean}-picture rule: a space shows taken at once, and free only after ${app.free_after_clean} clean pictures in a row. Then the free count per rack, and one history sample per rack per minute (counts only, never which space was used when). Apart from this order, on the clock: a rack without a result for ${stale} s shows stale.`,
      live: [
        ["Minimum confidence for free", `${app.min_confidence}`],
        ["Clean pictures before free", `${app.free_after_clean}`],
        ["Stale after", `${stale} s`],
        ["Last decision", ago(r.ok_age_s)],
        ["Racks live", `${live.length} of ${status.racks.length}`],
      ],
      labels: { window: `${app.free_after_clean}-picture rule` },
      notes: { foot: `clock rule: a rack shows stale after ${stale} s without a result${staleRacks ? `, ${staleRacks} stale now` : ""}` },
      parts: {
        discard: [pictures ? "good" : "muted", `${pictures} deleted, none kept`],
        confidence: [decideState, `free under ${app.min_confidence} or unsure: bike`],
        window: [decideState, `taken at once, free after ${app.free_after_clean} clean`],
        counts: [pagesState, `${live.length} of ${status.racks.length} racks live`],
        history: [live.length ? "good" : "muted", "1 per rack per minute"],
      },
    },
    pages: {
      state: pagesState,
      label: `${live.length} of ${status.racks.length} live`,
      sub: "user 5 s, admin 3 s",
      title: "Pages",
      text: "Two loops that run on their own. The user page asks for the racks every 5 seconds, and for the history of the selected rack every 30 seconds. This admin page asks for the status every 3 seconds.",
      live: [
        ["Free spaces now", `${totalFree} (live racks only)`],
        ["Racks live", `${live.length} of ${status.racks.length}`],
        ["Server version", `v${esc(app.version)}`],
      ],
      parts: {
        racks: [live.length ? "good" : "muted", `every 5 s, ${totalFree} free now`],
        history: [live.length ? "good" : "muted", "every 30 s"],
        user: [pagesState, `${status.racks.length} racks, ${status.racks.filter((k) => k.location).length} on the map`],
        status: ["good", `every 3 s, v${app.version}`],
        admin: ["good", `${cams.length} cameras, ${status.racks.length} racks`],
      },
    },
  };
}

/* Each lane is a line through the box with its parts as stations, like a metro map. The picture lane sits on the line
   the arrows use, labels alternating above and below; the quieter second lane runs below it with its labels underneath. */
const LANE_Y = [80, 103];
function buildInside(g, layout) {
  const box = g.querySelector(".node-box");
  const x0 = Number(box.getAttribute("x"));
  const x1 = x0 + Number(box.getAttribute("width"));
  const text = (cls, x, y, anchor, content) => `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(content)}</text>`;
  const lanes = layout.lanes.map((lane, l) => {
    const y = LANE_Y[l];
    const n = lane.parts.length;
    const first = x0 + (l ? 30 : 12); // the second lane leaves room for its tag
    const last = x1 - 12;
    const xs = lane.parts.map((_, i) => first + (i * (last - first)) / (n - 1));
    const from = l === 0 || lane.joins === "left" ? x0 : first - 4;
    const to = l === 0 || lane.joins === "right" ? x1 : last + 4;
    const mids = xs.slice(1).map((x, i) => `${(x + xs[i]) / 2},${y}`).join(" ");
    const back = l ? -3.5 : 3.5; // a loop returns on the side where this lane has no labels
    const sweep = l ? 0 : 1;
    const loop = lane.loop
      ? `<path class="loop" d="M${last + 4} ${y}A1.75 1.75 0 0 ${sweep} ${last + 4} ${y + back}H${first - 4}A1.75 1.75 0 0 ${sweep} ${first - 4} ${y}" marker-mid="url(#arrow-small)"/>`
      : "";
    const stations = lane.parts.map(([id, label, planned], i) => {
      const x = xs[i];
      const anchor = i === 0 ? "start" : i === n - 1 ? "end" : "middle";
      const tx = i === 0 ? x - 5 : i === n - 1 ? x + 5 : x;
      const far = !l && n === 5 && i === 2; // the middle one of three on the same side moves up a row
      const [labelY, valueY] = l ? [y + 6.5, y + 11.5] : i % 2 === 0 ? (far ? [58, 63.5] : [69.5, 75]) : [88, 94];
      return `<g class="part${l ? " minor" : ""}${planned ? " planned" : ""}" data-part="${id}">
        ${far ? `<line class="tick" x1="${x}" y1="${y - 3}" x2="${x}" y2="66"/>` : ""}
        <circle class="dot" cx="${x}" cy="${y}" r="${l ? 2 : 2.6}"/>
        ${text("part-label", tx, labelY, anchor, label)}
        ${text("part-value", tx, valueY, anchor, "...")}
      </g>`;
    });
    return `<polyline class="chain${l ? " lane-2" : ""}" points="${from},${y} ${mids} ${to},${y}" marker-mid="url(#arrow-small)"/>
      ${loop}
      ${lane.tag ? text("lane-tag", x0 + 3, l ? y - 2.5 : y + 6.5, "start", lane.tag) : ""}
      ${stations.join("")}`;
  });
  g.querySelector(".inside").innerHTML = `
    ${text("inside-caption", x0 + 6, 46.5, "start", g.querySelector(".node-title").textContent)}
    <text class="note" data-note="mode" x="${x1 - 6}" y="46.5" text-anchor="end"></text>
    ${lanes.join("")}
    ${layout.clock ? `<g class="note-clock"><circle cx="${x0 + 8}" cy="118.6" r="1.6"/><path d="M${x0 + 8} 117.6V118.6H${x0 + 9}"/></g>` : ""}
    <text class="note" data-note="foot" x="${x0 + (layout.clock ? 11 : 6)}" y="119.8"></text>
    <text class="note" data-note="status" x="${x1 - 6}" y="119.8" text-anchor="end"></text>`;
}

/* The detector's parts depend on its mode, so its inside is built again when the mode changes */
function ensureInside(g, detectorMode) {
  const key = g.dataset.node === "detector" ? detectorMode : "fixed";
  if (g.dataset.layout === key) return;
  g.dataset.layout = key;
  buildInside(g, g.dataset.node === "detector" ? FLOW_PARTS.detector[detectorMode] : FLOW_PARTS[g.dataset.node]);
}

function setLines(text, lines, ys) {
  const x = text.getAttribute("x");
  text.innerHTML = lines.map((line, i) => (line ? `<tspan x="${x}" y="${ys[i]}">${esc(line)}</tspan>` : "")).join("");
}

function renderFlow(status) {
  const nodes = flowNodes(status);
  document.querySelectorAll("#flow .node").forEach((g) => {
    const info = nodes[g.dataset.node];
    ensureInside(g, status.detector.mode);
    g.dataset.state = info.state;
    g.querySelector(".node-status").textContent = info.label;
    g.querySelector(".node-sub").textContent = info.sub;
    g.classList.toggle("selected", g.dataset.node === selectedNode);
    g.querySelectorAll(".part").forEach((part) => {
      const [state, value] = info.parts[part.dataset.part];
      part.dataset.state = state;
      part.querySelector(".part-value").textContent = value;
      const label = info.labels?.[part.dataset.part];
      if (label) part.querySelector(".part-label").textContent = label;
    });
    g.querySelectorAll(".note").forEach((note) => (note.textContent = info.notes?.[note.dataset.note] ?? ""));
  });
  nodes.edges.forEach(([label, notes, small], i) => {
    document.querySelector(`#flow .edge-label[data-edge="${i}"]`).textContent = label;
    setLines(document.querySelector(`#flow .edge-note[data-edge="${i}"]`), notes, [97, 109]);
    setLines(document.querySelector(`#flow .edge-small[data-edge="${i}"]`), small, [71.5, 76, 85.5]); // short lines: the margin beside a zoomed step is narrow
  });
  const info = nodes[selectedNode];
  el("flow-detail").innerHTML = kv([["Step", `<strong>${esc(info.title)}</strong>`], ["What happens", esc(info.text)], ...info.live]);

  const pictures = status.cameras.reduce((sum, c) => sum + c.pictures, 0);
  if (lastPictureTotal !== null && pictures > lastPictureTotal) launchPacket();
  lastPictureTotal = pictures;
  // A picture also counts as a sign of life, so only new readings or an alive later than the picture mean a health answer
  const healthKey = status.cameras.map((c) => `${c.temp_c}|${c.node.uptime_s}|${c.last_alive_at !== c.last_picture_at ? c.last_alive_at : ""}`).join(",");
  if (lastHealthKey !== null && healthKey !== lastHealthKey) launchPulse();
  lastHealthKey = healthKey;
}

/* A run always finishes: a new picture during a run queues one more run instead of restarting it. */
let packetBusy = false;
let packetQueued = false;
function launchPacket() {
  if (reducedMotion.matches) return;
  const packet = el("packet");
  const anim = el("packet-anim");
  if (!packet || !anim || typeof anim.beginElement !== "function") return;
  if (packetBusy) {
    packetQueued = true;
    return;
  }
  packetBusy = true;
  packet.dataset.runs = Number(packet.dataset.runs) + 1;
  packet.classList.add("moving");
  anim.beginElement();
  setTimeout(() => {
    packetBusy = false;
    packet.classList.remove("moving");
    if (packetQueued) {
      packetQueued = false;
      launchPacket();
    }
  }, PACKET_MS);
}

/* The health pulse is short: a new answer during a pulse is simply not shown again. */
let pulseBusy = false;
function launchPulse() {
  const pulse = el("pulse");
  const anim = el("pulse-anim");
  if (reducedMotion.matches || pulseBusy || !pulse || !anim || typeof anim.beginElement !== "function") return;
  pulseBusy = true;
  pulse.classList.add("moving");
  anim.beginElement();
  setTimeout(() => {
    pulseBusy = false;
    pulse.classList.remove("moving");
  }, PULSE_MS);
}

/* Zoom: the view eases towards a target each frame. Scale 1 is the overview; inside band() the step labels cross-fade to
   the parts inside, while the pane grows from the strip to its tall height. */
const flow = el("flow");
const svg = flow.querySelector("svg");
const VIEW = { w: 1200, h: 160 };
const ZOOM = { min: 1, max: 6, growFrom: 1.3, detailFrom: 1.8, detailTo: 2.2, step: 1.5, snapMs: 200 };
const BOX = { w: 150, h: 84, cy: 80, padX: 14, padY: 18 }; // padY leaves room for the step buttons above the box
const STEPS = [...flow.querySelectorAll(".node")].map((g) => ({ id: g.dataset.node, cx: Number(g.querySelector(".node-box").getAttribute("x")) + BOX.w / 2 }));
const view = { scale: 1, x: 0, y: 0 };
const target = { scale: 1, x: 0, y: 0 };
let frame = 0;
let drag = null;
let dragged = false;
let snapTimer = 0;

const clamp01 = (v) => Math.min(1, Math.max(0, v));
/* The fixed band moves below the fit scale where the fit is small (wide, short panes), so a step fitted by a snap or the
   step buttons is always fully detail and scale 1 always fully overview. */
function band() {
  const fit = Math.max(1.25, fitScale());
  return {
    grow: Math.min(ZOOM.growFrom, 1 + 0.3 * (fit - 1)),
    from: Math.min(ZOOM.detailFrom, 1 + 0.6 * (fit - 1)),
    to: Math.min(ZOOM.detailTo, 1 + 0.9 * (fit - 1)),
  };
}
const detailOf = (s) => {
  const { from, to } = band();
  return clamp01((s - from) / (to - from));
};
const paneWidth = () => svg.getBoundingClientRect().width;
const stripHeight = () => (paneWidth() * VIEW.h) / VIEW.w;
const tallHeight = () => Math.max(stripHeight(), Math.min(440, 0.6 * window.innerHeight));

function paneHeight(s) {
  const { grow, to } = band();
  const t = clamp01((s - grow) / (to - grow));
  return stripHeight() + (tallHeight() - stripHeight()) * t * t * (3 - 2 * t);
}

/* Width and height of the view in diagram units; it takes the pane's pixel shape, so a zoomed view is not a thin slice. */
function viewSize(s) {
  const w = VIEW.w / s;
  return [w, (w * paneHeight(s)) / paneWidth()];
}

/* The scale at which one whole step box with margin fills the tall pane */
function fitScale() {
  const w = Math.max(BOX.w + 2 * BOX.padX, ((BOX.h + 2 * BOX.padY) * paneWidth()) / tallHeight());
  return Math.min(ZOOM.max, VIEW.w / w);
}

const nearestStep = (x) => STEPS.reduce((best, step, i) => (Math.abs(step.cx - x) < Math.abs(STEPS[best].cx - x) ? i : best), 0);
const currentStep = () => nearestStep(target.x + viewSize(target.scale)[0] / 2);
const goToStep = (i) => zoomTo(fitScale(), STEPS[i].cx, BOX.cy);

function clampView(v) {
  v.scale = Math.min(ZOOM.max, Math.max(ZOOM.min, v.scale));
  const [w, h] = viewSize(v.scale);
  const pad = Math.max(0, (w - BOX.w) / 2) * detailOf(v.scale); // lets the first and last step sit in the middle too
  v.x = Math.min(VIEW.w - w + pad, Math.max(-pad, v.x));
  v.y = h >= VIEW.h ? (VIEW.h - h) / 2 : Math.min(VIEW.h - h, Math.max(0, v.y));
}

function applyView() {
  clampView(view);
  const s = view.scale;
  const [w, h] = viewSize(s);
  svg.style.height = `${paneHeight(s)}px`;
  svg.setAttribute("viewBox", `${view.x} ${view.y} ${w} ${h}`);
  const current = STEPS[nearestStep(view.x + w / 2)].id;
  flow.querySelectorAll("[data-step]").forEach((b) => (b.dataset.step === current ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
  const detail = detailOf(s);
  svg.style.setProperty("--detail", detail);
  flow.dataset.level = detail === 0 ? "overview" : detail === 1 ? "detail" : "mixed";
  flow.classList.toggle("zoomed", s > 1.001);
  el("packet").setAttribute("r", 7 / s); // constant size on screen
  svg.querySelectorAll("marker").forEach((m) => {
    m.setAttribute("markerWidth", Number(m.dataset.size) / s);
    m.setAttribute("markerHeight", Number(m.dataset.size) / s);
  });
}

function animateView() {
  if (frame) return;
  const tick = () => {
    frame = 0;
    const done = reducedMotion.matches
      || (Math.abs(target.scale - view.scale) < 0.001 && Math.abs(target.x - view.x) < 0.05 && Math.abs(target.y - view.y) < 0.05);
    if (done) {
      Object.assign(view, target);
    } else {
      view.scale += (target.scale - view.scale) * 0.22;
      view.x += (target.x - view.x) * 0.22;
      view.y += (target.y - view.y) * 0.22;
    }
    applyView();
    if (!done) frame = requestAnimationFrame(tick);
  };
  frame = requestAnimationFrame(tick);
}

function pointerFraction(event) {
  const rect = svg.getBoundingClientRect();
  return [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height];
}

/* Zoom by a factor, keeping the point at fraction (fx, fy) of the picture fixed. False when already at the limit. */
function zoomAt(factor, fx, fy) {
  const scale = Math.min(ZOOM.max, Math.max(ZOOM.min, target.scale * factor));
  if (scale === target.scale) return false;
  const [w0, h0] = viewSize(target.scale);
  const [w1, h1] = viewSize(scale);
  const px = target.x + fx * w0;
  const py = target.y + fy * h0;
  target.scale = scale;
  target.x = px - fx * w1;
  target.y = py - fy * h1;
  clampView(target);
  animateView();
  return true;
}

function zoomTo(scale, cx, cy) {
  const [w, h] = viewSize(scale);
  target.scale = scale;
  target.x = cx - w / 2;
  target.y = cy - h / 2;
  clampView(target);
  animateView();
}

const resetZoom = () => zoomTo(1, VIEW.w / 2, VIEW.h / 2);

/* When the wheel stops at the detail level, centre the step under the pointer at the scale that shows its whole box. */
function scheduleSnap(fx) {
  clearTimeout(snapTimer);
  const x = target.x + fx * viewSize(target.scale)[0];
  snapTimer = setTimeout(() => {
    const { from, to } = band();
    if (target.scale >= (from + to) / 2) goToStep(nearestStep(x)); // never rest half-faded: finish the fade
    else if (target.scale > from) zoomAt(from / target.scale, fx, 0.5); // or settle back before it starts
  }, ZOOM.snapMs);
}

svg.addEventListener("wheel", (event) => {
  const rect = svg.getBoundingClientRect();
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1;
  const [fx, fy] = pointerFraction(event);
  if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) { // sideways: pan when zoomed in, else let the page scroll
    if (target.scale <= 1) return;
    target.x += ((event.deltaX * unit) / rect.width) * viewSize(target.scale)[0];
    clampView(target);
    animateView();
  } else {
    const factor = Math.min(2, Math.max(0.5, Math.exp(-event.deltaY * unit * 0.0022)));
    if (!zoomAt(factor, fx, fy)) return;
  }
  event.preventDefault();
  scheduleSnap(fx);
}, { passive: false });

svg.addEventListener("pointerdown", (event) => {
  dragged = false;
  if (event.button !== 0 || target.scale <= 1) return;
  clearTimeout(snapTimer);
  drag = { x: event.clientX, y: event.clientY, vx: target.x, vy: target.y };
});
svg.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const dx = event.clientX - drag.x;
  const dy = event.clientY - drag.y;
  if (!dragged && Math.hypot(dx, dy) < 4) return;
  if (!dragged) svg.setPointerCapture(event.pointerId); // only a real drag captures, so a plain click still reaches its step
  dragged = true;
  flow.classList.add("dragging");
  const rect = svg.getBoundingClientRect();
  const [w, h] = viewSize(target.scale);
  target.x = drag.vx - (dx / rect.width) * w;
  target.y = drag.vy - (dy / rect.height) * h;
  clampView(target);
  Object.assign(view, target); // follows the pointer directly, no easing
  applyView();
});
const endDrag = () => {
  drag = null;
  flow.classList.remove("dragging");
};
svg.addEventListener("pointerup", endDrag);
svg.addEventListener("pointercancel", endDrag);

svg.addEventListener("dblclick", (event) => {
  event.preventDefault();
  const node = event.target.closest(".node");
  if (node && target.scale < fitScale() - 0.01) {
    goToStep(STEPS.findIndex((step) => step.id === node.dataset.node));
  } else {
    zoomAt(ZOOM.step, ...pointerFraction(event));
  }
});

el("flow-zoom-in").addEventListener("click", () => zoomAt(ZOOM.step, 0.5, 0.5));
el("flow-zoom-out").addEventListener("click", () => zoomAt(1 / ZOOM.step, 0.5, 0.5));
el("flow-zoom-reset").addEventListener("click", resetZoom);
el("flow-prev").addEventListener("click", () => goToStep(Math.max(0, currentStep() - 1)));
el("flow-next").addEventListener("click", () => goToStep(Math.min(STEPS.length - 1, currentStep() + 1)));
flow.querySelectorAll("[data-step]").forEach((b, i) => b.addEventListener("click", () => goToStep(i)));
flow.addEventListener("keydown", (event) => {
  if (target.scale <= 1 || (event.key !== "ArrowLeft" && event.key !== "ArrowRight")) return;
  event.preventDefault();
  el(event.key === "ArrowLeft" ? "flow-prev" : "flow-next").click();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && target.scale > 1) resetZoom();
});
window.addEventListener("resize", () => {
  clampView(target);
  animateView();
});

document.querySelectorAll("#flow .node").forEach((g) => {
  ensureInside(g, "fake");
  const select = () => {
    if (dragged) return;
    selectedNode = g.dataset.node;
    if (lastStatus) renderFlow(lastStatus);
  };
  g.addEventListener("click", select);
  g.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      select();
    }
  });
});
applyView();

/* Dashboard below the diagram: cards are built once per id and then patched, so bars and gauges ease and hover text survives the refresh */
const GAUGE_C = [20, 90];
const BLOCK_SCALE_BYTES = 200 * 1024;
const UPTIME_WINDOW_S = 3600;
const TREND_S = 600;
const ACTIVITY_S = 300;
const trends = { cameras: [], racks: [], free: [], latency: [] };
const seenPictures = new Map();
const seenErrors = new Map();
const marks = [];
let markId = 0;
let lastPollAt = null;
let lastOkAt = null;

const MORE_ICONS = {
  push: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 9V1.5M2 4.5l3-3 3 3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  pull: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 1v7.5M2 5.5l3 3 3-3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  reboot: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M8.6 5.4A3.6 3.6 0 1 1 5 1.4h1.4M5.2 0l1.6 1.4L5.2 2.8" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  vision: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M1 5c1.2-2.2 2.5-3.2 4-3.2S7.8 2.8 9 5c-1.2 2.2-2.5 3.2-4 3.2S2.2 7.2 1 5z" fill="none" stroke="currentColor" stroke-width="1.2"/><circle cx="5" cy="5" r="1.4" fill="currentColor"/></svg>',
};
const PREVIEW_EVERY_S = 3;

/* A rack decided by the vision service: what the service reports for its camera, then the app's own view */
function visionLine(v, rack) {
  const where = v.how === "pull" ? `asked every ${v.poll_s} s at ${esc(v.url)}` : "pushed by the service";
  if (v.updated_at === undefined) return `<b>Vision service</b>, camera ${esc(v.camera)}, ${where}: nothing received yet`;
  const counts = v.slot_count ? `${v.free} free, ${v.occupied} taken${v.unknown ? `, ${v.unknown} unknown` : ""} of ${v.slot_count} hooks` : "no slot layout on the service yet";
  const frame = v.frame_age_s === null || v.frame_age_s === undefined ? "no frame yet" : `frame ${ago(v.frame_age_s)}`;
  const notes = [v.online === false ? "camera offline at the service" : "", v.message, v.warning].filter(Boolean).map(esc).join("; ");
  const own = rack ? `. App: ${rack.status === "ok" ? `${rack.free} free` : rack.status}, last result ${ago(rack.age_s)}` : "";
  return `<b>Vision service</b>, camera ${esc(v.camera)}, ${where}: ${counts}, ${frame}${v.processing_ms ? `, ${Math.round(v.processing_ms)} ms` : ""}${notes ? `. ${notes}` : ""}${own}`;
}

/* The service's last preview (persons blurred there), fetched through the app; the src changes, the img stays */
function renderPreview(card, c, now) {
  const box = card.querySelector('[data-f="preview"]');
  if (!c.vision || !c.vision.preview) {
    if (box.firstChild) box.innerHTML = "";
    return;
  }
  let img = box.querySelector("img");
  if (!img) {
    box.innerHTML = '<img alt="Last picture as the vision service sees it: persons blurred, hooks coloured free or taken" loading="lazy">';
    img = box.querySelector("img");
    img._at = 0;
  }
  if (now - img._at >= PREVIEW_EVERY_S) {
    img._at = now;
    img.src = `/api/vision/${encodeURIComponent(c.rack)}/preview.jpg?t=${Date.now()}`;
  }
}
const icon = (kind, name) => `<i class="ico ico-${kind}">${ICONS[name]}</i>`;

function field(root, name, html) {
  const node = root.querySelector(`[data-f="${name}"]`);
  if (node._html !== html) {
    node.innerHTML = html;
    node._html = html;
  }
  return node;
}

/* Counts roll to a new value instead of jumping */
function roll(node, value) {
  const from = node._value;
  node._value = value;
  if (from === undefined || from === null || value === null || from === value || reducedMotion.matches) {
    node.textContent = value === null ? "?" : String(value);
    return;
  }
  const start = performance.now();
  const step = (time) => {
    if (node._value !== value) return;
    const t = Math.min(1, (time - start) / 500);
    node.textContent = String(Math.round(from + (value - from) * (1 - (1 - t) ** 3)));
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function keyed(container, items, key, build, kind) {
  const keys = items.map(key);
  [...container.children].forEach((child) => keys.includes(child.dataset.key) || child.remove());
  return items.map((item, i) => {
    let card = [...container.children].find((child) => child.dataset.key === keys[i]);
    if (!card) {
      card = document.createElement("article");
      card.className = `card ${kind} enter`;
      card.dataset.key = keys[i];
      card.style.animationDelay = `${i * 60}ms`;
      card.innerHTML = build(item);
      container.append(card);
    }
    return card;
  });
}

function pulse(card) {
  card.classList.remove("enter", "pulse");
  card.style.animationDelay = "";
  void card.offsetWidth; // restarts the animation
  card.classList.add("pulse");
}

/* One short line with the age, the full text on hover */
function errorLine(node, text, age) {
  node.hidden = !text;
  if (!text) return;
  node.title = text;
  node.querySelector("span").textContent = `${short(text, 46)}${age === null || age === undefined ? "" : `, ${ago(age)}`}`;
}

/* Sparkline of a series kept in the browser over the last `span` seconds; y is value / max */
function trendSvg(series, now, span, max, label, unit = "") {
  const w = 160;
  const h = 32;
  const pts = series.filter((p) => p.t >= now - span);
  const base = `<line class="spark-base" x1="0" y1="${h - 0.5}" x2="${w}" y2="${h - 0.5}"/>`;
  if (pts.length < 2) return `<svg class="spark" viewBox="0 0 ${w} ${h}" aria-hidden="true">${base}</svg>`;
  const from = Math.max(now - span, pts[0].t); // until the window is full, the line uses the whole width
  const x = (t) => 2 + ((t - from) / Math.max(1, now - from)) * (w - 6);
  const y = (v) => 3 + (1 - Math.min(1, v / (max || 1))) * (h - 6);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)} ${y(p.v).toFixed(1)}`).join("");
  const values = pts.map((p) => p.v);
  const last = pts[pts.length - 1];
  const text = `${label}: now ${last.v}${unit}, low ${Math.min(...values)}${unit}, high ${Math.max(...values)}${unit}`;
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(text)}"><title>${esc(text)}</title>${base}
    <path class="spark-area" d="${line}L${x(last.t).toFixed(1)} ${h - 0.5}L${x(pts[0].t).toFixed(1)} ${h - 0.5}Z"/>
    <path class="spark-line" d="${line}"/>
    <circle class="spark-dot" cx="${x(last.t).toFixed(1)}" cy="${y(last.v).toFixed(1)}" r="3"/></svg>`;
}

/* What changed since the last poll: picture ticks, error marks, trend samples. Returns the racks whose picture count rose. */
function notePoll(status) {
  const now = status.generated_at;
  const since = lastPollAt ?? now;
  const rose = new Set();
  const mark = (row, kind, t, text) => marks.push({ id: String((markId += 1)), row, kind, t, text });
  status.cameras.forEach((c) => {
    const before = seenPictures.get(c.rack);
    if (before === undefined) {
      if (c.last_picture_at !== null && now - c.last_picture_at <= ACTIVITY_S) mark(c.rack, "picture", c.last_picture_at, `Picture from ${c.rack}`);
    } else if (c.pictures > before) {
      rose.add(c.rack);
      const n = Math.min(c.pictures - before, 20);
      const end = c.last_picture_at ?? now;
      const start = Math.min(since, end);
      for (let i = 1; i <= n; i += 1) mark(c.rack, "picture", start + ((end - start) * i) / n, `Picture from ${c.rack}`);
    }
    seenPictures.set(c.rack, c.pictures);
    if (c.last_error_at !== null && seenErrors.get(c.rack) !== c.last_error_at) {
      seenErrors.set(c.rack, c.last_error_at);
      if (now - c.last_error_at <= ACTIVITY_S) mark(c.rack, "camera-error", c.last_error_at, `Camera ${c.rack} error: ${c.last_error}`);
    }
  });
  const r = status.detector_results;
  if (r.last_error_at !== null && seenErrors.get("detector") !== r.last_error_at) {
    seenErrors.set("detector", r.last_error_at);
    if (now - r.last_error_at <= ACTIVITY_S) mark("detector", "detector-error", r.last_error_at, `Detector error: ${r.last_error}`);
  }
  if (r.last_ok_at !== null && r.last_ok_at !== lastOkAt && r.last_latency_ms !== null) trends.latency.push({ t: r.last_ok_at, v: r.last_latency_ms });
  lastOkAt = r.last_ok_at;
  const live = status.racks.filter((k) => k.status === "ok");
  trends.cameras.push({ t: now, v: status.cameras.filter((c) => c.online).length });
  trends.racks.push({ t: now, v: live.length });
  trends.free.push({ t: now, v: live.reduce((sum, k) => sum + k.free, 0) });
  Object.values(trends).forEach((series) => {
    while (series.length && series[0].t < now - TREND_S) series.shift();
  });
  while (marks.length && marks[0].t < now - ACTIVITY_S) marks.shift();
  marks.sort((a, b) => a.t - b.t);
  lastPollAt = now;
  return rose;
}

function renderKpis(status) {
  const box = el("kpis");
  if (!box.dataset.built) {
    box.innerHTML = ["server", "detector", "cameras", "racks", "free"].map((key) => `<div class="stat" data-key="${key}">
      <p class="label" data-f="label"></p><p class="stat-value"><span data-f="n"></span><span class="of" data-f="of"></span></p>
      <p class="sub" data-f="sub"></p><div class="stat-trend" data-f="trend"></div></div>`).join("");
    box.dataset.built = "1";
  }
  const tile = (key) => box.querySelector(`[data-key="${key}"]`);
  const now = status.generated_at;
  const cams = status.cameras;
  const d = status.detector;
  const live = status.racks.filter((k) => k.status === "ok");
  const demoCams = cams.filter((c) => c.demo).length;
  const count = (key, label, value, of, sub, series, max) => {
    const t = tile(key);
    field(t, "label", label);
    roll(t.querySelector('[data-f="n"]'), value);
    field(t, "of", of);
    field(t, "sub", sub);
    field(t, "trend", trendSvg(series, now, TREND_S, max, `${label}, last 10 minutes`));
  };
  let t = tile("server");
  field(t, "label", "Server");
  field(t, "n", `v${esc(status.app.version)}`);
  field(t, "sub", `up ${ago(status.app.uptime_s).replace(" ago", "")}`);
  t = tile("detector");
  field(t, "label", "Detector");
  field(t, "n", d.reachable ? chip("good", "dot", "Reachable") : chip("critical", "square", "Down"));
  field(t, "sub", d.mode === "fake" ? "built-in stand-in" : esc(d.url));
  count("cameras", "Cameras online", cams.filter((c) => c.online).length, ` / ${cams.length}`,
    cams.some((c) => c.too_hot) ? `${icon("warning", "tri")}one or more too hot` : demoCams ? `${demoCams} fed with demo data` : "alive within the stale window",
    trends.cameras, cams.length);
  count("racks", "Racks live", live.length, ` / ${status.racks.length}`, live.length === status.racks.length ? "all racks reporting" : "some without live data", trends.racks, status.racks.length);
  count("free", "Free spaces now", live.reduce((sum, k) => sum + k.free, 0), ` / ${live.reduce((sum, k) => sum + k.capacity, 0)}`, "on the live racks", trends.free, live.reduce((sum, k) => sum + k.capacity, 0));
}

function renderActivity(status) {
  const box = el("activity");
  const now = status.generated_at;
  const rows = [...status.cameras.map((c) => [c.rack, c.rack, status.racks.find((k) => k.id === c.rack)?.name ?? c.rack]), ["detector", "Detector", "Detector"]];
  const shape = rows.map(([key]) => key).join();
  if (box.dataset.rows !== shape) {
    box.innerHTML = `${rows.map(([key, label, title]) => `<div class="act-row" data-row="${esc(key)}"><span class="act-label" title="${esc(title)}">${esc(label)}</span><div class="act-track"></div></div>`).join("")}
      <div class="act-axis" aria-hidden="true"><span>5 min ago</span><span>now</span></div>
      <p class="act-key"><span><i class="key-tick"></i>new picture</span><span>${icon("critical", "tri")}camera error</span><span>${icon("critical", "square")}detector error</span></p>`;
    box.dataset.rows = shape;
  }
  const ids = new Set(marks.map((m) => m.id));
  box.querySelectorAll(".mark").forEach((node) => ids.has(node.dataset.id) || node.remove());
  const at = (t) => `${((100 * (t - (now - ACTIVITY_S))) / ACTIVITY_S).toFixed(2)}%`;
  marks.forEach((m) => {
    const track = box.querySelector(`.act-row[data-row="${CSS.escape(m.row)}"] .act-track`);
    if (!track) return;
    let node = track.querySelector(`[data-id="${m.id}"]`);
    if (!node) {
      node = document.createElement("span");
      node.className = `mark mark-${m.kind}`;
      node.dataset.id = m.id;
      node.title = `${m.text}, ${new Date(m.t * 1000).toLocaleTimeString()}`;
      node.innerHTML = m.kind === "picture" ? "" : icon("critical", m.kind === "camera-error" ? "tri" : "square");
      node.style.left = at(m.t + 3); // so a new mark slides in with the rest
      track.append(node);
    }
    node.style.left = at(m.t);
  });
  const count = (kind) => marks.filter((m) => m.kind === kind).length;
  box.setAttribute("aria-label", `Last 5 minutes: ${count("picture")} pictures, ${count("camera-error")} camera errors, ${count("detector-error")} detector errors`);
}

function renderDetector(status) {
  const box = el("detector");
  const d = status.detector;
  const r = status.detector_results;
  if (!box.dataset.built) {
    box.innerHTML = `<div class="health-head"><p data-f="chip"></p><p class="health-mode" data-f="mode"></p></div>
      <div class="health-facts"><p><span class="k">Model</span><b data-f="model"></b></p><p><span class="k">Last result</span><b data-f="ok"></b></p></div>
      <p class="health-vision" data-f="vision" hidden></p>
      <div class="health-latency"><p class="k">Latency, last 5 minutes</p><div data-f="spark"></div><p class="health-now" data-f="lat"></p></div>
      <p class="card-error" hidden>${ICONS.square}<span></span></p>`;
    box.dataset.built = "1";
  }
  box.dataset.state = d.reachable ? "good" : "critical";
  field(box, "chip", d.reachable ? chip("good", "dot", "Reachable") : chip("critical", "square", "Down"));
  field(box, "mode", d.mode === "fake" ? "Built-in stand-in (fake)" : `Service at ${esc(d.url)}`);
  field(box, "model", esc(r.model ?? d.info.model ?? "unknown"));
  field(box, "ok", ago(r.ok_age_s));
  field(box, "spark", trendSvg(trends.latency, status.generated_at, ACTIVITY_S, Math.max(10, ...trends.latency.map((p) => p.v)), "Latency, last 5 minutes", " ms"));
  field(box, "lat", r.last_latency_ms === null ? "no call yet" : `last call ${r.last_latency_ms} ms`);
  const vision = status.cameras.filter((c) => c.vision).map((c) => `${esc(c.rack)}: ${c.vision.how === "pull" ? esc(c.vision.url) : "pushed by the service"}, camera ${esc(c.vision.camera)}, ${c.online ? "answering" : "silent"}`).join("; ");
  field(box, "vision", vision ? `<span class="k">Vision service (decides its racks on the laptop)</span><br>${vision}` : "").hidden = !vision;
  errorLine(box.querySelector(".card-error"), r.last_error ?? d.error, r.last_error ? r.error_age_s : null);
}

const GAUGE_ARC = "M6 34A26 26 0 0 1 58 34";
const tempShare = (t) => Math.min(1, Math.max(0, (t - GAUGE_C[0]) / (GAUGE_C[1] - GAUGE_C[0])));
const gaugeAt = (share, radius) => [32 + radius * Math.cos(Math.PI * (1 - share)), 34 - radius * Math.sin(Math.PI * (1 - share))];

function cameraSkeleton(c) {
  const [x1, y1] = gaugeAt(tempShare(HOT_LIMIT_C), 20);
  const [x2, y2] = gaugeAt(tempShare(HOT_LIMIT_C), 32);
  return `<header class="card-head"><p class="card-title"><span class="card-id">${esc(c.rack)}</span><span data-f="name"></span></p><p class="card-chips" data-f="chips"></p></header>
    <div class="instruments">
      <figure class="instrument"><div class="visual" data-f="bars"></div><figcaption data-f="wifi"></figcaption></figure>
      <figure class="instrument"><div class="visual"><svg class="gauge" viewBox="0 0 64 38" role="img">
        <path class="gauge-track" d="${GAUGE_ARC}" pathLength="100"/><path class="gauge-fill" d="${GAUGE_ARC}" pathLength="100" style="stroke-dasharray: 0 100"/>
        <line class="gauge-limit" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/></svg></div><figcaption data-f="temp"></figcaption></figure>
      <figure class="instrument"><div class="visual"><div class="block-bar"><div class="block-fill"></div><span class="block-mark" style="left: ${(100 * LOW_BLOCK_BYTES) / BLOCK_SCALE_BYTES}%"></span></div></div><figcaption data-f="mem"></figcaption></figure>
    </div>
    <div class="uptime"><div class="uptime-bar"><div class="uptime-fill"></div><span class="uptime-reboot" hidden>${MORE_ICONS.reboot}</span></div><p class="uptime-text" data-f="uptime"></p></div>
    <p class="card-detect" data-f="detect"></p>
    <div class="card-preview" data-f="preview"></div>
    <p class="card-foot"><span data-f="node"></span><span class="tag" data-f="source"></span><span><b class="pictures">0</b> pictures</span><span data-f="seen"></span></p>
    <p class="card-error" hidden>${ICONS.square}<span></span></p>`;
}

function wifiBars(dbm) {
  if (dbm === null) return '<svg class="bars" viewBox="0 0 30 24" aria-hidden="true">' + [1, 2, 3, 4].map((i) => `<rect class="bar" x="${(i - 1) * 8}" y="${24 - i * 6}" width="6" height="${i * 6}" rx="1.5"/>`).join("") + "</svg>";
  const level = dbm >= -60 ? 4 : dbm >= -67 ? 3 : dbm >= -75 ? 2 : dbm >= -85 ? 1 : 0;
  const bars = [1, 2, 3, 4].map((i) => `<rect class="bar${i <= level ? " on" : ""}" x="${(i - 1) * 8}" y="${24 - i * 6}" width="6" height="${i * 6}" rx="1.5"/>`).join("");
  return `<svg class="bars" data-level="${signal(dbm)}" viewBox="0 0 30 24" role="img" aria-label="WiFi ${dbm} dBm, ${signal(dbm)}, ${level} of 4 bars">${bars}</svg>`;
}

function renderCameras(status, rose) {
  const now = status.generated_at;
  keyed(el("cameras"), status.cameras, (c) => c.rack, cameraSkeleton, "cam").forEach((card, i) => {
    const c = status.cameras[i];
    const n = c.node;
    card.dataset.state = c.too_hot ? "warning" : c.online ? "good" : c.last_alive_at === null ? "muted" : "critical";
    field(card, "name", esc(status.racks.find((k) => k.id === c.rack)?.name ?? ""));
    const rack = status.racks.find((k) => k.id === c.rack);
    field(card, "chips", `${cameraChip(c)}${rack ? rackChip(rack) : ""}${demoChip(c, status)}`);

    const dbm = n.wifi_rssi_dbm;
    field(card, "bars", wifiBars(dbm));
    field(card, "wifi", dbm === null ? "<b>n/a</b><span>WiFi</span>" : `<b>${dbm} dBm</b><span>${signal(dbm) === "weak" ? icon("warning", "tri") : ""}WiFi ${signal(dbm)}</span>`);

    const temp = c.temp_c;
    const gauge = card.querySelector(".gauge");
    gauge.querySelector(".gauge-fill").style.strokeDasharray = `${temp === null ? 0 : (tempShare(temp) * 100).toFixed(1)} 100`;
    gauge.dataset.level = c.too_hot ? "hot" : "ok";
    gauge.setAttribute("aria-label", temp === null ? "Chip temperature not reported" : `Chip temperature ${temp} C, limit ${HOT_LIMIT_C} C`);
    field(card, "temp", temp === null ? `<b>n/a</b><span>Chip, limit ${HOT_LIMIT_C} C</span>`
      : `<b>${Math.round(temp)} C</b><span>${c.too_hot ? `${icon("warning", "tri")}over ${HOT_LIMIT_C} C` : `Chip, limit ${HOT_LIMIT_C} C`}</span>`);

    const block = n.heap_largest_block;
    const low = block !== null && block < LOW_BLOCK_BYTES;
    card.querySelector(".block-fill").style.width = `${block === null ? 0 : Math.min(100, (100 * block) / BLOCK_SCALE_BYTES).toFixed(1)}%`;
    card.querySelector(".block-bar").dataset.level = low ? "low" : "ok";
    card.querySelector(".block-bar").title = `Largest free memory block; below ${LOW_BLOCK_BYTES / 1024} KB pictures may fail`;
    field(card, "mem", block === null ? "<b>n/a</b><span>Free block</span>" : `<b>${Math.round(block / 1024)} KB</b><span>${low ? `${icon("warning", "tri")}Low, ` : ""}free block</span>`);

    const up = n.uptime_s;
    const rebooted = n.rebooted_at !== null && now - n.rebooted_at < REBOOT_SHOWN_S;
    const share = up === null ? 0 : Math.min(1, up / UPTIME_WINDOW_S);
    card.querySelector(".uptime-fill").style.width = `${(share * 100).toFixed(1)}%`;
    card.querySelector(".uptime-bar").title = "The last hour: the filled part is time since the node started";
    const marker = card.querySelector(".uptime-reboot");
    marker.hidden = !rebooted;
    marker.style.right = `${(share * 100).toFixed(1)}%`;
    field(card, "uptime", up === null ? "Uptime not reported" : `Up ${ago(up).replace(" ago", "")}${rebooted ? `, <b>rebooted</b> ${ago(now - n.rebooted_at)}` : ""}`);

    field(card, "node", esc(c.node_id ?? "unknown node"));
    const source = card.querySelector('[data-f="source"]');
    field(card, "source", c.source ? `${MORE_ICONS[c.source] ?? ""}${esc(c.source)}` : "");
    source.title = c.source === "pull" ? "The server fetches pictures from the node" : c.source === "push" ? "The node sends its pictures"
      : c.source === "vision" ? "The vision service on the laptop decides the spaces; the app reads them" : "";
    roll(card.querySelector(".pictures"), c.pictures);
    const bytes = c.last_picture_bytes;
    field(card, "seen", `last ${ago(c.picture_age_s)}${bytes ? `, ${bytes < 1024 ? `${bytes} B` : `${Math.round(bytes / 1024)} KB`}` : ""}`);
    field(card, "detect", c.vision ? visionLine(c.vision, rack) : !rack ? "" : rack.mode === "slots" ? `<b>By space</b>, ${rack.slots_ready} of ${rack.slots_observed} spaces ready, last result ${ago(rack.age_s)}`
      : rack.mode === "count" ? `<b>By count</b>, last result ${ago(rack.age_s)}` : "No detection yet");
    card.querySelector('[data-f="detect"]').title = c.vision
      ? "What the vision service reports for this rack's camera; the app shows its decided spaces as they are"
      : `A space is ready once it has a full window of ${status.app.free_after_clean} pictures`;
    renderPreview(card, c, now);
    errorLine(card.querySelector(".card-error"), c.last_error, c.error_age_s);
    if (rose.has(c.rack)) pulse(card);
  });
}

function render(status) {
  lastStatus = status;
  const rose = notePoll(status);
  renderKpis(status);
  renderFlow(status);
  renderActivity(status);
  renderDetector(status);
  renderCameras(status, rose);
  el("updated").textContent = `Refreshed ${new Date().toLocaleTimeString()}. Updates every 3 seconds.`;
}

async function refresh() {
  try {
    const response = await fetch("/api/admin/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (err) {
    el("updated").textContent = `Server unreachable (${err.message}). Retrying.`;
  }
}

refresh();
setInterval(refresh, 3000);
