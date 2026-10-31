"""Showcase page at /demo: live camera + annotated example images explaining what works so far.

Examples are computed on the fly from the images in ``assets/pictures`` (cached in RAM, pre-warmed in the
background at service start):
  * classifier - every picture in assets/pictures/DemoPictures with a <name>.layout.json next to it (the fixed
              slot polygons of the slot-classifier approach): per-slot free/occupied from the small CNN, compared
              with <name>.labels.json where it exists
  * golden  - hand-labelled layout, occupancy compared with the hand labels
  * auto3   - the reference scene calibrated automatically from its 3 images (different occupancy each),
              then occupancy of every image compared with the hand labels (no VLM, no admin)
  * auto    - layout proposed automatically from that single image (stands + bikes, no VLM, no admin),
              then occupancy on the same image
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

from . import config
from .geometry import imread, resize_max_side
from .occupancy import SlotMaskCache, analyze_frame, make_trackers
from .preview import render
from .schemas import Layout
from .store import golden_truth, layout_from_golden

router = APIRouter(prefix="/demo", tags=["demo"], include_in_schema=False)
GPT_DIR = config.ROOT / "assets" / "pictures" / "GeneratedByChatGPT"
SELFMADE_DIR = config.ROOT / "assets" / "pictures" / "Selfmade"
DEMO_PICTURES_DIR = config.ROOT / "assets" / "pictures" / "DemoPictures"

_lock = threading.Lock()   # one example computed at a time
_gpu_lock = None           # the store lock, so examples never run the detector concurrently with live frames
_cache: dict[str, dict] = {}


def bind(store_lock) -> None:
    global _gpu_lock
    _gpu_lock = store_lock


def examples() -> list[dict]:
    out: list[dict] = []
    for p in sorted(DEMO_PICTURES_DIR.glob("*.jpeg")):   # pictures with slot polygons drawn: the classifier approach
        layout = p.with_suffix(".layout.json")
        if layout.exists():
            out.append({"name": f"classifier_{p.stem}", "kind": "classifier", "path": p, "layout": layout,
                        "labels": p.with_suffix(".labels.json")})
    gold = sorted(config.GOLDEN_DIR.glob("*.png"))
    out += [{"name": p.stem, "kind": "golden", "path": p} for p in gold]
    if gold:
        out.append({"name": "reference_auto_3frames", "kind": "auto3", "path": gold[-1], "paths": gold})
    out += [{"name": p.stem, "kind": "auto", "path": p} for p in sorted(GPT_DIR.glob("*.png"))]
    out += [{"name": p.stem, "kind": "selfmade", "path": p} for p in sorted(SELFMADE_DIR.glob("*.jpeg"))]
    out += [{"name": p.stem, "kind": "demo_picks", "path": p} for p in sorted(DEMO_PICTURES_DIR.glob("*.jpeg"))]
    return out


def _compute_auto3(ex: dict) -> dict:
    """Calibrate from all golden images, then score every image against the hand labels."""
    from .calibrate import propose_layout
    t0 = time.time()
    imgs = [imread(str(p)) for p in ex["paths"]]
    frames = [resize_max_side(im, config.MAX_SIDE)[0] for im in imgs if im is not None]
    cam = SimpleNamespace(camera_id=ex["name"], frames=deque((None, f) for f in frames), layout=Layout(camera_id=ex["name"]))
    layout = propose_layout(cam, use_vlm=False, use_stands=True)
    layout.params = {**(layout.params or {}), "k_frames": 1, "min_frames": 1}
    truths = {t["file"]: t["slots"] for t in golden_truth()}
    correct = total = 0
    res = None
    for p, im in zip(ex["paths"], imgs):
        if im is None:
            continue
        res = analyze_frame(layout, make_trackers(layout), SlotMaskCache(), im, ex["name"])
        pred = {s.slot_id.split("-")[-1]: s.status for s in res.status.slots}
        for sid, want in truths.get(p.name, {}).items():
            total += 1
            correct += pred.get(sid.split("-")[-1]) == want
    st = res.status
    info = {"name": ex["name"], "kind": ex["kind"], "slots": st.slot_count, "free": st.free, "occupied": st.occupied,
            "unknown": st.unknown, "bikes": st.detections.get("bicycle", 0), "persons": st.detections.get("person", 0),
            "ms": round((time.time() - t0) * 1000), "message": st.message, "correct": correct, "truth_slots": total,
            "frames": len(frames), "notes": layout.calibration_hint.notes if layout.calibration_hint else ""}
    info["jpeg"] = render(res.frame, st, res.detections, res.assignments, max_width=960)
    return info


def _compute_classifier(ex: dict) -> dict:
    """New approach: the slot polygons drawn once for the demo picture + the per-crop classifier."""
    import json

    from .slotcls import pipeline as slotcls_pipeline
    from .slotcls import smoothing

    img = imread(str(ex["path"]))
    if img is None:
        raise HTTPException(404, f"cannot read {ex['path'].name}")
    layout = Layout.model_validate_json(Path(ex.get("layout", config.DEMO_LAYOUT)).read_text(encoding="utf-8"))
    layout.params = {**(layout.params or {}), "k_frames": 1, "min_frames": 1}
    t0 = time.time()
    res = slotcls_pipeline.analyze_frame(layout, img, layout.camera_id, smoothing.make_trackers(layout))
    st = res.status
    info = {"name": ex["name"], "kind": ex["kind"], "file": ex["path"].name, "slots": st.slot_count, "free": st.free,
            "occupied": st.occupied, "unknown": st.unknown, "bikes": st.occupied, "persons": st.detections.get("person", 0),
            "ms": round((time.time() - t0) * 1000), "message": st.message}
    labels = Path(ex.get("labels", config.DEMO_LABELS))
    if labels.exists():
        frames = json.loads(labels.read_text(encoding="utf-8")).get("frames", [])
        truth = next((f["slots"] for f in frames if f["file"] == ex["path"].name), {})
        truth = {k: v for k, v in truth.items() if v in ("free", "occupied")}   # unknown = not scored
        if truth:
            pred = {x.slot_id: x.status for x in st.slots}
            info["correct"] = sum(pred.get(k) == v for k, v in truth.items())
            info["truth_slots"] = len(truth)
            info["wrong"] = sorted(int(k.split("-S")[-1]) for k, v in truth.items() if pred.get(k) != v)
    info["jpeg"] = render(res.frame, st, res.detections, None, max_width=960, show_detections=False, show_scores=True)
    return info


def _compute(ex: dict) -> dict:
    if ex["kind"] == "auto3":
        return _compute_auto3(ex)
    if ex["kind"] == "classifier":
        return _compute_classifier(ex)
    img = imread(str(ex["path"]))
    if img is None:
        raise HTTPException(404, f"cannot read {ex['path'].name}")
    t0 = time.time()
    info: dict = {"name": ex["name"], "kind": ex["kind"]}
    if ex["kind"] == "golden":
        layout = layout_from_golden()
        truth = next((t["slots"] for t in golden_truth() if t["file"] == ex["path"].name), {})
    else:
        from .calibrate import propose_layout
        frame, _ = resize_max_side(img, config.MAX_SIDE)
        cam = SimpleNamespace(camera_id=ex["name"], frames=deque([(None, frame)]), layout=Layout(camera_id=ex["name"]))
        layout = propose_layout(cam, use_vlm=False, use_stands=True)
        truth = {}
    layout.params = {**(layout.params or {}), "k_frames": 1, "min_frames": 1}
    res = analyze_frame(layout, make_trackers(layout), SlotMaskCache(), img, ex["name"])
    st = res.status
    info.update(slots=st.slot_count, free=st.free, occupied=st.occupied, unknown=st.unknown,
                bikes=st.detections.get("bicycle", 0), persons=st.detections.get("person", 0),
                ms=round((time.time() - t0) * 1000), message=st.message)
    if truth:
        pred = {s.slot_id: s.status for s in st.slots}
        info["correct"] = sum(pred.get(k) == v for k, v in truth.items())
        info["truth_slots"] = len(truth)
    info["jpeg"] = render(res.frame, st, res.detections, res.assignments, max_width=960)
    return info


def get(name: str) -> dict:
    ex = next((e for e in examples() if e["name"] == name), None)
    if ex is None:
        raise HTTPException(404, f"unknown example '{name}'")
    with _lock:
        if name not in _cache:
            if _gpu_lock is not None:
                with _gpu_lock:
                    _cache[name] = _compute(ex)
            else:
                _cache[name] = _compute(ex)
        return _cache[name]


def prewarm() -> None:
    for ex in examples():
        try:
            get(ex["name"])
        except Exception as exc:  # a broken example must not take the service down
            print(f"[bikeslot] demo example {ex['name']} failed: {exc}")
    print(f"[bikeslot] demo examples ready ({len(_cache)})")


@router.get("/examples")
def list_examples():
    return [{k: v for k, v in get(e["name"]).items() if k != "jpeg"} if e["name"] in _cache
            else {"name": e["name"], "kind": e["kind"], "pending": True} for e in examples()]


@router.get("/examples/{name}.jpg")
def example_jpg(name: str):
    return Response(get(name)["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "max-age=600"})


@router.get("/examples/{name}.json")
def example_json(name: str):
    return {k: v for k, v in get(name).items() if k != "jpeg"}


@router.get("", response_class=HTMLResponse)
def page():
    return _PAGE


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>bikeslot demo</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--muted:#5d6676;--line:#dde1e7;--free:#2f9e44;--occ:#d9363e;--unk:#8a8f98;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--fg:#e8ebf0;--muted:#9aa3b2;--line:#2c3340;--acc:#6ea0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 6px}p{margin:4px 0}.muted{color:var(--muted)}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.step b{display:block;margin-bottom:4px}.step .n{color:var(--acc);font-weight:700;margin-right:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;margin-top:12px}
.ex img{width:100%;border-radius:6px;display:block;background:var(--line);min-height:120px}
.ex .meta{margin-top:8px;font-size:14px}.ex .title{font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:13px;font-weight:600;margin-right:4px}
.free{background:color-mix(in srgb,var(--free) 18%,transparent);color:var(--free)}
.occ{background:color-mix(in srgb,var(--occ) 18%,transparent);color:var(--occ)}
.unk{background:color-mix(in srgb,var(--unk) 22%,transparent);color:var(--unk)}
.ok{color:var(--free);font-weight:700}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:14px;margin-top:10px}.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.live{display:grid;grid-template-columns:minmax(0,2fr) minmax(220px,1fr);gap:14px;margin-top:12px}
@media (max-width:760px){.live{grid-template-columns:1fr}.grid{grid-template-columns:1fr}}
.live img{width:100%;border-radius:6px;display:block}.big{font-size:40px;font-weight:700;line-height:1.1}
ul{margin:6px 0;padding-left:20px}code{font-size:13px;background:var(--line);padding:1px 5px;border-radius:4px}
</style></head><body><main>
<h1>bikeslot &mdash; Find my Bike Slot</h1>
<p class="muted">A camera photographs the bike rack every few seconds. The laptop finds the bikes, decides which numbered slots are taken, and tells the app how many are free.</p>

<div class="steps">
 <div class="card step"><b><span class="n">1</span>Snapshot</b>The camera serves a JPEG; the service fetches one every 5 s (no video stream).</div>
 <div class="card step"><b><span class="n">2</span>Detect</b>YOLO11 segmentation finds bicycles, motorbikes and persons (~100 ms on the GPU, fully offline). Persons are blurred.</div>
 <div class="card step"><b><span class="n">3</span>Assign</b>The lowest pixels of each bike mask = where the wheel touches the ground (dot). That point decides which slot polygon the bike stands in.</div>
 <div class="card step"><b><span class="n">4</span>Report</b>Votes are smoothed over 5 frames, then <code>/sites</code> and <code>/cameras/{id}/status</code> give free/occupied per slot.</div>
</div>
<p class="muted" style="margin-top:10px">Where the slots are is learned once per camera (<b>calibration</b>): rack stands are found with an open-vocabulary model, a perspective lattice (equal slots on the ground, shrinking towards the horizon) is fitted to stands, painted lines and parked bikes, which also places the stands hidden behind bikes. GPT-4.1 optionally counts the slots. An admin confirms or drags a corner.</p>
<div class="legend"><span><span class="sw" style="background:var(--free)"></span>free slot</span><span><span class="sw" style="background:var(--occ)"></span>occupied slot</span><span><span class="sw" style="background:var(--unk)"></span>unknown</span><span>box + dot = detected bike and its wheel contact point, <code>-> S03</code> = assigned slot</span></div>

<h2>Live camera</h2>
<p class="muted" id="live-sub">Pulled from the real camera every 5 s.</p>
<div id="live"></div>

<h2>New approach &mdash; slots drawn once, a small CNN decides free / occupied per slot</h2>
<p class="muted">The camera is fixed, so the slots never move: the 19 hooks of a bay (between two pillars) are drawn once as polygons (4 corners + "19", split in perspective; for the frontal phone pictures the row is found automatically from the rail and the hook pitch). Every snapshot is cut into one crop per slot, a MobileNetV3 classifier (ImageNet-pretrained, head trained on the labelled crops) says free or occupied, and the result is smoothed over the last frames. No bike detection, no slot counting. The number in each slot is the probability "occupied". The first picture is the camera's own view by day; the others are the phone pictures taken on site on the evening of 2026-10-31 (bays 2, 3 and 4). "correct" compares with the hand labels; the model was trained on all of these frames, so leave-one-frame-out (slot-classifier.md) is the honest number.</p>
<div class="grid" id="classifier"></div>

<h2>Example 1 &mdash; reference scene with hand-labelled slots</h2>
<p class="muted">Slots drawn by hand once; every frame is decided automatically and compared with the hand labels.</p>
<div class="grid" id="golden"></div>

<h2>Example 2 &mdash; fully automatic: the reference scene calibrated from its 3 images</h2>
<p class="muted">No admin, no GPT: the camera saw the three images above (different bikes each time), the calibration fitted a perspective lattice to the stirrups, the painted lines and the bikes, and proposed the slots below. The occupancy of all three images is then compared with the hand labels.</p>
<div class="grid" id="auto3"></div>

<h2>Example 3 &mdash; fully automatic from one single image</h2>
<p class="muted">No admin, no GPT: the slots below were proposed by the calibration from this one picture, then occupancy was computed on it. Rows with stands or painted bays get their empty slots right; very dense double racks remain estimates.</p>
<div class="grid" id="auto"></div>

<h2>Example 4 &mdash; real photos from the team's own rack (WhatsApp)</h2>
<p class="muted">No admin, no GPT, no hand labels: each photo gets its own automatically proposed layout and occupancy, same as Example 3.</p>
<div class="grid" id="selfmade"></div>

<h2>Example 5 &mdash; picks for the live demo</h2>
<p class="muted">A curated handful of the photos above, same fully automatic pipeline.</p>
<div class="grid" id="demo_picks"></div>

<h2>Status</h2>
<div class="card"><ul>
<li><b>Works:</b> live camera ingest (push or pull), per-slot occupancy (27/27 on the reference scene), JSON API for the app, privacy blur, auto-proposed layouts with admin confirmation.</li>
<li><b>Needs help:</b> very dense double racks (per-slot assignment is a guess there), irregular hand-drawn racks, and the far rows of diagonal scenes can still be one slot off.</li>
<li><b>Next:</b> point the real camera at a rack, calibrate it, confirm in the admin UI.</li>
</ul><p class="muted">API docs: <a href="/docs">/docs</a> &middot; debug view: <a href="/">/</a></p></div>
</main>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pills=d=>`<span class="pill free">${d.free} free</span><span class="pill occ">${d.occupied} occupied</span>${d.unknown?`<span class="pill unk">${d.unknown} unknown</span>`:''}`;
async function live(){
  let cams=[],health={};
  try{cams=await (await fetch('/cameras')).json();health=await (await fetch('/health')).json();}catch(e){return;}
  const pulls=Object.fromEntries((health.pull||[]).map(p=>[p.camera_id,p]));
  const ids=[...new Set([...Object.keys(pulls),...cams.filter(c=>c.online).map(c=>c.camera_id)])];
  if(!ids.length){document.getElementById('live').innerHTML='<div class="card muted">No camera connected.</div>';return;}
  const parts=await Promise.all(ids.map(async id=>{
    let s=null;try{const r=await fetch(`/cameras/${id}/status`);if(r.ok)s=await r.json();}catch(e){}
    const p=pulls[id];
    const src=p?`<p class="muted">source <code>${esc(p.url)}</code><br>${p.frames} frames, ${p.errors} errors, fetch ${p.last_fetch_ms??'-'} ms, last ${p.last_ok_age_s??'-'} s ago</p>`:'';
    const err=p&&p.last_error?`<p style="color:var(--occ)">${esc(p.last_error)}</p>`:'';
    const body=s?`<div class="big">${s.slot_count?s.free+' / '+s.slot_count:'&ndash;'}</div><p class="muted">${s.slot_count?'slots free':'no slot layout yet'}</p>
      <p>${s.slot_count?pills(s):''}</p><p>${s.detections.bicycle} bikes, ${s.detections.person} persons detected, ${Math.round(s.processing_ms)} ms</p>
      ${s.message?`<p class="muted">&#9888; ${esc(s.message)}</p>`:''}<p class="muted">layout: ${s.layout_status}</p>`:'<p class="muted">waiting for first frame&hellip;</p>';
    return `<div class="live"><div class="card"><img src="/cameras/${id}/preview.jpg?t=${Date.now()}" alt="live ${esc(id)}" onerror="this.style.display='none'"></div>
      <div class="card"><b>${esc(id)}</b>${body}${src}${err}</div></div>`;
  }));
  document.getElementById('live').innerHTML=parts.join('');
}
function exampleCard(name,kind){
  const el=document.createElement('div');el.className='card ex';
  el.innerHTML=`<img alt="${esc(name)}" loading="lazy"><div class="meta"><div class="title">${esc(name)}</div><div class="muted">computing&hellip;</div></div>`;
  document.getElementById(kind).appendChild(el);
  return el;
}
async function loadExample(e){
  const el=exampleCard(e.name,e.kind);
  try{
    const d=await (await fetch(`/demo/examples/${e.name}.json`)).json();
    el.querySelector('img').src=`/demo/examples/${e.name}.jpg`;
    const wrong=d.wrong&&d.wrong.length?` (hooks ${d.wrong.join(', ')} wrong)`:'';
    const score=d.truth_slots?`<span class="ok">${d.correct}/${d.truth_slots} correct</span> vs hand labels${wrong}${d.kind==='auto3'?` over ${d.frames} images &middot; ${d.slots} slots proposed automatically`:''} &middot; `:`${d.slots} slots proposed &middot; `;
    const tail=d.kind==='classifier'?`${d.slots} slots drawn once &middot; per-slot classifier &middot; ${d.ms} ms`:`${d.bikes} bikes detected &middot; ${d.ms} ms`;
    el.querySelector('.meta .muted').outerHTML=`<p>${pills(d)}</p><p class="muted">${score}${tail}</p>`;
  }catch(err){el.querySelector('.meta .muted').textContent='failed: '+err;}
}
(async()=>{
  live();setInterval(live,3000);
  const list=await (await fetch('/demo/examples')).json();
  for(const e of list) await loadExample(e);   // one at a time: the GPU is shared with the live camera
})();
</script></body></html>"""
