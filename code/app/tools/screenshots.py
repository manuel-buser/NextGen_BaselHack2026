"""Render both pages in Chromium with every rack state and save screenshots for a visual check.

    uv run python tools/screenshots.py OUTDIR

Starts a server on a free port with five example racks (live, full, starting, stale, no data; three of them
marked demo), feeds it through the camera endpoints (rack C is polled from a stand-in node instead), waits for the
rack tiles and the open tile with its predictions, and saves phone and desktop screenshots in light
and dark mode, plus two of the map zoomed into rack A with its plan (spaces at zoom 19, the outline one level out)
and two of the place search, typed and answered (the answer comes from this tool, not the real service).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

RACKS = """racks:
  - {id: A, name: Main entrance shelter, capacity: 15, location: {lat: 47.5463, lng: 7.5950}, layout: {rows: 1, cols: 15, bearing: 35}}
  - {id: B, name: "Building 2, north side", capacity: 12, location: {lat: 47.5472, lng: 7.5969}, demo: true, layout: {rows: 2, cols: 6, bearing: 120}}
  - {id: C, name: Visitor parking, capacity: 8, location: {lat: 47.5458, lng: 7.5975}, layout: {rows: 1, cols: 8, bearing: 80}, camera: {url: "http://127.0.0.1:NODE_PORT", poll_s: 1, picture_every_s: 3600}}
  - {id: D, name: Workshop yard, capacity: 20, location: {lat: 47.5477, lng: 7.5942}, demo: true, layout: {rows: 2, cols: 10, bearing: 10}}
  - {id: E, name: Canteen side, capacity: 10, demo: true, layout: {bearing: 150, spaces: [{id: 1, x: 0, y: 0}, {id: 2, x: 1, y: 0}, {id: 3, x: 2, y: 0}, {id: 4, x: 3, y: 0}, {id: 5, x: 4, y: 0}, {id: 6, x: 6, y: 0}, {id: 7, x: 7, y: 0}, {id: 8, x: 8, y: 0}, {id: 9, x: 9, y: 0}, {id: 10, x: 10, y: 0}]}}
places:
  - {name: Main entrance, location: {lat: 47.5461, lng: 7.5956}}
  - {name: Building 2, location: {lat: 47.5475, lng: 7.5965}}
"""
STALE_S = 4.0
VIEWPORTS = {"phone": (390, 844), "desktop": (1280, 800), "laptop": (1440, 900)}
# What an offline node last reported: weak signal, small free block, and a reboot (the uptime dropped)
WEAK_NODE = {"wifi_rssi_dbm": -79, "wifi_connected": True, "heap_largest_block": 81920, "uptime_s": 4100}


def free_ports(count: int) -> list[int]:
    socks = [socket.socket() for _ in range(count)]  # all open at once, so the ports differ
    for sock in socks:
        sock.bind(("127.0.0.1", 0))
    ports = [sock.getsockname()[1] for sock in socks]
    for sock in socks:
        sock.close()
    return ports


def wait_for(url: str) -> None:
    for _ in range(50):
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.2)


def picture(base: str, rack: str, capacity: int, taken: int = 0, times: int = 1, pattern: str | None = None) -> None:
    # fake detector: a DEMO: pattern spells out the spaces, otherwise len % (capacity + 1) slots are taken
    body = f"DEMO:{pattern}".encode() if pattern else b"x" * (10 * (capacity + 1) + taken)
    for _ in range(times):
        httpx.post(f"{base}/api/camera/{rack}/picture", content=body, headers={"Content-Type": "image/jpeg", "X-Node-Id": f"esp32-{rack.lower()}"})


def alive(base: str, rack: str, temp_c: float, too_hot: bool = False, node: dict | None = None) -> None:
    httpx.post(f"{base}/api/camera/{rack}/alive", json={"tempC": temp_c, "tooHot": too_hot, **(node or {})}, headers={"X-Node-Id": f"esp32-{rack.lower()}"})


def seed(base: str) -> None:
    picture(base, "D", 20, 6, times=3)  # will go stale
    alive(base, "D", 44.0, node=WEAK_NODE)
    alive(base, "D", 44.0, node={**WEAK_NODE, "uptime_s": 12})  # the uptime dropped: shown as rebooted
    picture(base, "C", 8, 2, times=1)  # starting: this one plus the one the poller fetched at startup
    time.sleep(STALE_S + 0.5)


def refresh_live(base: str) -> None:
    picture(base, "A", 15, pattern="111011001110100", times=3)  # gaps, so the expanded spaces show where
    alive(base, "A", 41.2)
    picture(base, "B", 12, 12, times=3)
    alive(base, "B", 77.8, too_hot=True)


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    port, node_port = free_ports(2)
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as tmp:
        racks_file = Path(tmp) / "racks.yaml"
        racks_file.write_text(RACKS.replace("NODE_PORT", str(node_port)), encoding="utf-8")
        env = {**os.environ, "DETECTOR_URL": "fake", "BIKESLOT_RACKS": str(racks_file), "STALE_AFTER_S": str(STALE_S), "HISTORY_FILE": ""}
        quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        node = subprocess.Popen([sys.executable, "-m", "uvicorn", "tools.fake_node:app", "--port", str(node_port)], **quiet)
        wait_for(f"http://127.0.0.1:{node_port}/status")  # up before the app, so its first poll succeeds
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "bikeslot.main:app", "--port", str(port)], env=env, **quiet)
        try:
            wait_for(f"{base}/api/health")
            seed(base)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                for scheme in ("light", "dark"):
                    for name, (width, height) in VIEWPORTS.items():
                        context = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme, device_scale_factor=2 if name == "phone" else 1)
                        page = context.new_page()
                        for path, label in (("/", "user"), ("/admin#statistics", "admin")):
                            refresh_live(base)
                            page.goto(base + path)
                            page.wait_for_selector(".tile[data-rack], .stat")
                            if label == "user":
                                page.wait_for_selector(".leaflet-marker-icon .pin, .map-note", timeout=20000)
                                page.wait_for_selector(".tile.is-open .plan")  # rack A has the most space, so its tile opens first
                                page.wait_for_selector("#forecast:not([data-state='loading'])", timeout=10000)  # chart, learning note or quiet line
                                page.wait_for_timeout(2500)  # map tiles
                            page.wait_for_timeout(600)
                            page.evaluate("document.fonts.ready")
                            if label == "admin":  # the packet paused on the second arrow, mid-run
                                pause_at = "(x) => { const svg = document.querySelector('#flow svg'); svg.pauseAnimations(); svg.setCurrentTime(document.getElementById('packet-anim').getStartTime() + (x / 1198) * 3); }"
                                picture(base, "A", 15, 7)
                                page.wait_for_selector("#packet.moving", timeout=8000)
                                page.evaluate(pause_at, 467)
                                page.wait_for_timeout(100)
                            target = outdir / f"{label}-{name}-{scheme}.png"
                            page.screenshot(path=str(target), full_page=True)
                            print(target)
                            if label == "user" and page.locator(".leaflet-marker-icon").count():  # zoomed into rack A: the real-size plan with its card above
                                for zoom, shape, suffix in ((19, "path.plan-space-free", ""), (18, "path.plan-outline-ok", "-out")):
                                    page.evaluate(f"flyToRack(lastRacks.find((r) => r.id === 'A'), {zoom})")
                                    page.wait_for_function(f"map.getZoom() === {zoom}", timeout=10000)
                                    page.wait_for_timeout(2500)  # map tiles
                                    refresh_live(base)
                                    page.evaluate("refresh()")  # rack A went stale meanwhile (short STALE_S): show it live again
                                    page.wait_for_selector(f".leaflet-overlay-pane {shape}")
                                    page.wait_for_timeout(400)  # the card's entrance animation
                                    target = outdir / f"user-map{suffix}-{name}-{scheme}.png"
                                    page.screenshot(path=str(target))
                                    print(target)
                                page.route("**/api/geocode*", lambda route: route.fulfill(json=[{"name": "Basel SBB", "lat": 47.5476, "lng": 7.5897}]))
                                if name == "phone":
                                    page.click("#dest-open")  # folded on a phone
                                page.fill("#dest-q", "Basel SBB")
                                for suffix, action in (("-typed", None), ("", "Enter")):
                                    if action:
                                        page.press("#dest-q", action)
                                        page.wait_for_selector("#dest-result:has-text('walk')")
                                        page.wait_for_timeout(2500)  # the fit and map tiles
                                    target = outdir / f"user-search{suffix}-{name}-{scheme}.png"
                                    page.screenshot(path=str(target))
                                    print(target)
                                page.unroute("**/api/geocode*")
                            if label == "admin":  # the tall pane fitted on the camera step with the dot paused inside it, then every next step
                                page.evaluate("document.querySelector('#flow svg').unpauseAnimations()")
                                page.dblclick("#flow .node[data-node='camera'] .node-box")
                                page.wait_for_selector("#flow[data-level='detail'] #flow-steps [data-step='camera'][aria-current='step']")
                                page.wait_for_selector("#packet:not(.moving)", state="attached", timeout=8000)
                                picture(base, "A", 15, 7)
                                page.wait_for_selector("#packet.moving", timeout=8000)
                                page.evaluate(pause_at, 60)
                                page.wait_for_timeout(100)
                                target = outdir / f"admin-zoom-{name}-{scheme}.png"
                                page.locator(".flow-wrap").screenshot(path=str(target))
                                print(target)
                                page.evaluate("document.querySelector('#flow svg').unpauseAnimations()")
                                for step in ("relay", "vision", "decide", "pages"):
                                    page.click("#flow-next")
                                    page.wait_for_selector(f"#flow-steps [data-step='{step}'][aria-current='step']")
                                    page.wait_for_timeout(900)  # the pan settles
                                    target = outdir / f"admin-zoom-{step}-{name}-{scheme}.png"
                                    page.locator(".flow-wrap").screenshot(path=str(target))
                                    print(target)
                        context.close()
                browser.close()
        finally:
            for process in (server, node):
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots"))
