"""Browser tests with Playwright. Skipped when Chromium cannot start on this machine."""

import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

pytestmark = pytest.mark.ui

RACKS = """racks:
  - {id: A, name: Main entrance shelter, capacity: 15, location: {lat: 47.5463, lng: 7.5950}, layout: {rows: 1, cols: 15, bearing: 35}}
  - {id: B, name: "Building 2, north side", capacity: 12, location: {lat: 47.5472, lng: 7.5969}}
  - {id: C, name: Visitor parking, capacity: 12, demo: true}
places:
  - {name: Main entrance, location: {lat: 47.5461, lng: 7.5956}}
  - {name: Building 2, location: {lat: 47.5475, lng: 7.5965}}
"""
CAPACITY_A = 15


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def picture(base: str, rack: str, body: bytes, times: int = 1) -> None:
    for _ in range(times):
        httpx.post(f"{base}/api/camera/{rack}/picture", content=body, headers={"Content-Type": "image/jpeg"})


def taken_a(taken: int) -> bytes:
    return b"x" * (10 * (CAPACITY_A + 1) + taken)  # fake detector: len % (capacity + 1) slots taken


def fetch_or_abort(route):
    """route.fetch(), or None after aborting: a reset under load must not fail the test at teardown."""
    from playwright.sync_api import Error

    try:
        return route.fetch()
    except Error:
        try:
            route.abort()
        except Error:
            pass
        return None


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    racks_file = tmp_path_factory.mktemp("racks") / "racks.yaml"
    racks_file.write_text(RACKS, encoding="utf-8")
    port = free_port()
    env = {**os.environ, "DETECTOR_URL": "fake", "BIKESLOT_RACKS": str(racks_file), "HISTORY_FILE": ""}  # history in memory only
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "bikeslot.main:app", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            if httpx.get(f"{base}/api/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        process.kill()
        pytest.fail("server did not start")
    yield base
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture(scope="module")
def page(server):
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Error as exc:
            pytest.skip(f"Chromium cannot start here: {str(exc).splitlines()[0]}")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        yield page
        browser.close()


def test_user_page_shows_free_count_and_no_horizontal_scroll(server, page):
    picture(server, "A", taken_a(5), times=3)  # 5 taken of 15
    page.goto(server + "/")
    page.wait_for_selector(".tile[data-rack='A'].tile--live")
    tile = page.locator(".tile[data-rack='A']")
    assert (tile.locator(".count__n").inner_text(), tile.locator(".count__of").inner_text()) == ("10", "free of 15")
    assert "Live" in tile.locator(".tile__head > .chip").inner_text()
    assert tile.locator(".tile__name").inner_text() == "Main entrance shelter"
    assert tile.locator(".tile__title .chip--real").inner_text() == "Real camera"  # rack C is demo, so the real rack is marked
    assert page.locator(".tile[data-rack='C'] .chip--real").count() == 0
    assert page.locator(".tile[data-rack='B'] .tile__head > .chip").inner_text() == "No data"
    assert page.locator(".tile[data-rack='B'] .count__n").inner_text() == "?"
    assert page.locator(".rack-row, #panel").count() == 0  # the old switcher and panel are gone
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def test_map_shows_a_pin_per_rack_and_opens_the_spaces(server, page):
    page.goto(server + "/")
    page.wait_for_selector(".leaflet-marker-icon .pin, .map-note", timeout=20000)
    if page.locator(".map-note").count() and not page.locator(".leaflet-marker-icon").count():
        pytest.skip("map library did not load (no internet)")
    assert page.locator(".leaflet-marker-icon .pin").count() == 2  # rack C has no location
    assert page.locator(".leaflet-marker-icon .pin:not(.pin--none) .pin__n").first.inner_text() == "10"
    page.locator(".leaflet-marker-icon .pin:not(.pin--none)").first.click()
    page.wait_for_selector(".leaflet-marker-icon .mcard .plan", timeout=10000)  # rack A has a layout: the card shows the plan
    assert page.locator(".leaflet-marker-icon .mcard .cell--free").count() == 10


def test_open_tile_shows_the_selected_rack_and_follows_the_refresh(server, page):
    page.goto(server + "/")
    page.wait_for_selector(".tile.is-open .plan .cell")  # rack A is live with the most space: its tile opens first
    tile = page.locator(".tile[data-rack='A']")
    assert page.locator(".tile.is-open").count() == 1 and tile.locator(".tile__head").get_attribute("aria-expanded") == "true"
    assert tile.locator(".tile__title .chip--real").inner_text() == "Real camera"
    assert tile.locator(".count__n").inner_text() == "10"
    body = tile.locator(".tile__live")
    assert (body.locator(".plan .cell--taken").count(), body.locator(".plan .cell--free").count()) == (5, 10)
    assert body.locator(".plan-free").inner_text() == "Free spaces: 6 to 15"
    assert tile.locator(".tile__meta").inner_text().startswith("Updated")
    assert body.locator(".plan-key").is_visible()
    page.focus(".tile[data-rack='A'] .tile__head")
    picture(server, "A", taken_a(6))  # one more bike: the tile follows the next refresh and marks the change
    page.wait_for_function("document.querySelectorAll('.tile.is-open .tile__live .plan .cell--taken').length === 6", timeout=8000)
    assert body.locator(".cell.is-changed").count() == 1
    assert body.locator(".plan-free").inner_text() == "Free spaces: 7 to 15"
    assert page.evaluate("document.activeElement.closest('.tile').dataset.rack") == "A"  # the refresh kept the keyboard focus


def test_demo_rack_looks_like_any_rack_and_shows_made_up_spaces(server, page):
    picture(server, "C", b"DEMO:101100000000", times=3)  # made-up picture: spaces 1, 3 and 4 taken
    page.goto(server + "/")
    page.wait_for_selector(".tile[data-rack='C'].tile--live")
    tile = page.locator(".tile[data-rack='C']")
    assert tile.locator(".tile__title .chip").count() == 0
    assert tile.locator(".tile__head > .chip").inner_text() == "Live"
    assert page.locator(".tile[data-rack='A'] .tile__title .chip--real").inner_text() == "Real camera"
    assert "demo" not in page.locator("body").inner_text().lower()  # no chip, map key entry or footer note
    assert page.locator(".chip--demo, .pin--demo, #note").count() == 0
    tile.locator(".tile__head").click()
    page.wait_for_selector(".tile[data-rack='C'].is-open .plan-wrap--free .cell")  # no layout: the spaces simply wrap
    assert page.locator(".tile.is-open").count() == 1
    assert tile.locator(".plan-free").inner_text() == "Free spaces: 2, 5 to 12"


def test_admin_page_lists_components(server, page):
    page.goto(server + "/admin")
    page.wait_for_selector("#detector:has-text('fake')")
    assert "Online" in page.locator("#cameras").inner_text()
    assert "Real camera" in page.locator("#cameras").inner_text()  # rack C is demo
    assert "Reachable" in page.locator("#detector").inner_text()
    assert page.locator("#kpis .stat").count() == 4
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def test_admin_flow_diagram_is_live_and_clickable(server, page):
    picture(server, "A", taken_a(5))
    page.goto(server + "/admin")
    page.wait_for_selector("#flow .node[data-node='camera'][data-state]")
    camera = page.locator("#flow .node[data-node='camera']")
    assert camera.get_attribute("data-state") == "warning"  # racks A and C online, rack B never seen
    assert camera.locator(".node-status").text_content() == "2 of 3 online"
    assert page.locator("#flow .node[data-node='detector'] .node-status").text_content() == "stand-in"
    page.locator("#flow .node[data-node='decide']").click()
    page.wait_for_selector("#flow-detail:has-text('applies the rules')")
    picture(server, "A", taken_a(5))
    page.wait_for_selector("#packet.moving", timeout=8000)  # the next refresh sees a new picture and moves the dot


def test_admin_flow_zooms_into_the_camera_step(server, page):
    page.goto(server + "/admin")
    page.wait_for_selector("#flow .node[data-node='camera'] .part[data-part='capture'][data-state]", state="attached")  # hidden until zoomed
    assert page.locator("#flow").get_attribute("data-level") == "overview"
    camera = page.locator("#flow .node[data-node='camera']")
    camera.scroll_into_view_if_needed()
    box = camera.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, -300)  # two notches up: the view eases to about 3.7x around the camera step
    page.mouse.wheel(0, -300)
    page.wait_for_selector("#flow[data-level='detail']")
    assert camera.locator(".part[data-part='capture'] .part-value").text_content().endswith("s ago")
    assert camera.locator(".part[data-part='alive'] .part-value").text_content() == "2 of 3 online"
    assert camera.locator(".part-label").first.is_visible() and not camera.locator(".node-title").is_visible()
    picture(server, "A", taken_a(5))
    page.wait_for_selector("#packet.moving", timeout=8000)  # the dot still travels while zoomed in
    page.click("#flow-zoom-reset")
    page.wait_for_selector("#flow[data-level='overview']")
    assert camera.locator(".node-title").is_visible() and not camera.locator(".part-label").first.is_visible()


def test_both_pages_show_the_team_badge(server, page):
    for path, (badge_at, name, credit) in (("/", ("header.hdr .badge", ".badge__name", ".badge__sub")), ("/admin", ("header.masthead .team", ".team-name", ".team-credit"))):
        page.goto(server + path)
        badge = page.locator(badge_at)
        assert badge.is_visible()
        assert badge.locator(name).inner_text() == "NEXTGEN"
        assert badge.locator(credit).inner_text() == "Endress+Hauser challenge at BaselHack 2026"
        assert badge.locator("a").count() == 0
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


def test_rack_plan_in_the_panel_and_at_real_size_on_the_map(server, page):
    picture(server, "A", taken_a(5), times=3)  # 5 taken of 15
    page.goto(server + "/")
    page.wait_for_selector(".tile.is-open .plan .cell--taken")  # rack A is selected first
    body = page.locator(".tile[data-rack='A'] .tile__live")
    plan = body.locator(".plan")
    assert plan.locator(".cell").count() == 15
    assert (plan.locator(".cell--taken").count(), plan.locator(".cell--free").count()) == (5, 10)
    assert plan.locator(".cell > span").first.text_content() == "1" and plan.locator(".cell > span").count() == 15
    assert body.locator(".plan__entry").count() == 1 and "Entrance" in body.locator(".plan-key").inner_text()
    page.wait_for_selector(".leaflet-marker-icon .pin, .map-note", timeout=20000)
    if page.locator(".map-note").count() and not page.locator(".leaflet-marker-icon").count():
        pytest.skip("map library did not load (no internet)")
    spaces = page.locator(".leaflet-overlay-pane path.plan-space")
    page.evaluate("map.setView([47.5463, 7.5950], 16, {animate: false})")
    page.wait_for_function("!document.querySelector('.leaflet-overlay-pane .plan-space, .leaflet-overlay-pane .plan-outline')")
    page.evaluate("map.setView([47.5463, 7.5950], 18, {animate: false})")
    page.wait_for_selector(".leaflet-overlay-pane path.plan-outline-ok")  # a space would be under 3 px: the outline only
    assert spaces.count() == 0
    page.evaluate("map.setView([47.5463, 7.5950], 19, {animate: false})")
    page.wait_for_selector(".leaflet-overlay-pane path.plan-space-taken")
    assert (spaces.count(), page.locator("path.plan-space-free").count(), page.locator("path.plan-space-taken").count()) == (15, 10, 5)
    assert page.locator("path.plan-entrance").count() == 1 and page.locator("path.plan-outline").count() == 0
    assert page.locator(".leaflet-overlay-pane #plan-hatch").count() == 1  # taken reads without colour too
    assert page.locator(".leaflet-marker-icon .mcard .plan .cell--free").count() == 10  # the upright card stays
    card_bottom = page.evaluate("document.querySelector('.leaflet-marker-icon .mcard').getBoundingClientRect().bottom")
    plan_top = page.evaluate("Math.min(...[...document.querySelectorAll('path.plan-space, path.plan-entrance')].map((p) => p.getBoundingClientRect().top))")
    assert card_bottom + 8 <= plan_top  # the card and its pointer sit above the plan


def test_opening_a_tile_selects_the_rack_and_closes_the_open_one(server, page):
    page.goto(server + "/")
    page.wait_for_selector(".tile.is-open .plan")
    page.click(".tile[data-rack='B'] .tile__head")
    page.wait_for_selector(".tile[data-rack='B'].is-open")
    assert page.locator(".tile.is-open").count() == 1
    assert page.locator(".tile[data-rack='A'] .tile__head").get_attribute("aria-expanded") == "false"
    assert page.locator(".tile[data-rack='A'] .tile__body").get_attribute("inert") is not None
    assert page.locator(".tile[data-rack='B'] .count__n").inner_text() == "?"  # rack B has no data
    assert page.locator(".tile[data-rack='B'] #forecast").count() == 1  # the predictions moved with the selection
    if page.locator(".leaflet-marker-icon").count():  # the same selection on the map
        assert page.locator(".leaflet-marker-icon[title='Building 2, north side'] .is-selected").count() == 1
    page.click(".tile[data-rack='A'] .tile__head")
    page.wait_for_selector(".tile[data-rack='A'].is-open")
    assert page.locator(".tile[data-rack='B'].is-open").count() == 0
    assert page.locator(".tile[data-rack='A'] .tile__live .plan .cell").count() == 15
    if page.locator(".leaflet-marker-icon").count():  # the map flies to the rack like a pin click
        page.wait_for_function("map.getZoom() >= 19", timeout=8000)
    page.click(".tile[data-rack='A'] .tile__head")  # the open tile closes, and the selection with it
    page.wait_for_selector(".tile.is-open", state="detached")
    page.evaluate("refresh()")
    assert page.locator(".tile.is-open").count() == 0 and page.locator(".leaflet-marker-icon .is-selected").count() == 0


def test_nearest_with_space_uses_the_browser_position(server, page):
    picture(server, "A", taken_a(5), times=3)
    context = page.context.browser.new_context(
        viewport={"width": 390, "height": 844},
        permissions=["geolocation"],
        geolocation={"latitude": 47.5467, "longitude": 7.5950},  # about 45 m north of rack A; rack B has no data
    )
    near = context.new_page()
    near.goto(server + "/")
    near.wait_for_selector(".tile[data-rack='B']")
    near.wait_for_selector(".leaflet-marker-icon .pin, .map-note", timeout=20000)
    if near.locator(".map-note").count() and not near.locator(".leaflet-marker-icon").count():
        context.close()
        pytest.skip("map library did not load (no internet)")
    near.click(".tile[data-rack='B'] .tile__head")
    near.click("#dest-open")  # folded on a phone
    near.click(".leaflet-control #nearest")
    near.wait_for_selector("#dest-result:has-text('by bike')")
    assert near.locator("#dest-result").inner_text() == "Main entrance shelter: 10 free, about 1 min by bike"
    assert near.locator("path.you-dot").count() == 1 and near.locator("path.dest-line").count() == 1
    assert near.locator(".tile.is-open .tile__name").inner_text() == "Main entrance shelter"
    assert near.locator(".tile[data-rack='A'] .tile__extra").inner_text().startswith("40 m")
    assert near.locator(".tile[data-rack='B'] .tile__extra").inner_text().startswith("150 m")
    context.close()


def map_ready(page) -> None:
    page.wait_for_selector(".leaflet-marker-icon .pin, .map-note", timeout=20000)
    if page.locator(".map-note").count() and not page.locator(".leaflet-marker-icon").count():
        pytest.skip("map library did not load (no internet)")


def test_map_plan_keeps_its_real_size_when_zooming(server, page):
    picture(server, "A", taken_a(5), times=3)
    page.goto(server + "/")
    map_ready(page)
    union = """(selector) => { const boxes = [...document.querySelectorAll(selector)].map((p) => p.getBBox());
        const w = Math.max(...boxes.map((b) => b.x + b.width)) - Math.min(...boxes.map((b) => b.x));
        const h = Math.max(...boxes.map((b) => b.y + b.height)) - Math.min(...boxes.map((b) => b.y)); return [w, h]; }"""
    sizes = {}
    for zoom, selector in ((21, "path.plan-space"), (20, "path.plan-space"), (18, "path.plan-outline"), (17, "path.plan-outline")):
        page.evaluate(f"map.setView([47.5463, 7.5950], {zoom}, {{animate: false}})")
        page.wait_for_selector(f".leaflet-overlay-pane {selector}")
        sizes[zoom] = page.evaluate(union, selector)
    for near, far in ((21, 20), (18, 17)):  # one level out halves the plan on screen, so it stays on the rack's spot
        assert all(abs(f / n - 0.5) < 0.05 for n, f in zip(sizes[near], sizes[far])), (near, sizes)


def test_clicking_a_pin_selects_its_rack(server, page):
    page.goto(server + "/")
    map_ready(page)
    page.wait_for_selector(".tile[data-rack='A'].is-open")
    page.click(".leaflet-marker-icon[title='Building 2, north side'] .pin")
    page.wait_for_selector(".tile[data-rack='B'].is-open")
    assert page.locator(".tile[data-rack='A'].is-open").count() == 0
    assert page.locator(".tile.is-open .tile__name").inner_text() == "Building 2, north side"
    page.wait_for_function("map.getZoom() >= 19", timeout=8000)  # and the map still flies there


def test_destination_from_the_list_picks_the_nearest_free_rack(server, page):
    picture(server, "A", taken_a(5), times=3)
    page.goto(server + "/")
    map_ready(page)
    page.wait_for_selector(".tile[data-rack='A'].tile--live")
    page.click(".tile[data-rack='B'] .tile__head")
    asked = []
    page.route("**/api/geocode*", lambda route: (asked.append(route.request.url), route.abort()))
    page.click("#dest-open")
    page.fill("#dest-q", "building 2 ")  # a configured place, typed: rack B is nearest but has no data
    page.press("#dest-q", "Enter")
    page.wait_for_selector("#dest-result:has-text('walk')")
    assert page.locator("#dest-result").inner_text() == (
        "Building 2, north side has no live data, next: Main entrance shelter: 10 free, about 3 min walk")
    assert page.locator(".tile[data-rack='A'].is-open").count() == 1
    assert page.locator("path.dest-line").count() == 1 and page.locator(".dest-mark").count() == 1
    page.wait_for_function("map.getBounds().contains([47.5475, 7.5965]) && map.getBounds().contains([47.5463, 7.5950])", timeout=8000)
    page.unroute("**/api/geocode*")
    assert asked == []  # a configured place never goes to the place search


def test_tapping_the_map_sets_a_destination(server, page):
    picture(server, "A", taken_a(5), times=3)
    page.goto(server + "/")
    map_ready(page)
    page.wait_for_selector(".tile[data-rack='A'].tile--live")
    x, y = page.evaluate("""() => { const p = map.latLngToContainerPoint([47.5463, 7.5950]);
        const r = document.getElementById('map').getBoundingClientRect(); return [r.left + p.x + 70, r.top + p.y + 12]; }""")
    page.mouse.click(x, y)
    page.wait_for_selector("#dest-result:has-text('walk')")
    assert page.locator("#dest-result").inner_text().startswith("Main entrance shelter: 10 free, about ")
    assert page.locator(".dest-mark").count() == 1 and page.locator("path.dest-line").count() == 1
    assert page.locator("#dest-q").input_value() == ""


def history(learning: bool = False, days: int = 4) -> str:
    typical = [{"t": f"{h:02d}:{m:02d}", "free": None if h < 5 else max(0.0, 14.2 - (h - 5) * 1.5), "full_share": 0.0, "n": days} for h in range(24) for m in (0, 15, 30, 45)]
    today = [{"t": f"{6 + i // 4:02d}:{(i % 4) * 15:02d}", "free": 12.0 - i} for i in range(7)]
    insights = {"usually_full_by": "08:15", "best_before": "07:45", "usual_free_now": 3.4, "days": days, "learning": False}
    if learning:
        insights = {"usually_full_by": None, "best_before": None, "usual_free_now": None, "days": days, "learning": True}
    body = {"rack": "A", "now": "07:40", "day_type": "weekday", "free_now": 6, "capacity": 15, "today": today, "typical": typical, "insights": insights}
    return json.dumps(body)


def page_with_history(page, server, handler):
    other = page.context.browser.new_context(viewport={"width": 390, "height": 844}).new_page()
    other.route("**/api/racks/*/history", handler)
    other.goto(server + "/")
    return other


def test_predictions_render_from_the_history(server, page):
    body = history()
    other = page_with_history(page, server, lambda route: route.fulfill(status=200, content_type="application/json", body=body))
    other.wait_for_selector("#forecast[data-state='ready'] svg.chart")
    forecast = other.locator("#forecast")
    text = forecast.inner_text()
    assert "Usually full by 08:15 on weekdays" in text and "Best chance before 07:45" in text
    assert "Now 6 free, usually 3 at this time" in text
    assert forecast.locator(".chart-typical").count() == 1 and forecast.locator(".chart-today").count() == 1
    assert forecast.locator(".chart-now").count() == 1 and forecast.locator(".chart-now-dot").count() == 1
    assert forecast.locator(".chart-tick").count() == 10  # 06:00 to 20:00 every 2 hours, plus capacity and 0
    assert other.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    other.context.close()


def test_predictions_say_when_still_learning(server, page):
    body = history(learning=True, days=1)
    other = page_with_history(page, server, lambda route: route.fulfill(status=200, content_type="application/json", body=body))
    other.wait_for_selector("#forecast[data-state='learning']")
    text = other.locator("#forecast").inner_text()
    assert "Still learning: 1 day so far" in text and "Usually full by" not in text
    other.context.close()


def test_failing_history_shows_a_quiet_line(server, page):
    other = page_with_history(page, server, lambda route: route.fulfill(status=500, body="boom"))
    other.wait_for_selector("#forecast[data-state='error']")
    assert other.locator("#forecast").inner_text() == "Predictions are not available right now."
    assert other.locator(".tile").count() == 3 and other.locator(".tile.is-open .tile__name").is_visible()
    other.context.close()


FLOW_HEIGHT = "document.querySelector('#flow svg').getBoundingClientRect().height"
STEP_IN_VIEW = """(step) => {
  const pane = document.querySelector('#flow svg').getBoundingClientRect();
  const box = document.querySelector(`#flow .node[data-node='${step}'] .node-box`).getBoundingClientRect();
  return box.left >= pane.left && box.right <= pane.right && box.top >= pane.top && box.bottom <= pane.bottom;
}"""


def test_admin_flow_pane_grows_and_moves_between_steps(server, page):
    page.goto(server + "/admin")
    page.wait_for_selector("#flow .node[data-node='camera'][data-state]")
    strip = page.evaluate(FLOW_HEIGHT)
    assert strip < 200 and not page.locator("#flow-next").is_visible()
    page.dblclick("#flow .node[data-node='camera'] .node-box")
    page.wait_for_selector("#flow[data-level='detail'] #flow-steps [data-step='camera'][aria-current='step']")
    page.wait_for_function(f"{FLOW_HEIGHT} > 300")
    page.wait_for_function(STEP_IN_VIEW, arg="camera")  # the whole box, not a slice
    page.click("#flow-next")
    page.wait_for_selector("#flow-steps [data-step='receive'][aria-current='step']")
    page.wait_for_function(STEP_IN_VIEW, arg="receive")
    assert page.locator("#flow .node[data-node='receive'] .part-label").first.is_visible()
    page.click("#flow-steps [data-step='camera']")
    page.wait_for_selector("#flow-steps [data-step='camera'][aria-current='step']")
    page.keyboard.press("ArrowRight")  # the focused chip is inside the diagram
    page.wait_for_selector("#flow-steps [data-step='receive'][aria-current='step']")
    page.wait_for_function(STEP_IN_VIEW, arg="receive")
    page.click("#flow-zoom-reset")
    page.wait_for_function(f"Math.abs({FLOW_HEIGHT} - {strip}) < 2")
    assert page.locator("#flow").get_attribute("data-level") == "overview"


def test_admin_flow_dot_finishes_its_run_before_the_next(server, page):
    page.goto(server + "/admin")
    page.wait_for_selector("#flow .node[data-node='camera'][data-state]")
    picture(server, "A", taken_a(5))
    page.wait_for_selector("#packet.moving", timeout=8000)
    runs = int(page.locator("#packet").get_attribute("data-runs"))
    start = page.evaluate("document.getElementById('packet-anim').getStartTime()")
    picture(server, "A", taken_a(5))
    page.evaluate("refresh()")  # sees the new picture while the dot is still travelling
    assert page.evaluate("document.getElementById('packet-anim').getStartTime()") == start
    assert int(page.locator("#packet").get_attribute("data-runs")) == runs
    page.wait_for_function(f"document.getElementById('packet').dataset.runs === '{runs + 1}'", timeout=5000)  # queued, runs next
    assert page.evaluate("document.getElementById('packet-anim').getStartTime()") >= start + 2.9


def test_admin_flow_dot_stays_under_the_boxes_in_the_overview(server, page):
    page.goto(server + "/admin")
    page.wait_for_selector("#flow .node[data-node='camera'][data-state]")
    page.locator("#flow .node[data-node='camera']").scroll_into_view_if_needed()
    picture(server, "A", taken_a(5))
    page.wait_for_selector("#packet.moving", timeout=8000)
    top_at = """async (x) => {
      const svg = document.querySelector('#flow svg');
      svg.pauseAnimations();
      svg.setCurrentTime(document.getElementById('packet-anim').getStartTime() + (x / 1198) * 3);
      await new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
      const point = new DOMPoint(x, 80).matrixTransform(svg.getScreenCTM());
      const top = document.elementFromPoint(point.x, point.y);
      return top.id === 'packet' ? 'packet' : top.closest('.node')?.dataset.node;
    }"""
    assert page.evaluate(top_at, 206) == "packet"  # on the first arrow the dot shows
    assert page.evaluate(top_at, 75) == "camera"  # inside the camera box its fill and text cover it
    page.evaluate("document.querySelector('#flow svg').unpauseAnimations()")


def test_admin_camera_cards_fit_a_laptop_screen(server, page):
    node = {"tempC": 41.2, "tooHot": False, "wifi_rssi_dbm": -79, "wifi_connected": True, "heap_largest_block": 81920}
    for uptime_s in (4100, 12):  # the uptime dropped: shown as restarted
        httpx.post(f"{server}/api/camera/A/alive", json={**node, "uptime_s": uptime_s}, headers={"X-Node-Id": "esp32-a"})
    page.set_viewport_size({"width": 1280, "height": 800})
    try:
        page.goto(server + "/admin")
        page.wait_for_selector("#cameras:has-text('restarted')")
        card = page.locator("#cameras .card[data-key='A']")
        assert card.locator(".chip-real").inner_text() == "Real camera"  # rack C is demo
        assert page.evaluate("(() => { const box = document.querySelector('#cameras'); return box.scrollWidth <= box.clientWidth; })()")
    finally:
        page.set_viewport_size({"width": 390, "height": 844})


def test_admin_step_can_be_selected_while_zoomed_in(server, page):
    desk = page.context.browser.new_page(viewport={"width": 1280, "height": 900})
    try:
        desk.goto(server + "/admin")
        desk.wait_for_selector("#flow .node[data-node='camera'][data-state]")
        receive = desk.locator("#flow .node[data-node='receive'] .node-box")
        box = receive.bounding_box()
        desk.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        desk.mouse.wheel(0, -300)
        desk.mouse.wheel(0, -300)
        desk.wait_for_selector("#flow[data-level='detail']")
        desk.wait_for_timeout(1500)  # the zoom eases in and snaps onto the step
        box = receive.bounding_box()
        desk.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] * 0.85)
        desk.wait_for_selector("#flow-detail:has-text('App receives the picture')", timeout=3000)
    finally:
        close(desk)


def test_admin_camera_card_shows_signal_bars_and_a_temperature_gauge(server, page):
    node = {"tempC": 41.2, "tooHot": False, "wifi_rssi_dbm": -79, "wifi_connected": True, "heap_largest_block": 81920, "uptime_s": 5000}
    httpx.post(f"{server}/api/camera/A/alive", json=node, headers={"X-Node-Id": "esp32-a"})
    page.goto(server + "/admin")
    card = page.locator("#cameras .card[data-key='A']")
    card.locator(".bars[data-level='weak']").wait_for()
    assert card.locator(".bars .bar.on").count() == 1  # -79 dBm: one bar of four
    assert "-79 dBm" in card.inner_text()
    assert card.locator(".gauge .gauge-limit").count() == 1
    assert card.locator(".gauge").get_attribute("aria-label") == "Chip temperature 41.2 C, limit 75 C"
    assert "Low" in card.locator("[data-f='mem']").inner_text()  # 80 KB is under the 100 KB mark


def test_admin_camera_card_carries_the_rack_detection(server, page):
    picture(server, "A", taken_a(4), times=3)
    page.goto(server + "/admin")
    card = page.locator("#cameras .card[data-key='A']")
    card.locator("[data-f='detect']:has-text('spaces ready')").wait_for()
    assert card.locator(".chip-good:has-text('Live')").count() == 1  # the rack's status chip
    assert "last result" in card.locator("[data-f='detect']").inner_text()


def test_admin_activity_strip_ticks_for_a_new_picture(server, page):
    page.goto(server + "/admin")
    page.wait_for_selector("#activity .act-row[data-row='A']")
    ticks = "#activity .act-row[data-row='A'] .mark-picture"
    before = page.locator(ticks).count()
    picture(server, "A", taken_a(2))
    page.wait_for_function(f"document.querySelectorAll(\"{ticks}\").length > {before}", timeout=8000)
    assert page.locator(ticks).last.get_attribute("title").startswith("Picture from A")


def test_admin_flow_detail_shows_lanes_and_the_stand_in(server, page):
    picture(server, "A", taken_a(5))
    page.goto(server + "/admin")
    page.wait_for_selector("#flow .node[data-node='detector'] .part[data-part='standin'][data-state]", state="attached")
    page.dblclick("#flow .node[data-node='camera'] .node-box")
    page.wait_for_selector("#flow[data-level='detail'] #flow-steps [data-step='camera'][aria-current='step']")
    camera = page.locator("#flow .node[data-node='camera']")
    assert camera.locator(".inside .chain").count() == 2  # the picture lane and the health lane
    assert camera.locator(".lane-tag").all_text_contents() == ["picture", "health"]
    assert camera.locator(".lane-tag").last.is_visible()
    assert camera.locator(".part.minor[data-part='alive']").count() == 1 and camera.locator(".part.minor[data-part='capture']").count() == 0
    detector = page.locator("#flow .node[data-node='detector'] .inside")
    assert "built-in stand-in, no recognition" in detector.text_content()  # detector mode fake: nothing is recognised
    assert detector.locator(".part[data-part='bikes']").count() == 0
    page.click("#flow-zoom-reset")
    page.wait_for_selector("#flow[data-level='overview']")


def test_admin_flow_decide_step_deletes_the_picture_before_the_rules(server, page):
    page.goto(server + "/admin")
    page.wait_for_selector("#flow .node[data-node='decide'] .part[data-state]", state="attached")
    decide = page.locator("#flow .node[data-node='decide']")
    order = decide.locator(".part").evaluate_all("(parts) => parts.map((p) => p.dataset.part)")
    assert order == ["discard", "confidence", "window", "counts", "history"]
    labels = decide.locator(".part-label").all_text_contents()
    assert labels.index("confidence rule") < labels.index("3-picture rule")
    assert decide.locator(".note[data-note='foot']").text_content().startswith("clock rule: a rack shows stale after")
    decide.click()
    page.wait_for_selector("#flow-detail:has-text('applies the rules')")
    page.click("#flow-detail summary")  # "What happens" is collapsed
    text = page.locator("#flow-detail .flow-how").inner_text()
    assert text.index("deleted") < text.index("confidence rule") < text.index("3-picture rule")


def test_admin_flow_step_view_is_fully_detail_on_a_wide_short_window(server, page):
    wide = page.context.browser.new_page(viewport={"width": 1865, "height": 675})
    try:
        wide.goto(server + "/admin")
        wide.wait_for_selector("#flow .node[data-node='camera'][data-state]")
        wide.add_style_tag(content=".page.wide { max-width: none; } .flow-wrap { width: 1899px; }")  # a pane about 1865 px wide, as on a wide laptop window
        assert 1.8 < wide.evaluate("fitScale()") < 2.2  # this pane shape fits a step inside the old fixed cross-fade band
        wide.dblclick("#flow .node[data-node='camera'] .node-box")
        wide.wait_for_selector("#flow[data-level='detail'] #flow-steps [data-step='camera'][aria-current='step']")
        wide.click("#flow-next")
        wide.wait_for_selector("#flow-steps [data-step='receive'][aria-current='step']")
        wide.wait_for_timeout(1200)  # the pan settles
        assert wide.locator("#flow").get_attribute("data-level") == "detail"
        assert not any(text.is_visible() for text in wide.locator("#flow .overview text").all())
        assert wide.locator("#flow .node[data-node='receive'] .part-label").first.is_visible()
    finally:
        wide.close()


def test_admin_wheel_stop_inside_the_fade_never_rests_half_faded(server, page):
    desk = page.context.browser.new_page(viewport={"width": 1280, "height": 900})
    try:
        desk.goto(server + "/admin")
        desk.wait_for_selector("#flow .node[data-node='camera'][data-state]")
        # each case stops the wheel inside the fade band (1.8 to 2.2 here): at about 1.93 and about 2.09
        for pre_zoom, delta, settled in ((False, -300, "overview"), (True, -150, "detail")):
            desk.click("#flow-zoom-reset")
            desk.wait_for_selector("#flow[data-level='overview']")
            if pre_zoom:
                desk.click("#flow-zoom-in")  # to 1.5, below the band
                desk.wait_for_timeout(600)
            box = desk.locator("#flow .node[data-node='receive'] .node-box").bounding_box()
            desk.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            desk.mouse.wheel(0, delta)
            desk.wait_for_selector(f"#flow[data-level='{settled}']", timeout=4000)
            desk.wait_for_timeout(800)
            assert desk.locator("#flow").get_attribute("data-level") == settled
    finally:
        close(desk)


SBB = {"name": "Basel SBB", "lat": 47.5476, "lng": 7.5897}  # about 420 m from rack A and 540 m from rack B


def close(view) -> None:
    """Close a page whose routes may still be fetching: closing would fail that fetch and raise its error here."""
    view.unroute_all(behavior="ignoreErrors")
    view.close()


def search_page(server, page, width: int, asked: list, b_live: bool = False, a_full: dict | None = None):
    """A fresh page whose place search is answered here (never the real service); optionally rack B shown
    live with 3 free and rack A full once a_full["on"] is set."""
    fresh = page.context.browser.new_page(viewport={"width": width, "height": 844})

    def geocode(route):
        asked.append(route.request.url)
        route.fulfill(json=[SBB])

    def racks(route):
        response = fetch_or_abort(route)
        if response is None:
            return
        data = response.json()
        for rack in data["racks"]:
            if rack["id"] == "B" and b_live:
                rack.update(status="ok", free=3, age_s=1, slots=[{"id": i + 1, "state": "free" if i < 3 else "taken"} for i in range(12)])
            if rack["id"] == "A" and a_full and a_full["on"]:
                rack.update(free=0, slots=[{**slot, "state": "taken"} for slot in rack["slots"]])
        route.fulfill(response=response, json=data)

    fresh.route("**/api/geocode*", geocode)
    fresh.route("**/api/racks", racks)
    fresh.goto(server + "/")
    try:
        map_ready(fresh)
    except BaseException:
        close(fresh)
        raise
    fresh.wait_for_selector(".tile[data-rack='A'].tile--live")
    return fresh


def test_typed_destination_is_looked_up_and_picks_the_nearest_live_rack(server, page):
    picture(server, "A", taken_a(5), times=3)
    asked = []
    desk = search_page(server, page, 1280, asked)
    try:
        assert desk.locator("#dest-open").is_hidden() and desk.locator("#dest-q").is_visible()
        assert desk.locator("#dest-places option").count() == 2  # the configured places are suggestions
        desk.click(".tile[data-rack='B'] .tile__head")
        desk.locator("#dest-q").press_sequentially("Basel SBB")
        desk.wait_for_timeout(300)
        assert asked == []  # nothing is looked up while typing
        desk.click("#dest-go")
        desk.wait_for_selector("#dest-result:has-text('walk')")
        assert len(asked) == 1 and "q=Basel%20SBB" in asked[0]
        assert desk.locator("#dest-result").inner_text() == "Main entrance shelter: 10 free, about 6 min walk"
        assert desk.locator(".dest-name").inner_text() == "Basel SBB" and desk.locator("path.dest-line").count() == 1
        assert desk.locator(".tile[data-rack='A'].is-open").count() == 1
        desk.wait_for_function("map.getBounds().contains([47.5476, 7.5897]) && map.getBounds().contains([47.5463, 7.5950])", timeout=8000)
        assert "OpenStreetMap" in desk.locator(".dest-note").inner_text()
    finally:
        close(desk)


def test_a_rack_that_fills_up_hands_over_on_the_next_refresh(server, page):
    picture(server, "A", taken_a(5), times=3)
    a_full = {"on": False}
    desk = search_page(server, page, 1280, [], b_live=True, a_full=a_full)
    try:
        desk.fill("#dest-q", "Basel SBB")
        desk.press("#dest-q", "Enter")
        desk.wait_for_selector("#dest-result:has-text('Main entrance shelter: 10 free')")
        a_full["on"] = True
        desk.wait_for_selector("#dest-result:has-text('next:')", timeout=8000)
        assert desk.locator("#dest-result").inner_text() == "Main entrance shelter is full, next: Building 2, north side: 3 free, about 7 min walk"
        assert desk.locator(".tile[data-rack='B'].is-open").count() == 1
        assert desk.locator("path.dest-line").count() == 1 and desk.locator(".dest-mark").count() == 1
    finally:
        close(desk)


def test_far_destination_says_how_far_and_a_failing_search_keeps_the_rest(server, page):
    picture(server, "A", taken_a(5), times=3)
    desk = search_page(server, page, 1280, [])
    try:
        desk.unroute("**/api/geocode*")
        desk.route("**/api/geocode*", lambda route: route.fulfill(json=[{"name": "Zurich HB", "lat": 47.378, "lng": 8.540}]))
        desk.fill("#dest-q", "Zurich HB")
        desk.press("#dest-q", "Enter")
        desk.wait_for_selector("#dest-result:has-text('from the nearest rack')")
        assert desk.locator("#dest-result").inner_text().endswith(". Zurich HB is 73.4 km from the nearest rack")
        desk.unroute("**/api/geocode*")
        desk.route("**/api/geocode*", lambda route: route.fulfill(status=502, json={"detail": "x"}))
        desk.fill("#dest-q", "Somewhere")
        desk.press("#dest-q", "Enter")
        desk.wait_for_selector("#dest-result:has-text('not answering')")
        desk.fill("#dest-q", "Main entrance")
        desk.press("#dest-q", "Enter")
        desk.wait_for_selector("#dest-result:has-text('Main entrance shelter: 10 free')")
    finally:
        close(desk)


def test_phone_control_folds_into_one_button_clear_of_the_rack_card(server, page):
    picture(server, "A", taken_a(5), times=3)
    asked = []
    phone = search_page(server, page, 390, asked)
    try:
        box = phone.locator(".dest").bounding_box()
        assert phone.locator("#dest-open").is_visible() and phone.locator("#dest-q").is_hidden() and box["height"] < 50
        phone.click(".tile[data-rack='B'] .tile__head")  # rack A opens first; picking it again from another rack flies there
        phone.click(".tile[data-rack='A'] .tile__head")
        phone.wait_for_function("map.getZoom() >= 19", timeout=8000)
        card_a = ".leaflet-marker-icon[title='Main entrance shelter'] .mcard"
        phone.wait_for_selector(card_a)
        phone.wait_for_timeout(500)
        card = phone.locator(card_a).bounding_box()
        box = phone.locator(".dest").bounding_box()
        assert box["y"] + box["height"] <= card["y"] or box["x"] + box["width"] <= card["x"], (box, card)
        phone.click("#dest-open")
        assert phone.locator("#dest-q").is_visible() and phone.locator("#dest-open").get_attribute("aria-expanded") == "true"
        phone.fill("#dest-q", "Basel SBB")
        phone.press("#dest-q", "Enter")
        phone.wait_for_selector("#dest-result:has-text('walk')")
        assert phone.locator("#dest-q").is_hidden() and phone.locator("#dest-result").is_visible()  # folded again, answer kept
        assert phone.evaluate("document.documentElement.scrollWidth") <= 390
    finally:
        close(phone)


def test_skeleton_shows_until_the_first_data(server, page):
    held = []
    other = page.context.browser.new_page(viewport={"width": 1440, "height": 900})
    try:
        other.route("**/api/racks", lambda route: held.append(route))
        other.goto(server + "/")
        for _ in range(100):
            if held:
                break
            other.wait_for_timeout(50)
        assert other.locator("#hero .sk").count() == 4 and other.locator(".tile--sk").count() == 3
        assert other.locator(".tile[data-rack]").count() == 0
        for route in held:
            route.continue_()
        other.unroute("**/api/racks")
        other.wait_for_selector(".tile[data-rack='A']")
        assert other.locator(".sk, .tile--sk").count() == 0
        assert other.locator("#racks").get_attribute("aria-busy") is None
        assert other.locator(".summary__n").inner_text() != ""
    finally:
        other.close()


def test_two_panes_on_a_laptop_and_one_column_on_a_phone(server, page):
    for width, height in ((1440, 900), (390, 844)):
        view = page.context.browser.new_page(viewport={"width": width, "height": height})
        try:
            view.goto(server + "/")
            view.wait_for_selector(".tile[data-rack='A']")
            header, area, pane = (view.locator(s).bounding_box() for s in ("header.hdr", "section.map", "section.pane"))
            if width == 1440:  # header across the top, the map left at full height, the pane right
                assert area["x"] == 0 and abs(area["y"] - header["height"]) < 1 and abs(area["y"] + area["height"] - height) < 1
                assert abs(pane["x"] - area["width"]) < 1 and abs(pane["x"] + pane["width"] - width) < 1 and abs(pane["y"] - area["y"]) < 1
                assert view.evaluate("document.documentElement.scrollHeight") <= height  # the pane scrolls, not the page
            else:  # stacked: the map, then the pane below it, both full width
                assert area["width"] == width and abs(pane["width"] - width) < 1 and pane["y"] >= area["y"] + area["height"] - 1
                assert 320 <= area["height"] <= 520
                assert view.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        finally:
            view.close()


THEME = "document.documentElement.dataset.theme || null"
BACKGROUND = "getComputedStyle(document.body).backgroundColor"
LABEL = "document.getElementById('theme-toggle').getAttribute('aria-label')"


def test_theme_toggle_follows_the_device_until_clicked_and_remembers_the_choice(server, page):
    view = page.context.browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="light")  # own storage
    try:
        view.goto(server + "/")
        toggle = view.locator("header.hdr #theme-toggle")
        light = view.evaluate(BACKGROUND)
        assert view.evaluate(THEME) is None and toggle.get_attribute("aria-label") == "Switch to dark mode"
        assert toggle.locator(".moon").is_visible() and not toggle.locator(".sun").is_visible()
        box = toggle.bounding_box()
        assert box["width"] >= 40 and box["height"] >= 40
        view.emulate_media(color_scheme="dark")  # nothing stored yet: follows the device
        assert view.evaluate(THEME) is None and view.evaluate(BACKGROUND) != light
        view.wait_for_function(f"{LABEL} === 'Switch to light mode'")  # the label follows on the device's change event
        assert toggle.locator(".sun").is_visible()
        view.emulate_media(color_scheme="light")
        view.wait_for_function(f"{LABEL} === 'Switch to dark mode'")
        view.keyboard.press("Tab")  # the first stop on the page
        assert view.evaluate("document.activeElement.id") == "theme-toggle"
        assert toggle.evaluate("(b) => getComputedStyle(b).outlineStyle") == "solid"
        view.keyboard.press("Enter")
        assert view.evaluate(THEME) == "dark" and view.evaluate(BACKGROUND) != light  # colours follow without a reload
        assert toggle.get_attribute("aria-label") == "Switch to light mode" and toggle.locator(".sun").is_visible()
        view.reload()
        assert view.evaluate(THEME) == "dark"
        view.goto(server + "/admin")  # one choice for both pages
        toggle = view.locator("header.masthead #theme-toggle")
        assert view.evaluate(THEME) == "dark" and toggle.get_attribute("aria-label") == "Switch to light mode"
        box = toggle.bounding_box()
        assert box["width"] >= 40 and box["height"] >= 40
        dark = view.evaluate(BACKGROUND)
        toggle.click()
        chosen = view.evaluate(BACKGROUND)
        assert view.evaluate(THEME) == "light" and chosen != dark
        assert toggle.get_attribute("aria-label") == "Switch to dark mode" and toggle.locator(".moon").is_visible()
        view.emulate_media(color_scheme="dark")  # a stored choice wins over the device
        view.reload()
        assert view.evaluate(THEME) == "light" and view.evaluate(BACKGROUND) == chosen
        assert toggle.get_attribute("aria-label") == "Switch to dark mode"
    finally:
        view.close()


def test_stored_theme_is_set_before_the_body_is_parsed(server, page):
    view = page.context.browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="light")

    def probe(route):  # an inline script at the top of <body> reads the theme before any of the page shows
        response = fetch_or_abort(route)
        if response is None:
            return
        body = response.text().replace("<body>", "<body><script>window.themeAtBody = document.documentElement.dataset.theme || null;</script>", 1)
        route.fulfill(response=response, body=body)

    try:
        view.goto(server + "/admin")
        view.evaluate("localStorage.setItem('bikeslot-theme', 'dark')")
        for path in ("/", "/admin"):
            view.route(server + path, probe)
            view.goto(server + path)
            assert view.evaluate("window.themeAtBody") == "dark"
            view.unroute(server + path)
    finally:
        view.close()


def test_theme_toggle_works_when_storage_is_blocked(server, page):
    view = page.context.browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="light")
    errors = []
    view.on("pageerror", lambda exc: errors.append(str(exc)))
    view.add_init_script("Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('blocked', 'SecurityError'); } });")
    try:
        for path in ("/", "/admin"):
            view.goto(server + path)
            view.wait_for_selector(".tile[data-rack], .stat")
            view.click("#theme-toggle")
            assert view.evaluate(THEME) == "dark"
            view.reload()
            assert view.evaluate(THEME) is None  # nothing could be stored: back to the device
        assert errors == []
    finally:
        view.close()


def test_map_keeps_its_colours_in_both_themes(server, page):
    view = page.context.browser.new_page(viewport={"width": 390, "height": 844})
    map_filter = "getComputedStyle(document.documentElement).getPropertyValue('--map-filter').trim()"
    try:
        filters = {}
        for scheme in ("light", "dark"):
            view.emulate_media(color_scheme=scheme)
            view.goto(server + "/")
            filters[scheme] = view.evaluate(map_filter)
        view.emulate_media(color_scheme="light")
        view.evaluate("document.documentElement.dataset.theme = 'dark'")
        assert view.evaluate(map_filter) == filters["dark"]  # the toggle's dark block matches the device's
        assert "grayscale" not in filters["light"] + filters["dark"]
        assert "hue-rotate(180deg)" in filters["dark"]  # turns the inverted colours back: green parks, blue water
    finally:
        view.close()


def test_admin_shows_the_cameras_first_and_the_diagram_below(server, page):
    page.goto(server + "/admin")
    page.wait_for_selector("#cameras .card")
    order = page.evaluate("[...document.querySelectorAll('#cameras, #flow, #kpis, #activity, #detector')].map((node) => node.id)")
    assert order == ["cameras", "flow", "kpis", "activity", "detector"]
    assert page.locator("main h2").first.inner_text().lower() == "cameras"


def test_admin_info_button_opens_on_hover_focus_and_tap_and_closes(server, page):
    desk = page.context.browser.new_page(viewport={"width": 1280, "height": 900})
    try:
        desk.goto(server + "/admin")
        desk.wait_for_selector("#flow .node[data-node='camera'][data-state]")
        button = desk.locator("[aria-controls='flow-help']")
        tip = desk.locator("#flow-help")
        assert button.get_attribute("aria-label") == "How to read the diagram" and desk.locator("#flow figcaption").count() == 0
        assert not tip.is_visible() and button.get_attribute("aria-expanded") == "false"
        button.hover()
        tip.wait_for(state="visible")
        assert "left to right" in tip.inner_text() and button.get_attribute("aria-expanded") == "true"
        box = tip.bounding_box()
        desk.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)  # the pointer can move onto the tip
        assert tip.is_visible()
        desk.mouse.move(5, 5)
        tip.wait_for(state="hidden")
        desk.locator("[aria-controls='cameras-help']").focus()
        assert desk.locator("#cameras-help").is_visible()
        desk.keyboard.press("Tab")  # the next stop is the diagram's info button
        assert desk.evaluate("document.activeElement.getAttribute('aria-controls')") == "flow-help"
        assert not desk.locator("#cameras-help").is_visible()
        tip.wait_for(state="visible")
        desk.keyboard.press("Escape")
        tip.wait_for(state="hidden")
        desk.keyboard.press("Enter")  # a key press toggles
        tip.wait_for(state="visible")
        desk.keyboard.press("Enter")
        tip.wait_for(state="hidden")
    finally:
        close(desk)
    phone = page.context.browser.new_page(viewport={"width": 390, "height": 844}, has_touch=True)
    try:
        phone.goto(server + "/admin")
        phone.wait_for_selector("#cameras .card")
        button = phone.locator("[aria-controls='cameras-help']")
        tip = phone.locator("#cameras-help")
        button.tap()
        tip.wait_for(state="visible")
        assert "chip temperature" in tip.inner_text()
        tip.tap()  # reading it keeps it open
        assert tip.is_visible()
        phone.tap("h1")
        tip.wait_for(state="hidden")
        button.tap()
        tip.wait_for(state="visible")
        button.tap()  # a second tap closes it
        tip.wait_for(state="hidden")
        assert phone.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    finally:
        phone.close()


def test_admin_what_happens_is_collapsed_and_the_live_numbers_show(server, page):
    page.goto(server + "/admin")
    page.wait_for_selector("#flow-detail .flow-title:has-text('Camera node')")
    how = page.locator("#flow-detail details.flow-how")
    assert how.get_attribute("open") is None and not how.locator("p").is_visible()
    rows = page.locator("#flow-detail dl")  # the camera cards above already show every camera number
    assert rows.inner_html() == "" and not rows.is_visible()
    how.locator("summary").click()
    assert "heat guard" in how.locator("p").inner_text()
    page.evaluate("refresh()")
    assert how.locator("p").is_visible()  # patched, not rebuilt: still open after a refresh
    page.locator("#flow .node[data-node='pages']").click()
    assert how.locator("p").inner_text().startswith("Two loops")
    assert "Free spaces now" in page.locator("#flow-detail dl").inner_text()


def test_admin_uptime_is_plain_text_without_a_restart_marker(server, page):
    node = {"tempC": 41.2, "tooHot": False, "wifi_rssi_dbm": -60, "wifi_connected": True, "heap_largest_block": 150000}
    for uptime_s in (4100, 12):  # the uptime dropped: the node restarted
        httpx.post(f"{server}/api/camera/A/alive", json={**node, "uptime_s": uptime_s}, headers={"X-Node-Id": "esp32-a"})
    page.goto(server + "/admin")
    uptime = page.locator("#cameras .card[data-key='A'] [data-f='uptime']")
    uptime.locator("b:has-text('restarted')").wait_for()
    text = uptime.inner_text()
    assert text.startswith("Up 12 s, restarted ") and text.endswith(" ago")
    assert page.locator("#cameras .uptime-bar, #cameras .uptime-reboot").count() == 0 and uptime.locator("svg").count() == 0


def test_admin_page_has_no_demo_labels(server, page):
    picture(server, "C", b"DEMO:101100000000", times=3)  # rack C is demo
    page.goto(server + "/admin")
    page.wait_for_selector("#cameras .card[data-key='C'] .chip")
    words = "() => [document.body.textContent, ...[...document.querySelectorAll('[aria-label], [title]')].map((n) => `${n.getAttribute('aria-label')} ${n.getAttribute('title')}`)].join(' ').toLowerCase()"
    for step in ("camera", "receive", "detector", "decide", "pages"):  # the detail shows one step at a time
        page.locator(f"#flow .node[data-node='{step}']").click()
        page.wait_for_selector(f"#flow .node[data-node='{step}'].selected")
        assert "demo" not in page.evaluate(words)
    assert page.locator("#cameras .chip-real").count() == 2  # racks A and B still stand out as the real cameras


def routed_page(server, page, width: int, change):
    """A fresh page whose /api/racks answer is passed through change(racks) first."""
    fresh = page.context.browser.new_page(viewport={"width": width, "height": 844})

    def racks(route):
        response = fetch_or_abort(route)
        if response is None:
            return
        data = response.json()
        change(data["racks"])
        route.fulfill(response=response, json=data)

    fresh.route("**/api/racks", racks)
    fresh.goto(server + "/")
    return fresh


def live(rack: dict, free: int) -> None:
    rack.update(status="ok", free=free, age_s=1, slots=[{"id": i + 1, "state": "free" if i < free else "taken"} for i in range(rack["capacity"])])


def test_most_space_button_stays_hidden_until_two_racks_are_live(server, page):
    count = {"live": 1}

    def change(racks):
        for index, rack in enumerate(racks):
            if index < count["live"]:
                live(rack, 4 + index)
            else:
                rack.update(status="starting", free=None, slots=[])

    fresh = routed_page(server, page, 390, change)
    try:
        fresh.wait_for_selector(".tile[data-rack='A'].tile--live")
        assert fresh.locator("#hero .btn").is_hidden()  # one live rack: no empty button with only its icons
        count["live"] = 2
        fresh.wait_for_selector("#hero .btn:visible", timeout=8000)
        assert fresh.locator("#hero .btn__label").inner_text() == "Most space: Building 2, north side"
    finally:
        close(fresh)


def test_only_the_selected_rack_shows_its_card_on_the_map(server, page):
    picture(server, "A", taken_a(5), times=3)
    page.goto(server + "/")
    map_ready(page)
    page.wait_for_selector(".tile[data-rack='A'].is-open")
    page.evaluate("map.setView([47.5467, 7.5960], 19, {animate: false})")
    page.wait_for_selector(".leaflet-marker-icon .mcard")
    assert page.locator(".leaflet-marker-icon .mcard").count() == 1
    assert page.locator(".leaflet-marker-icon[title='Main entrance shelter'] .mcard").count() == 1
    assert page.locator(".leaflet-marker-icon[title='Building 2, north side'] .pin").count() == 1
    page.evaluate("map.setView([47.5472, 7.5969], 19, {animate: false})")
    page.click(".leaflet-marker-icon[title='Building 2, north side'] .pin")  # the card moves to the tapped rack
    page.wait_for_selector(".leaflet-marker-icon[title='Building 2, north side'] .mcard")
    assert page.locator(".leaflet-marker-icon .mcard").count() == 1
    assert page.locator(".leaflet-marker-icon[title='Main entrance shelter'] .pin").count() == 1
    page.click(".tile[data-rack='B'] .tile__head")  # nothing selected: no card at all
    page.wait_for_selector(".leaflet-marker-icon .mcard", state="detached")
    assert page.locator(".leaflet-marker-icon .pin").count() == 2


LABEL_OVERLAPS = """(plan) => { const boxes = [...plan.querySelectorAll('.cell > span')].filter((s) => !s.hidden)
    .map((s) => s.getBoundingClientRect()).sort((a, b) => a.left - b.left);
    return boxes.some((box, i) => i > 0 && box.left < boxes[i - 1].right + 1); }"""


def test_plan_numbers_show_only_where_they_fit(server, page):
    for capacity in (8, 15, 29, 36):

        def change(racks, capacity=capacity):
            rack = racks[0]
            rack["capacity"] = capacity
            rack["layout"].update(width=capacity, spaces=[{"id": i + 1, "x": i, "y": 0, "w": 1, "h": 1} for i in range(capacity)])
            live(rack, capacity - 3)
            for other in racks[1:]:  # rack A alone is live, so its tile opens first
                other.update(status="starting", free=None, slots=[])

        for width in (390, 1280):
            fresh = routed_page(server, page, width, change)
            try:
                fresh.wait_for_selector(".tile[data-rack='A'].is-open .plan .cell--free")
                plan = fresh.locator(".tile[data-rack='A'] .tile__live .plan")
                assert plan.locator(".cell").count() == capacity
                shown = plan.locator(".cell > span:not([hidden])").all_inner_texts()
                assert shown[0] == "1" and not plan.evaluate(LABEL_OVERLAPS), (capacity, width, shown)
                if capacity >= 29:  # narrow cells: 1 and every 5th, under the cells
                    assert len(shown) < capacity and "5" in shown and "10" in shown, (capacity, width, shown)
                    assert "is-sparse" in plan.get_attribute("class")
                if capacity == 8:
                    assert len(shown) == 8
                expected = f"Free spaces: 1 to {capacity - 3}"
                assert fresh.locator(".tile[data-rack='A'] .plan-free").inner_text() == expected
            finally:
                close(fresh)


def test_read_once_texts_sit_behind_info_buttons(server, page):
    view = page.context.browser.new_page(viewport={"width": 1280, "height": 800})
    try:
        view.goto(server + "/")
        view.wait_for_selector(".tile[data-rack='A']")
        privacy = view.locator(".pane__foot .info__btn")
        assert privacy.get_attribute("aria-label") == "Privacy" and view.locator("#privacy-note").is_hidden()
        privacy.hover()
        assert view.locator("#privacy-note").inner_text() == "Only counts are kept. No pictures are stored."
        view.keyboard.press("Escape")
        assert view.locator("#privacy-note").is_hidden()  # Escape closes it under the pointer too
        if view.locator(".leaflet-control .dest").count():
            note = view.locator("#dest-note")
            assert note.is_hidden() and view.locator(".dest .info__btn").get_attribute("aria-label") == "About the place search"
            view.locator(".dest .info__btn").click()
            assert note.is_visible() and view.locator(".dest .info__btn").get_attribute("aria-expanded") == "true"
            view.mouse.click(5, 300)  # a tap elsewhere closes it
            assert note.is_hidden() and view.locator(".dest .info__btn").get_attribute("aria-expanded") == "false"
            view.locator(".dest .info__btn").focus()
            view.keyboard.press("Shift+Tab")
            view.keyboard.press("Tab")  # keyboard focus shows it
            assert note.is_visible()
    finally:
        view.close()


def as_location(racks: list) -> None:
    """Rack A and a second section beside it, served as one location the way the server sends a group."""
    a = racks[0]
    live(a, 9)
    second = {**a, "id": "A9", "name": "Second section", "location": {"lat": a["location"]["lat"] - 0.00012, "lng": a["location"]["lng"] + 0.00009}}
    live(second, 0)
    second["status"] = "starting"
    racks[0] = {**a, "id": "Row", "name": "Main row", "capacity": 2 * a["capacity"], "free": 9, "layout": None, "slots": [], "parts": [a, second],
                "location": {"lat": a["location"]["lat"] - 0.00006, "lng": a["location"]["lng"] + 0.000045}}


def test_a_location_is_one_pin_one_entry_and_its_parts_as_rows(server, page):
    for width in (390, 1280):
        fresh = routed_page(server, page, width, as_location)
        try:
            map_ready(fresh)
            fresh.wait_for_selector(".tile[data-rack='Row'].is-open .part")
            assert fresh.locator(".tile[data-rack]").count() == 3 and fresh.locator("#rack-count").inner_text() == "3"
            assert fresh.locator(".tile[data-rack='Row'] .count__n").inner_text() == "9"
            assert fresh.locator(".tile[data-rack='Row'] .count__of").inner_text() == f"free of {2 * CAPACITY_A}"
            rows = fresh.locator(".tile[data-rack='Row'] .part")
            assert rows.count() == 2 and rows.locator(".plan").count() == 2 and rows.locator(".plan__entry").count() == 2
            assert rows.locator(".part__n").all_inner_texts() == ["9", "?"]  # the free count ends each row, unknown is never free
            assert fresh.locator(".tile[data-rack='Row'] .plan-free").count() == 0
            assert fresh.locator(".leaflet-marker-icon[title='Main row']").count() == 1
            assert fresh.locator(".leaflet-marker-icon").count() == 2  # the location and rack B; C has no place
            fresh.evaluate("map.setView([47.54624, 7.59505], 19, {animate: false})")
            fresh.wait_for_selector(".leaflet-marker-icon[title='Main row'] .mcard .seg")
            assert fresh.locator(".mcard .seg").all_inner_texts() == ["9", "?"]
            assert fresh.locator("path.plan-space").count() == 2 * CAPACITY_A  # both sections drawn on the ground
            assert fresh.evaluate("document.documentElement.scrollWidth") <= width
        finally:
            close(fresh)


def test_a_pick_on_a_phone_scrolls_the_opened_entry_into_view(server, page):
    def change(racks):
        live(racks[0], 9)
        live(racks[1], 4)

    fresh = routed_page(server, page, 390, change)
    try:
        map_ready(fresh)
        fresh.wait_for_selector(".tile[data-rack='A'].is-open")
        fresh.evaluate("map.setView([47.5472, 7.5969], 17, {animate: false})")
        fresh.click(".leaflet-marker-icon[title='Building 2, north side'] .pin")
        fresh.wait_for_selector(".tile[data-rack='B'].is-open")
        # at the top of the screen, or as far as the page scrolls
        fresh.wait_for_function("""() => { const top = document.querySelector(".tile[data-rack='B']").getBoundingClientRect().top;
            const end = Math.abs(scrollY + innerHeight - document.documentElement.scrollHeight) < 2;
            return scrollY > 0 && top >= -1 && (top < 2 || end); }""", timeout=8000)
    finally:
        close(fresh)


def test_folded_place_button_is_the_whole_box_and_the_map_key_shows_shapes_only(server, page):
    phone = page.context.browser.new_page(viewport={"width": 390, "height": 844})
    try:
        phone.goto(server + "/")
        map_ready(phone)
        button, box = phone.locator("#dest-open").bounding_box(), phone.locator(".dest").bounding_box()
        assert button["height"] >= 44 and box["height"] - button["height"] <= 2 and box["width"] - button["width"] <= 2
        key = phone.locator("#map-key")
        assert key.locator(".pin").all_inner_texts() == ["", "", ""]
        assert key.inner_text().split() == ["Free", "Full", "No", "data"]
    finally:
        phone.close()


GEO_CALLS = """window.geoCalls = 0;
    const read = navigator.geolocation.getCurrentPosition.bind(navigator.geolocation);
    navigator.geolocation.getCurrentPosition = (...args) => { window.geoCalls += 1; return read(...args); };"""


def test_the_open_tile_says_how_many_are_free_on_arrival_without_sending_the_position(server, page):
    picture(server, "A", taken_a(5), times=3)
    context = page.context.browser.new_context(
        viewport={"width": 390, "height": 844},
        permissions=["geolocation"],
        geolocation={"latitude": 47.5467, "longitude": 7.5950},  # about 45 m north of rack A: 1 min by bike
    )
    sent = []
    context.on("request", lambda request: sent.append(f"{request.url} {request.post_data or ''}"))
    near = context.new_page()
    try:
        near.goto(server + "/")
        near.wait_for_selector(".tile[data-rack='A']")
        if not near.locator(".tile[data-rack='A'].is-open").count():
            near.click(".tile[data-rack='A'] .tile__head")
        line = near.locator(".tile[data-rack='A'] #forecast .arrival")
        line.wait_for(timeout=10000)
        assert line.inner_text() == "About 10 free when you arrive (in 1 min)"
        assert near.locator(".pos-ask").count() == 0
        assert any("/api/racks/A/history?minutes=1 " in request for request in sent)
        assert not [request for request in sent if "47.546" in request or "7.595" in request]
    finally:
        context.close()


def test_without_a_position_there_is_no_arrival_line_and_no_prompt_on_load(server, page):
    picture(server, "A", taken_a(5), times=3)
    context = page.context.browser.new_context(viewport={"width": 390, "height": 844})  # permission not decided
    context.add_init_script(GEO_CALLS)
    sent = []
    context.on("request", lambda request: sent.append(request.url))
    fresh = context.new_page()
    try:
        fresh.goto(server + "/")
        fresh.wait_for_selector(".tile.is-open #forecast .chart")
        offer = fresh.locator(".tile.is-open .pos-ask")
        offer.wait_for(timeout=5000)
        assert offer.inner_text() == "On arrival" and fresh.locator(".arrival").count() == 0
        assert fresh.evaluate("window.geoCalls") == 0
        assert not [url for url in sent if "minutes=" in url]
    finally:
        context.close()
