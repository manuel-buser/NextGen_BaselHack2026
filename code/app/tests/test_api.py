import asyncio
import json
import logging
import time
from datetime import date, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from bikeslot.config import Settings
from bikeslot.geocoder import USER_AGENT, Geocoder, HideQueries
from bikeslot.main import create_app
from bikeslot.poller import CameraPoller
from bikeslot.state import PullCamera
from tools.fake_node import make_app as make_fake_node

CAPACITY = 4


@pytest.fixture
def racks_file(tmp_path):
    path = tmp_path / "racks.yaml"
    path.write_text(f"racks:\n  - id: A\n    name: Test rack\n    capacity: {CAPACITY}\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def client(racks_file):
    settings = Settings(racks_file=racks_file, detector_url="fake", free_after_clean=3)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def picture(client: TestClient, taken: int):
    # The fake detector marks len(body) % (capacity + 1) slots as taken.
    body = b"x" * (10 * (CAPACITY + 1) + taken)
    return client.post("/api/camera/A/picture", content=body, headers={"Content-Type": "image/jpeg", "X-Node-Id": "n1"})


def test_racks_listed_before_any_picture(client):
    data = client.get("/api/racks").json()
    rack = data["racks"][0]
    assert (rack["id"], rack["capacity"], rack["free"], rack["status"]) == ("A", CAPACITY, None, "unknown")


def test_free_shows_after_three_clean_pictures(client):
    assert picture(client, taken=1).json()["ok"] is True
    picture(client, taken=1)
    assert client.get("/api/racks/A").json()["status"] == "starting"
    assert picture(client, taken=1).json() == {"ok": True, "rack": "A", "free": 3, "capacity": CAPACITY, "status": "ok"}
    rack = client.get("/api/racks/A").json()
    assert [s["state"] for s in rack["slots"]] == ["taken", "free", "free", "free"]
    assert rack["location"] is None


def test_taken_flips_at_once_and_free_needs_three(client):
    for _ in range(3):
        picture(client, taken=1)
    assert picture(client, taken=2).json()["free"] == 2
    assert picture(client, taken=1).json()["free"] == 2
    picture(client, taken=1)
    assert picture(client, taken=1).json()["free"] == 3


def test_alive_and_admin_status(client):
    assert client.post("/api/camera/A/alive", json={"tempC": 41.5, "tooHot": False}, headers={"X-Node-Id": "esp-1"}).status_code == 200
    status = client.get("/api/admin/status").json()
    camera = status["cameras"][0]
    assert (camera["rack"], camera["temp_c"], camera["too_hot"], camera["node_id"], camera["online"]) == ("A", 41.5, False, "esp-1", True)
    assert status["detector"]["mode"] == "fake" and status["detector"]["reachable"] is True
    assert status["app"]["version"]
    assert status["app"]["min_confidence"] == 0.4  # the admin page shows the confidence rule


def test_unknown_rack_is_404(client):
    assert client.post("/api/camera/Z/alive", json={}).status_code == 404
    assert client.post("/api/camera/Z/picture", content=b"abc").status_code == 404
    assert client.get("/api/racks/Z").status_code == 404


def test_empty_picture_is_400(client):
    assert client.post("/api/camera/A/picture", content=b"").status_code == 400


def test_pages_are_served(client):
    assert "Find my Bike Slot" in client.get("/").text
    assert "Bike Slot admin" in client.get("/admin").text
    assert client.get("/api/health").json() == {"status": "ok"}


def test_detector_down_gives_503_and_admin_shows_it(racks_file):
    settings = Settings(racks_file=racks_file, detector_url="http://127.0.0.1:9", detector_timeout_s=1.0)
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/camera/A/picture", content=b"abc")
        assert response.status_code == 503 and response.json()["ok"] is False
        status = client.get("/api/admin/status").json()
        assert status["detector"]["reachable"] is False
        assert status["detector_results"]["last_error"]
        assert status["cameras"][0]["pictures"] == 1


def test_demo_rack_is_labelled_and_decided_by_the_stand_in(tmp_path):
    path = tmp_path / "racks.yaml"
    path.write_text(
        f"racks:\n  - {{id: A, name: Real rack, capacity: {CAPACITY}}}\n  - {{id: C, name: Demo rack, capacity: {CAPACITY}, demo: true}}\n",
        encoding="utf-8",
    )
    # The detector is unreachable: a real rack gets 503, a demo rack never asks it.
    settings = Settings(racks_file=str(path), detector_url="http://127.0.0.1:9", detector_timeout_s=1.0, free_after_clean=1)
    with TestClient(create_app(settings)) as client:
        assert {r["id"]: r["demo"] for r in client.get("/api/racks").json()["racks"]} == {"A": False, "C": True}
        response = client.post("/api/camera/C/picture", content=b"DEMO:1010", headers={"X-Node-Id": "demo-C"})
        assert response.json() == {"ok": True, "rack": "C", "free": 2, "capacity": CAPACITY, "status": "ok"}
        assert [s["state"] for s in client.get("/api/racks/C").json()["slots"]] == ["taken", "free", "taken", "free"]
        assert client.post("/api/camera/A/picture", content=b"DEMO:1010").status_code == 503
        status = client.get("/api/admin/status").json()
        assert status["detector_results"]["last_ok_at"] is None
        assert [(c["rack"], c["demo"], c["node_id"]) for c in status["cameras"]] == [("A", False, None), ("C", True, "demo-C")]


@pytest.mark.parametrize("path, demo_ids", [("racks.yaml", []), ("racks.demo.yaml", ["B", "C"])])
def test_shipped_racks_files_load(path, demo_ids):
    with TestClient(create_app(Settings(racks_file=path, camera_pull=False))) as client:  # never contact a real node
        racks = client.get("/api/racks").json()["racks"]
    assert [r["id"] for r in racks if r["demo"]] == demo_ids
    assert racks[0]["id"] == "A" and all(r["location"] for r in racks)


def layout_app(tmp_path, layout: str, capacity: int = 6):
    path = tmp_path / "racks.yaml"
    path.write_text(f"racks:\n  - {{id: A, name: Test rack, capacity: {capacity}, layout: {layout}}}\n", encoding="utf-8")
    return create_app(Settings(racks_file=str(path), detector_url="fake"))


def test_grid_layout_is_normalised(tmp_path):
    with TestClient(layout_app(tmp_path, "{rows: 2, cols: 3, bearing: 35}")) as client:
        layout = client.get("/api/racks/A").json()["layout"]
    assert (layout["bearing"], layout["entrance"], layout["width"], layout["height"]) == (35, "start", 3, 2)
    assert [(s["id"], s["x"], s["y"], s["w"], s["h"]) for s in layout["spaces"]] == [
        (1, 0, 0, 1, 1), (2, 1, 0, 1, 1), (3, 2, 0, 1, 1), (4, 0, 1, 1, 1), (5, 1, 1, 1, 1), (6, 2, 1, 1, 1)]


def test_explicit_spaces_layout(tmp_path):
    spaces = "[{id: 2, x: 2.5, y: 0}, {id: 1, x: 0, y: 0, w: 1, h: 2}]"
    with TestClient(layout_app(tmp_path, f"{{spaces: {spaces}, bearing: 350, entrance: end}}", capacity=2)) as client:
        layout = client.get("/api/racks/A").json()["layout"]
    assert (layout["bearing"], layout["entrance"], layout["width"], layout["height"]) == (350, "end", 3.5, 2)
    # sorted by id, w and h default to 1
    assert layout["spaces"] == [{"id": 1, "x": 0, "y": 0, "w": 1, "h": 2}, {"id": 2, "x": 2.5, "y": 0, "w": 1, "h": 1}]


def test_layout_is_null_without_one(client):
    assert client.get("/api/racks/A").json()["layout"] is None


def test_layout_ids_must_match_the_capacity(tmp_path):
    with pytest.raises(ValueError, match="rack A: layout space ids must be exactly 1 to 6"):
        layout_app(tmp_path, "{rows: 2, cols: 2}")
    with pytest.raises(ValueError, match="bearing must be 0 to 360"):
        layout_app(tmp_path, "{rows: 2, cols: 3, bearing: 400}")


# Pull mode: the app polls the node. The fake node's first status says 51.3 C and its first picture has 5 taken.
PULL_CAPACITY = 15
NODE_STATUS = {
    "uptime_s": 155, "temperature_c": 51.3, "stream_active": False, "stream_frames": 0, "stream_dropped": 0,
    "heap_free": 202840, "heap_min_free": 198216, "heap_largest_block": 159732, "psram_free": 7987060,
    "psram_total": 8388608, "wifi_connected": True, "wifi_rssi_dbm": -38, "wifi_channel": 5, "ip": "192.168.21.95",
}


def pull_app(tmp_path, camera: str = "{url: 'http://node.test'}", **settings):
    path = tmp_path / "racks.yaml"
    path.write_text(f"racks:\n  - {{id: A, name: Pull rack, capacity: {PULL_CAPACITY}, camera: {camera}}}\n", encoding="utf-8")
    return create_app(Settings(racks_file=str(path), detector_url="fake", free_after_clean=1, camera_pull=False, **settings))


def poll(poller: CameraPoller, transport, now: float) -> None:
    # One round of the poller loop with the node mounted in-process, instead of the sleeping background task.
    async def one_round():
        async with httpx.AsyncClient(transport=transport) as client:
            await poller.poll_once(client, now)

    asyncio.run(one_round())


def node_answering(status: dict):
    return httpx.MockTransport(lambda request: httpx.Response(200, json=status) if request.url.path == "/status" else httpx.Response(200, content=b"x" * 16))


def test_pull_round_marks_the_camera_online_and_ingests_a_picture(tmp_path):
    with TestClient(pull_app(tmp_path)) as client:
        rack = client.app.state.store.racks["A"]
        assert rack.pull == PullCamera("http://node.test") and rack.pull.status_url == "http://node.test/status"
        poll(CameraPoller(client.app, rack), httpx.ASGITransport(app=make_fake_node()), now=time.time())
        camera = client.get("/api/admin/status").json()["cameras"][0]
        assert (camera["online"], camera["source"], camera["node_id"], camera["temp_c"]) == (True, "pull", "node.test", 51.3)
        assert (camera["too_hot"], camera["too_hot_derived"], camera["last_error"]) == (False, True, None)
        assert (camera["node"]["wifi_rssi_dbm"], camera["node"]["heap_largest_block"], camera["node"]["ip"]) == (-38, 159732, "127.0.0.1")
        assert camera["node"]["uptime_s"] >= 0 and camera["node"]["rebooted_at"] is None
        # the picture went through the same rules as a posted one: 5 of 15 taken by the stand-in's length rule
        assert camera["pictures"] == 1 and camera["last_picture_bytes"] == 1605
        rack_view = client.get("/api/racks/A").json()
        assert (rack_view["free"], rack_view["status"]) == (10, "ok")


def test_pull_fetches_a_picture_only_when_due(tmp_path):
    with TestClient(pull_app(tmp_path, "{url: 'http://node.test', poll_s: 5, picture_every_s: 10}")) as client:
        poller = CameraPoller(client.app, client.app.state.store.racks["A"])
        node = httpx.ASGITransport(app=make_fake_node())
        for now in (1000.0, 1005.0, 1009.9, 1010.0):  # status every round, a picture at 1000 and 1010
            poll(poller, node, now)
        camera = client.get("/api/admin/status").json()["cameras"][0]
        assert (camera["pictures"], camera["last_picture_at"], camera["last_alive_at"]) == (2, 1010.0, 1010.0)


def test_pull_too_hot_is_derived_unless_the_node_says_and_a_reboot_is_noticed(tmp_path):
    with TestClient(pull_app(tmp_path, hot_limit_c=60.0)) as client:
        poller = CameraPoller(client.app, client.app.state.store.racks["A"])

        def camera() -> dict:
            return client.get("/api/admin/status").json()["cameras"][0]

        poll(poller, node_answering({**NODE_STATUS, "temperature_c": 66.0}), now=time.time())
        assert (camera()["too_hot"], camera()["too_hot_derived"], camera()["node"]["ip"]) == (True, True, "192.168.21.95")
        poll(poller, node_answering({**NODE_STATUS, "temperature_c": 66.0, "too_hot": False}), now=time.time())
        assert (camera()["too_hot"], camera()["too_hot_derived"], camera()["node"]["uptime_s"]) == (False, False, 155)
        poll(poller, node_answering({**NODE_STATUS, "uptime_s": 3}), now=time.time())
        assert camera()["node"]["uptime_s"] == 3 and camera()["node"]["rebooted_at"] is not None


def test_pull_status_failure_is_recorded_and_the_camera_stays_offline(tmp_path):
    with TestClient(pull_app(tmp_path, "{url: 'http://node.test', status_path: '/nope'}")) as client:
        poller = CameraPoller(client.app, client.app.state.store.racks["A"])
        poll(poller, httpx.ASGITransport(app=make_fake_node()), now=time.time())
        camera = client.get("/api/admin/status").json()["cameras"][0]
        assert (camera["online"], camera["last_alive_at"], camera["pictures"]) == (False, None, 0)
        assert camera["last_error"] == "status: HTTP 404" and camera["error_age_s"] is not None

        def refused(request):
            raise httpx.ConnectError("connection refused")

        poll(poller, httpx.MockTransport(refused), now=time.time())
        camera = client.get("/api/admin/status").json()["cameras"][0]
        assert camera["last_error"] == "status: ConnectError: connection refused" and camera["online"] is False


def test_pull_tasks_start_with_the_app_unless_turned_off(tmp_path):
    path = tmp_path / "racks.yaml"
    path.write_text(f"racks:\n  - {{id: A, name: Pull rack, capacity: 4, camera: {{url: 'http://127.0.0.1:9'}}}}\n", encoding="utf-8")
    app = create_app(Settings(racks_file=str(path), detector_url="fake"))
    with TestClient(app):
        assert [task.get_name() for task in app.state.poll_tasks] == ["poll-A"]
    assert all(task.done() for task in app.state.poll_tasks)
    with TestClient(create_app(Settings(racks_file=str(path), detector_url="fake", camera_pull=False))) as client:
        assert client.app.state.poll_tasks == []


def test_camera_block_is_validated(tmp_path):
    with pytest.raises(ValueError, match="rack A: camera url must start with http"):
        pull_app(tmp_path, "{status_path: /status}")
    with pytest.raises(ValueError, match="rack A: camera poll_s and picture_every_s must be above 0"):
        pull_app(tmp_path, "{url: 'http://node.test', poll_s: 0}")
    with pytest.raises(ValueError, match="must start with /"):
        pull_app(tmp_path, "{url: 'http://node.test', picture_path: capture}")


def test_push_alive_uses_the_same_field_mapping(client):
    client.post("/api/camera/A/alive", json={"tempC": 80.5}, headers={"X-Node-Id": "esp-1"})
    camera = client.get("/api/admin/status").json()["cameras"][0]
    assert (camera["too_hot"], camera["too_hot_derived"], camera["source"], camera["node_id"]) == (True, True, "push", "esp-1")
    client.post("/api/camera/A/alive", json=NODE_STATUS, headers={"X-Node-Id": "esp-1"})
    camera = client.get("/api/admin/status").json()["cameras"][0]
    assert (camera["temp_c"], camera["too_hot"], camera["node"]["wifi_rssi_dbm"], camera["node"]["uptime_s"]) == (51.3, False, -38, 155)


# History and prediction: only free counts per rack over time. A demo rack runs on the driver's clock (X-Demo-Clock).
HISTORY_RACKS = f"racks:\n  - {{id: A, name: Real rack, capacity: {CAPACITY}}}\n  - {{id: C, name: Demo rack, capacity: {CAPACITY}, demo: true}}\n"
# A morning that fills up: 4 free until 07:45, 1 free at 08:00, full from 08:15 (spaces taken per time).
MORNING = [("07:00", 0), ("07:30", 0), ("07:45", 0), ("08:00", 3), ("08:15", 4), ("08:30", 4), ("09:00", 4)]


def history_app(tmp_path, history_file=None):
    path = tmp_path / "racks.yaml"
    path.write_text(HISTORY_RACKS, encoding="utf-8")
    return create_app(Settings(racks_file=str(path), detector_url="fake", free_after_clean=1, history_file=history_file))


def demo_picture(client: TestClient, taken: int, clock: str, rack: str = "C"):
    body = b"DEMO:" + ("1" * taken + "0" * (CAPACITY - taken)).encode()
    return client.post(f"/api/camera/{rack}/picture", content=body, headers={"X-Demo-Clock": clock})


def play_mornings(client: TestClient, days: list[str]) -> None:
    for day in days:
        for hhmm, taken in MORNING:
            assert demo_picture(client, taken, f"{day}T{hhmm}").json()["status"] == "ok"


def test_full_from_0815_on_three_weekdays_is_learned(tmp_path):
    with TestClient(history_app(tmp_path)) as client:
        play_mornings(client, ["2026-11-02", "2026-11-03", "2026-11-04"])  # Monday to Wednesday
        history = client.get("/api/racks/C/history").json()
        demo_picture(client, 4, "2026-11-07T08:15")  # a Saturday: weekend days are learned apart
        weekend = client.get("/api/racks/C/history").json()
    assert (history["clock"], history["now"], history["day_type"], history["free_now"]) == ("demo", "09:00", "weekday", 0)
    assert history["insights"] == {"usually_full_by": "08:15", "best_before": "07:45", "usual_free_now": 0.0, "days": 3, "learning": False}
    assert (weekend["day_type"], weekend["insights"]["days"], weekend["insights"]["usually_full_by"]) == ("weekend", 1, None)


def test_one_day_only_is_still_learning(tmp_path):
    with TestClient(history_app(tmp_path)) as client:
        play_mornings(client, ["2026-11-02"])
        insights = client.get("/api/racks/C/history").json()["insights"]
        forecast = client.get("/api/racks/C").json()["forecast"]
    assert insights == {"usually_full_by": None, "best_before": None, "usual_free_now": 0.0, "days": 1, "learning": True}
    assert forecast == {"usually_full_by": None, "usual_free_now": 0.0, "days": 1, "learning": True}


def test_at_most_one_sample_per_rack_per_minute(tmp_path):
    with TestClient(history_app(tmp_path)) as client:
        for taken in (1, 2, 3):
            demo_picture(client, taken, "2026-11-02T07:35")
        demo_picture(client, 0, "2026-11-02T07:36")
        samples = client.app.state.store.history.samples["C"]
        typical = client.get("/api/racks/C/history").json()["typical"]
    assert samples == {"2026-11-02T07:35": (1, CAPACITY), "2026-11-02T07:36": (4, CAPACITY)}  # the latest in a minute wins
    assert typical[30] == {"t": "07:30", "free": 2.5, "full_share": 0.0, "n": 2}


def test_demo_clock_is_used_for_a_demo_rack_and_ignored_for_a_real_one(tmp_path):
    with TestClient(history_app(tmp_path)) as client:
        demo_picture(client, 1, "2026-11-02T07:35")
        demo = client.get("/api/racks/C/history").json()
        client.post("/api/camera/A/picture", content=b"x" * (10 * (CAPACITY + 1) + 1), headers={"X-Demo-Clock": "2026-11-02T07:35"})
        real = client.get("/api/racks/A/history").json()
        recorded = list(client.app.state.store.history.samples["A"])
        demo_picture(client, 1, "not a time")  # a malformed clock counts as none: back to the real clock
        assert client.get("/api/racks/C/history").json()["clock"] == "real"
    assert list(client.app.state.store.history.samples["C"])[0] == "2026-11-02T07:35"
    assert (demo["clock"], demo["now"], demo["day_type"], demo["free_now"]) == ("demo", "07:35", "weekday", 3)
    assert (real["clock"], real["free_now"], len(recorded)) == ("real", 3, 1)
    assert abs(datetime.strptime(recorded[0], "%Y-%m-%dT%H:%M") - datetime.now()) < timedelta(minutes=2)


def test_history_shape_and_unknown_rack(tmp_path):
    with TestClient(history_app(tmp_path)) as client:
        assert client.get("/api/racks/Z/history").status_code == 404
        empty = client.get("/api/racks/C/history").json()
        play_mornings(client, ["2026-11-02"])
        history = client.get("/api/racks/C/history").json()
    assert set(history) == {"rack", "clock", "now", "day_type", "free_now", "capacity", "today", "typical", "insights"}
    assert (empty["clock"], empty["free_now"], empty["today"], empty["insights"]["days"]) == ("real", None, [], 0)
    assert (history["rack"], history["capacity"], len(history["typical"])) == ("C", CAPACITY, 96)
    assert history["typical"][0] == {"t": "00:00", "free": None, "full_share": 0.0, "n": 0}
    assert history["typical"][33] == {"t": "08:15", "free": 0.0, "full_share": 1.0, "n": 1}
    assert history["today"] == [{"t": t, "free": float(CAPACITY - taken)} for t, taken in MORNING]


def test_every_rack_carries_a_forecast(tmp_path):
    with TestClient(history_app(tmp_path)) as client:
        play_mornings(client, ["2026-11-02", "2026-11-03"])
        forecasts = {r["id"]: r["forecast"] for r in client.get("/api/racks").json()["racks"]}
    assert forecasts == {
        "A": {"usually_full_by": None, "usual_free_now": None, "days": 0, "learning": True},
        "C": {"usually_full_by": "08:15", "usual_free_now": 0.0, "days": 2, "learning": False},
    }


def test_history_file_setting(monkeypatch):
    monkeypatch.delenv("HISTORY_FILE", raising=False)
    assert Settings().history_file is None and Settings.from_env().history_file == "data/history.json"
    monkeypatch.setenv("HISTORY_FILE", "")
    assert Settings.from_env().history_file is None  # empty: kept in memory only


def test_history_survives_a_restart_and_a_bad_file_starts_empty(tmp_path, caplog):
    file = tmp_path / "data" / "history.json"  # the folder is made on the first save
    monday = date.today() - timedelta(days=date.today().weekday() + 7)  # last week, well inside the 8 weeks kept
    days = [(monday + timedelta(days=i)).isoformat() for i in range(3)]
    with TestClient(history_app(tmp_path, str(file))) as client:
        play_mornings(client, days)
    saved = json.loads(file.read_text(encoding="utf-8"))
    assert len(saved["racks"]["C"]) == 3 * len(MORNING) and not file.with_name("history.json.tmp").exists()
    saved["racks"]["C"]["2020-01-06T08:15"] = [0, CAPACITY]  # older than 8 weeks: dropped on load
    file.write_text(json.dumps(saved), encoding="utf-8")
    with TestClient(history_app(tmp_path, str(file))) as client:
        samples = client.app.state.store.history.samples["C"]
        assert len(samples) == 3 * len(MORNING) and "2020-01-06T08:15" not in samples
        demo_picture(client, 4, f"{days[2]}T09:00")
        assert client.get("/api/racks/C/history").json()["insights"]["usually_full_by"] == "08:15"
    file.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="bikeslot.history"):
        with TestClient(history_app(tmp_path, str(file))) as client:
            assert client.app.state.store.history.samples == {}
            assert client.get("/api/racks/C/history").json()["insights"]["days"] == 0
    assert "starting empty" in caplog.text


def test_layout_space_size_defaults_and_custom_values(tmp_path):
    with TestClient(layout_app(tmp_path, "{rows: 2, cols: 3}")) as client:
        layout = client.get("/api/racks/A").json()["layout"]
    assert (layout["space_m"], layout["row_gap_m"]) == ([0.7, 1.9], 0.6)
    with TestClient(layout_app(tmp_path, "{rows: 2, cols: 3, space_m: [0.8, 2], row_gap_m: 1.2}")) as client:
        layout = client.get("/api/racks/A").json()["layout"]
    assert (layout["space_m"], layout["row_gap_m"]) == ([0.8, 2], 1.2)


def test_layout_space_size_must_be_sensible(tmp_path):
    with pytest.raises(ValueError, match="rack A: layout space_m must be"):
        layout_app(tmp_path, "{rows: 2, cols: 3, space_m: [0.7]}")
    with pytest.raises(ValueError, match="rack A: layout space_m must be"):
        layout_app(tmp_path, "{rows: 2, cols: 3, space_m: [0, 1.9]}")
    with pytest.raises(ValueError, match="rack A: layout row_gap_m must be"):
        layout_app(tmp_path, "{rows: 2, cols: 3, row_gap_m: -1}")


def places_app(tmp_path, places: str):
    path = tmp_path / "racks.yaml"
    path.write_text(f"racks:\n  - {{id: A, name: Test rack, capacity: 2}}\n{places}", encoding="utf-8")
    return create_app(Settings(racks_file=str(path), detector_url="fake"))


def test_places_are_listed_with_the_racks(tmp_path):
    places = "places:\n  - {name: Canteen, location: {lat: 47.5468, lng: 7.5981}}\n"
    with TestClient(places_app(tmp_path, places)) as client:
        assert client.get("/api/racks").json()["places"] == [{"name": "Canteen", "location": {"lat": 47.5468, "lng": 7.5981}}]
    with TestClient(places_app(tmp_path, "")) as client:
        assert client.get("/api/racks").json()["places"] == []


def test_places_need_a_name_and_a_location(tmp_path):
    with pytest.raises(ValueError, match="places: each place needs a name and a location"):
        places_app(tmp_path, "places:\n  - {name: Canteen}\n")
    with pytest.raises(ValueError, match="places: Canteen: location is off the globe"):
        places_app(tmp_path, "places:\n  - {name: Canteen, location: {lat: 147.5, lng: 7.6}}\n")


NOMINATIM = [{"name": f"Place {i}", "display_name": f"Place {i}, Basel", "lat": f"47.5{i}", "lon": f"7.5{i}"} for i in range(7)]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def geocode_app(tmp_path, handler, url="https://geo.test/search", clock=None):
    path = tmp_path / "racks.yaml"
    path.write_text("racks:\n  - {id: A, name: Test rack, capacity: 2, location: {lat: 47.5463, lng: 7.5950}}\n", encoding="utf-8")
    app = create_app(Settings(racks_file=str(path), geocoder_url=url))
    if url:  # the same geocoder, but answered by the handler and timed by a fake clock: no internet, no real waits
        clock = clock or FakeClock()
        box = app.state.geocoder.box
        app.state.geocoder = Geocoder(url, box, transport=httpx.MockTransport(handler), clock=clock.clock, sleep=clock.sleep)
    return app


def test_geocode_maps_results_limits_them_to_five_and_prefers_the_racks_area(tmp_path):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"name": "", "display_name": "Bahnhof SBB, Basel", "lat": "47.5476", "lon": "7.5897"}, *NOMINATIM])

    with TestClient(geocode_app(tmp_path, handler)) as client:
        response = client.get("/api/geocode", params={"q": "  Basel   SBB "})
    assert response.status_code == 200
    assert response.json()[:2] == [{"name": "Bahnhof SBB", "lat": 47.5476, "lng": 7.5897}, {"name": "Place 0", "lat": 47.5, "lng": 7.5}]
    assert len(response.json()) == 5
    params = seen[0].url.params
    assert (params["q"], params["format"], params["limit"]) == ("Basel SBB", "jsonv2", "5")
    assert params["viewbox"] == "7.545,47.5963,7.645,47.4963" and "bounded" not in params and "countrycodes" not in params
    assert seen[0].headers["user-agent"] == USER_AGENT and "@" not in USER_AGENT


def test_geocode_answers_a_repeated_query_from_the_cache(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["q"])
        return httpx.Response(200, json=NOMINATIM[:1])

    with TestClient(geocode_app(tmp_path, handler)) as client:
        first = client.get("/api/geocode", params={"q": "Basel SBB"}).json()
        again = client.get("/api/geocode", params={"q": "basel  sbb"}).json()
    assert first == again and calls == ["Basel SBB"]


def test_geocode_calls_upstream_one_at_a_time_at_most_once_a_second():
    clock = FakeClock()
    times = []

    def handler(request: httpx.Request) -> httpx.Response:
        times.append(clock.now)
        return httpx.Response(200, json=NOMINATIM[:1])

    async def three_at_once():
        geocoder = Geocoder("https://geo.test/search", transport=httpx.MockTransport(handler), clock=clock.clock, sleep=clock.sleep)
        await asyncio.gather(*(geocoder.search(q) for q in ("one", "two", "three", "one")))
        await geocoder.aclose()

    asyncio.run(three_at_once())
    assert times == [1000.0, 1001.0, 1002.0]  # three distinct queries, spaced a second apart; the repeat came from the cache


def test_geocode_rejects_empty_or_long_queries_and_is_404_when_off(tmp_path):
    with TestClient(geocode_app(tmp_path, lambda request: httpx.Response(200, json=[]))) as client:
        assert client.get("/api/geocode").status_code == 400
        assert client.get("/api/geocode", params={"q": "   "}).status_code == 400
        assert client.get("/api/geocode", params={"q": "x" * 121}).status_code == 400
    with TestClient(geocode_app(tmp_path, None, url="")) as client:
        assert client.get("/api/geocode", params={"q": "Basel SBB"}).status_code == 404


@pytest.mark.parametrize(
    "answer",
    [httpx.Response(503, text="busy"), httpx.Response(200, text="not json"), httpx.Response(200, json=[{"name": "x"}]), None],
)
def test_geocode_upstream_failure_is_502_and_not_cached(tmp_path, answer):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if answer is None:
            raise httpx.ConnectError("no route", request=request)
        return answer

    with TestClient(geocode_app(tmp_path, handler)) as client:
        for _ in range(2):
            response = client.get("/api/geocode", params={"q": "Basel SBB"})
            assert response.status_code == 502 and response.json()["detail"] == "The place search is not answering right now."
    assert len(calls) == 2


def test_typed_places_stay_out_of_the_logs():
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:5", "GET", "/api/geocode?q=Basel%20SBB", "1.1", 200), None)
    HideQueries(("/api/geocode",)).filter(record)
    assert "Basel" not in record.getMessage() and "/api/geocode?(hidden)" in record.getMessage()
