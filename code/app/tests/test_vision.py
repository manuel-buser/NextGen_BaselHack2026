"""The vision path: a rack whose spaces are decided by the vision service (code/bikeslot), read by polling (pull)
or received from the service (push). The stand-in in tools/fake_vision.py serves the service's JSON shape."""

import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from bikeslot.config import Settings
from bikeslot.main import create_app
from bikeslot.vision import VisionPoller, vision_result
from tools.fake_vision import make_app as make_fake_vision

CAPACITY = 18
OCCUPIED = (3, 6, 10, 14, 18)  # the demo picture
STAMP = "2026-11-01T08:00:00+00:00"


def status_json(occupied=OCCUPIED, updated_at=STAMP, capacity=CAPACITY, unknown=()) -> dict:
    """A CameraStatus as the vision service serves it (the fields the app reads, plus a few it ignores)."""
    def state(i):
        return "unknown" if i in unknown else "occupied" if i in occupied else "free"

    slots = [{"slot_id": f"live01-S{i:02d}", "index": i, "status": state(i), "confidence": 1.0, "score": 0.9, "polygon": []} for i in range(1, capacity + 1)]
    counts = {s: sum(1 for x in slots if x["status"] == s) for s in ("free", "occupied", "unknown")}
    return {"camera_id": "live01", "site_id": "site01", "online": True, "frame_age_s": 1.5, "updated_at": updated_at,
            "layout_status": "confirmed", "occupancy_mode": "per_slot", "estimated": False, "frame_size": [1280, 960],
            "slot_count": capacity, **counts, "slots": slots, "detections": {"person": 0, "bicycle": counts["occupied"], "motorcycle": 0},
            "processing_ms": 42.0, "preview_url": "/cameras/live01/preview.jpg", "message": ""}


def vision_app(tmp_path, vision: str = "{url: 'http://vision.test:8000', camera: live01}", more: str = "", fields: str = "", **settings):
    """One vision rack A; `fields` adds to its mapping (", demo: true"), `more` adds further racks (block style)."""
    path = tmp_path / "racks.yaml"
    path.write_text(f"racks:\n  - {{id: A, name: Camera stretch, capacity: {CAPACITY}, vision: {vision}{fields}}}\n{more}", encoding="utf-8")
    return create_app(Settings(racks_file=str(path), detector_url="fake", camera_pull=False, **settings))


def poll(poller: VisionPoller, transport, now: float) -> None:
    async def one_round():
        async with httpx.AsyncClient(transport=transport) as client:
            await poller.poll_once(client, now)

    asyncio.run(one_round())


def camera_of(client: TestClient) -> dict:
    return client.get("/api/admin/status").json()["cameras"][0]


def states(result: dict) -> dict:
    return {s["id"]: s["state"] for s in result["slots"]}


def test_vision_result_passes_decided_states_and_counts_unknown_as_occupied():
    result, warning = vision_result(status_json(unknown=(1,)), CAPACITY)
    assert warning is None and [s["id"] for s in result["slots"]] == list(range(1, CAPACITY + 1))
    assert (states(result)[1], states(result)[2], states(result)[3]) == ("taken", "free", "taken")
    assert all(s["confidence"] == 1.0 for s in result["slots"])
    every_taken = {i: "taken" for i in range(1, CAPACITY + 1)}
    assert states(vision_result(status_json(unknown=range(1, CAPACITY + 1)), CAPACITY)[0]) == every_taken
    assert states(vision_result({"slots": []}, CAPACITY)[0]) == every_taken == states(vision_result({}, CAPACITY)[0])


def test_vision_result_warns_when_the_numbering_does_not_fit_the_rack():
    result, warning = vision_result(status_json(capacity=19), CAPACITY)
    assert warning == "slot 19 is outside 1 to 18, the rack's capacity" and len(result["slots"]) == CAPACITY
    result, warning = vision_result(status_json(capacity=17), CAPACITY)
    assert warning == "the service reports 17 slots, the rack has 18" and states(result)[18] == "taken"  # not reported


def test_vision_block_is_validated(tmp_path):
    with pytest.raises(ValueError, match="rack A: vision needs the camera id"):
        vision_app(tmp_path, "{url: 'http://vision.test:8000'}")
    with pytest.raises(ValueError, match="rack A: vision url must start with http"):
        vision_app(tmp_path, "{url: 'vision.test:8000', camera: live01}")
    with pytest.raises(ValueError, match="rack A: vision poll_s must be above 0"):
        vision_app(tmp_path, "{camera: live01, poll_s: 0}")
    with pytest.raises(ValueError, match="rack A: vision cannot be combined with camera or demo"):
        vision_app(tmp_path, "{camera: live01}", fields=", demo: true")
    with pytest.raises(ValueError, match="rack A: vision cannot be combined with camera or demo"):
        vision_app(tmp_path, "{camera: live01}", fields=", camera: {url: 'http://node.test'}")


def test_pushed_status_applies_at_once_and_only_a_new_frame_counts(tmp_path):
    with TestClient(vision_app(tmp_path, "{camera: live01}")) as client:  # push only: no url, nothing to poll
        rack = client.app.state.store.racks["A"]
        assert rack.window == 1 and rack.vision.url is None and rack.vision.node_id == "live01"
        assert client.get("/api/racks/A").json()["source"] == "vision"
        reply = client.post("/api/vision/A/status", json=status_json())
        assert reply.json() == {"ok": True, "rack": "A", "free": 13, "capacity": CAPACITY, "status": "ok", "applied": True}
        view = client.get("/api/racks/A").json()
        assert (view["free"], view["status"], view["mode"]) == (13, "ok", "slots")
        assert [s["id"] for s in view["slots"] if s["state"] == "taken"] == list(OCCUPIED)
        camera = camera_of(client)
        assert (camera["source"], camera["node_id"], camera["online"], camera["pictures"], camera["last_picture_bytes"]) == ("vision", "live01", True, 1, 0)
        assert (camera["vision"]["how"], camera["vision"]["free"], camera["vision"]["occupied"], camera["vision"]["preview"]) == ("push", 13, 5, False)
        # the same frame again: alive, but nothing new to apply
        assert client.post("/api/vision/A/status", json=status_json()).json()["applied"] is False
        assert camera_of(client)["pictures"] == 1
        # a new frame with one more bike: taken at once (the service has smoothed already)
        reply = client.post("/api/vision/A/status", json=status_json(occupied=OCCUPIED + (7,), updated_at="2026-11-01T08:00:05+00:00"))
        assert (reply.json()["free"], reply.json()["applied"]) == (12, True)
        assert camera_of(client)["pictures"] == 2
        # and one less: free at once too
        reply = client.post("/api/vision/A/status", json=status_json(occupied=(3,), updated_at="2026-11-01T08:00:10+00:00"))
        assert reply.json()["free"] == 17
        status = client.get("/api/admin/status").json()
        assert status["detector_results"]["last_ok_at"] is None  # the vision service says nothing about DETECTOR_URL


def test_pushed_status_needs_the_token_when_one_is_set(tmp_path):
    with TestClient(vision_app(tmp_path, "{camera: live01}", vision_token="s3cret")) as client:
        assert client.post("/api/vision/A/status", json=status_json()).status_code == 401
        assert client.post("/api/vision/A/status", json=status_json(), headers={"X-Vision-Token": "wrong"}).status_code == 401
        assert client.post("/api/vision/A/status", json=status_json(), headers={"X-Vision-Token": "s3cret"}).status_code == 200


def test_push_rejects_unknown_racks_racks_without_vision_and_bad_bodies(tmp_path):
    more = "  - {id: B, name: Plain rack, capacity: 4}\n"
    with TestClient(vision_app(tmp_path, "{camera: live01}", more=more)) as client:
        assert client.post("/api/vision/Z/status", json=status_json()).status_code == 404
        assert client.post("/api/vision/B/status", json=status_json()).status_code == 409
        assert client.post("/api/vision/A/status", content=b"not json", headers={"Content-Type": "application/json"}).status_code == 400
        assert client.post("/api/vision/A/status", json=[1, 2]).status_code == 400
        assert camera_of(client)["last_error"] == "vision: the body is not a status object"


def test_undecided_status_shows_the_rack_full(tmp_path):
    with TestClient(vision_app(tmp_path, "{camera: live01}")) as client:
        reply = client.post("/api/vision/A/status", json=status_json(unknown=range(1, CAPACITY + 1)))
        assert reply.json() == {"ok": True, "rack": "A", "free": 0, "capacity": CAPACITY, "status": "ok", "applied": True}
        camera = camera_of(client)
        assert (camera["online"], camera["pictures"], camera["vision"]["unknown"]) == (True, 1, CAPACITY)
        # a person in front of two free hooks: the other sixteen are decided, the two count as occupied
        reply = client.post("/api/vision/A/status", json=status_json(unknown=(1, 2), updated_at="2026-11-01T08:00:05+00:00"))
        assert (reply.json()["free"], reply.json()["status"]) == (11, "ok")
        assert [s["state"] for s in client.get("/api/racks/A").json()["slots"]][:2] == ["taken", "taken"]
        reply = client.post("/api/vision/A/status", json=status_json(updated_at="2026-11-01T08:00:10+00:00"))
        assert reply.json()["free"] == 13


def test_slot_numbering_outside_the_capacity_is_reported(tmp_path):
    with TestClient(vision_app(tmp_path, "{camera: live01}")) as client:
        assert client.post("/api/vision/A/status", json=status_json(capacity=19)).json()["free"] == 13
        camera = camera_of(client)
        assert camera["last_error"] == "vision: slot 19 is outside 1 to 18, the rack's capacity" and camera["vision"]["warning"]


def test_offline_camera_shows_the_rack_full_on_every_status(tmp_path):
    with TestClient(vision_app(tmp_path, "{camera: live01}", stale_after_s=10)) as client:
        assert client.post("/api/vision/A/status", json=status_json()).json()["free"] == 13
        offline = {**status_json(), "online": False}  # the same old frame, the camera has gone quiet
        reply = client.post("/api/vision/A/status", json=offline).json()
        assert (reply["free"], reply["status"], reply["applied"]) == (0, "ok", True)
        assert client.get("/api/racks/A").json()["tracking"] is False  # drawn grey on the page
        assert camera_of(client)["online"] is False  # the service answers, but its camera has no picture
        client.post("/api/vision/A/status", json=status_json(updated_at="2026-11-01T08:00:05+00:00"))
        assert (client.get("/api/racks/A").json()["free"], client.get("/api/racks/A").json()["tracking"]) == (13, True)


def test_old_pictures_show_the_rack_full_and_the_source_fills_the_camera_card(tmp_path):
    with TestClient(vision_app(tmp_path, "{camera: live01}", stale_after_s=10)) as client:
        assert client.post("/api/vision/A/status", json=status_json()).json()["free"] == 13
        # the service is up but its last picture is old (the camera stopped sending): full, not tracked live
        old = {**status_json(updated_at="2026-11-01T08:00:05+00:00"), "frame_age_s": 95.0}
        reply = client.post("/api/vision/A/status", json=old).json()
        assert (reply["free"], reply["applied"]) == (0, True) and client.get("/api/racks/A").json()["tracking"] is False
        # what the service learned from the relay reaches the admin page and the camera card's instruments
        src = {"kind": "relay", "online": True, "camera_id": "cam01", "last_seen_s": 0.3, "snapshot_age_s": 0.4, "state": "ok",
               "last_error": None, "temperature_c": 55.3, "wifi_rssi_dbm": -41, "wifi_connected": True, "uptime_s": 459}
        client.post("/api/vision/A/status", json={**status_json(updated_at="2026-11-01T08:00:10+00:00"), "source": src})
        camera = camera_of(client)
        assert camera["vision"]["source"] == src and client.get("/api/racks/A").json()["tracking"] is True
        assert (camera["temp_c"], camera["too_hot"], camera["node"]["wifi_rssi_dbm"], camera["node"]["uptime_s"], camera["node"]["wifi_connected"]) == (55.3, False, -41, 459, True)
        # the relay has not heard from the camera: the service says offline, the rack goes grey at once
        silent = {**status_json(updated_at="2026-11-01T08:00:15+00:00"), "online": False, "source": {**src, "online": False, "last_seen_s": 40.0}}
        assert client.post("/api/vision/A/status", json=silent).json()["free"] == 0
        assert camera_of(client)["online"] is False and camera_of(client)["vision"]["source"]["last_seen_s"] == 40.0


def test_pull_round_reads_the_service_and_the_rack_shows_its_spaces(tmp_path):
    fake = make_fake_vision(capacity=CAPACITY, occupied=OCCUPIED, frame_s=1000)  # one frame, until the rehearsal endpoints force a new one
    with TestClient(vision_app(tmp_path)) as client:
        rack = client.app.state.store.racks["A"]
        assert rack.vision.status_url == "http://vision.test:8000/cameras/live01/status"
        poller = VisionPoller(client.app, rack)
        poll(poller, httpx.ASGITransport(app=fake), now=time.time())
        camera = camera_of(client)
        assert (camera["online"], camera["source"], camera["node_id"], camera["pictures"], camera["last_error"]) == (True, "vision", "vision.test", 1, None)
        assert (camera["vision"]["how"], camera["vision"]["url"], camera["vision"]["preview"], camera["vision"]["processing_ms"]) == ("pull", "http://vision.test:8000", True, 12.5)
        view = client.get("/api/racks/A").json()
        assert (view["free"], view["status"]) == (13, "ok")
        assert [s["id"] for s in view["slots"] if s["state"] == "taken"] == list(OCCUPIED)
        poll(poller, httpx.ASGITransport(app=fake), now=time.time())
        assert camera_of(client)["pictures"] == 1  # the same frame: alive, nothing new
        with TestClient(fake) as service:
            service.post("/fake/park/7")
        poll(poller, httpx.ASGITransport(app=fake), now=time.time())
        assert (client.get("/api/racks/A").json()["free"], camera_of(client)["pictures"]) == (12, 2)


def test_pull_errors_are_recorded_and_show_the_rack_full(tmp_path):
    with TestClient(vision_app(tmp_path, stale_after_s=10)) as client:
        poller = VisionPoller(client.app, client.app.state.store.racks["A"])
        poll(poller, httpx.ASGITransport(app=make_fake_vision(camera="cam9")), now=time.time())
        camera = camera_of(client)
        assert camera["last_error"] == "vision: camera 'live01' is not known to the service yet (it appears with its first frame)"
        assert (camera["online"], camera["pictures"]) == (False, 0)
        assert client.get("/api/racks/A").json()["free"] == 0

        def refused(request):
            raise httpx.ConnectError("connection refused")

        poll(poller, httpx.ASGITransport(app=make_fake_vision(capacity=CAPACITY, occupied=OCCUPIED)), now=time.time())
        assert client.get("/api/racks/A").json()["free"] == 13
        poll(poller, httpx.MockTransport(refused), now=time.time())
        assert camera_of(client)["last_error"] == "vision: ConnectError: connection refused"
        assert client.get("/api/racks/A").json()["free"] == 0
        poll(poller, httpx.MockTransport(lambda request: httpx.Response(200, content=b"<html>")), now=time.time())
        assert camera_of(client)["last_error"].startswith("vision: JSONDecodeError")
        # a good round, then silence: the rack shows stale after STALE_AFTER_S without a new frame
        poll(poller, httpx.ASGITransport(app=make_fake_vision(frame_s=1000)), now=1000.0)
        rack = client.app.state.store.racks["A"]
        assert rack.status(client.app.state.settings, 1005.0) == "ok" and rack.status(client.app.state.settings, 1011.0) == "stale"


def test_preview_is_fetched_through_the_app(tmp_path):
    with TestClient(vision_app(tmp_path)) as client:
        client.app.state.vision_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=make_fake_vision()))
        response = client.get("/api/vision/A/preview.jpg")
        assert response.status_code == 200 and response.headers["content-type"].startswith("image/svg+xml")
        assert response.headers["cache-control"] == "no-store" and b"<svg" in response.content
        assert client.get("/api/vision/Z/preview.jpg").status_code == 404
    with TestClient(vision_app(tmp_path, "{camera: live01}")) as client:
        assert client.get("/api/vision/A/preview.jpg").status_code == 404  # push mode: no address to fetch from
    with TestClient(vision_app(tmp_path)) as client:
        def down(request):
            raise httpx.ConnectError("connection refused")

        client.app.state.vision_http = httpx.AsyncClient(transport=httpx.MockTransport(down))
        assert client.get("/api/vision/A/preview.jpg").status_code == 502


def test_vision_poll_task_starts_with_the_app_unless_push_only_or_turned_off(tmp_path):
    path = tmp_path / "racks.yaml"
    path.write_text("racks:\n  - {id: A, name: Vision rack, capacity: 4, vision: {url: 'http://127.0.0.1:9', camera: live01}}\n", encoding="utf-8")
    app = create_app(Settings(racks_file=str(path), detector_url="fake"))
    with TestClient(app):
        assert [task.get_name() for task in app.state.poll_tasks] == ["vision-A"]
    assert all(task.done() for task in app.state.poll_tasks)
    with TestClient(create_app(Settings(racks_file=str(path), detector_url="fake", camera_pull=False))) as client:
        assert client.app.state.poll_tasks == []
    path.write_text("racks:\n  - {id: A, name: Vision rack, capacity: 4, vision: {camera: live01}}\n", encoding="utf-8")
    with TestClient(create_app(Settings(racks_file=str(path), detector_url="fake"))) as client:
        assert client.app.state.poll_tasks == []
