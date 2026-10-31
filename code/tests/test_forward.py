"""Push mode towards the app (forward.py) and the live-camera seeding of the demo layout. No model needed."""
from bikeslot import config, forward
from bikeslot.schemas import CameraStatus


def test_parse_racks_ignores_malformed_parts():
    assert forward.parse_racks("live01=MX-CAM, cam02=B,,=x,y=") == {"live01": "MX-CAM", "cam02": "B"}
    assert forward.parse_racks("") == {}


def test_disabled_without_an_app_url():
    f = forward.Forwarder("", "live01=A")
    assert f.enabled is False and f.submit("live01", CameraStatus(camera_id="live01")) is False
    assert f.info() == {"app_url": None, "targets": []}


def test_latest_status_wins_and_is_posted_with_the_token():
    calls = []

    def post(url, payload, headers):
        calls.append((url, payload, headers))
        return 200, {"ok": True, "rack": "MX-CAM", "free": 13, "applied": True}

    f = forward.Forwarder("http://app.test:8080/", "live01=MX-CAM", token="s3cret", post=post)
    assert f.enabled and f.targets["live01"].url == "http://app.test:8080/api/vision/MX-CAM/status"
    assert f.submit("live01", CameraStatus(camera_id="live01", free=12)) is True
    assert f.submit("other", CameraStatus(camera_id="other")) is False
    f.submit("live01", CameraStatus(camera_id="live01", free=13))
    assert f.flush() == 1 and len(calls) == 1          # two submits, one post: only the newest status matters
    url, payload, headers = calls[0]
    assert url == f.targets["live01"].url and payload["free"] == 13 and payload["camera_id"] == "live01"
    assert headers == {"Content-Type": "application/json", "X-Vision-Token": "s3cret"}
    target = f.info()["targets"][0]
    assert (target["sent"], target["errors"], target["last_error"], target["last_reply"]["free"]) == (1, 0, None, 13)
    assert f.flush() == 0


def test_errors_are_counted_and_never_raise():
    answers = [ConnectionError("refused"), (401, {"detail": "missing or wrong X-Vision-Token"}), (200, {"ok": True})]

    def post(url, payload, headers):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    f = forward.Forwarder("http://app.test", {"live01": "A"}, post=post)
    for _ in range(3):
        f.submit("live01", {"camera_id": "live01", "free": 1})
        f.flush()
    t = f.targets["live01"]
    assert (t.sent, t.errors, t.last_error) == (1, 2, None)
    assert f.info()["targets"][0]["last_ok_age_s"] is not None


def test_module_start_uses_the_config(monkeypatch):
    monkeypatch.setattr(config, "APP_URL", "http://app.test:8080")
    monkeypatch.setattr(config, "APP_RACKS", "live01=MX-CAM")
    monkeypatch.setattr(config, "APP_TOKEN", "")
    f = forward.start()
    try:
        assert forward.info()["app_url"] == "http://app.test:8080" and forward.info()["targets"][0]["rack_id"] == "MX-CAM"
        assert forward.submit("live01", CameraStatus(camera_id="live01")) is True
    finally:
        forward.stop()
        forward.forwarder = forward.Forwarder("", {})
    assert f.enabled


def test_demo_layout_is_seeded_for_the_live_camera_with_renamed_slot_ids(tmp_path):
    from bikeslot.store import Store

    store = Store(tmp_path)
    layout = store.seed_layout_file(config.DEMO_LAYOUT, camera_id="live01")
    assert layout is not None and layout.camera_id == "live01" and len(layout.slots) == 18
    assert layout.slots[0].slot_id == "live01-S01" and layout.slots[-1].slot_id == "live01-S18"
    assert layout.params.get("engine") == "classifier" and layout.status == "confirmed"
    assert store.get("live01").layout is layout and store.get("demo01") is None
    assert not (tmp_path / "layouts" / "live01.json").exists()   # seeded in memory only; an admin layout would be saved
    # the file's own camera keeps its ids
    demo = store.seed_layout_file(config.DEMO_LAYOUT)
    assert demo.camera_id == "demo01" and demo.slots[0].slot_id == "demo01-S01"
