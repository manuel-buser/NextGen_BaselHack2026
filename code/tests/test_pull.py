"""Pull mode (pull.py) against the relay: fresh picture, 304, 404, 401, stale picture, the view key, /health."""
from bikeslot import config, pull
from bikeslot.schemas import utcnow
from bikeslot.store import Store

URL = "https://relay.test/snapshot?key=SECRET"
HEALTH = "https://relay.test/health?key=SECRET"
JPEG = b"\xff\xd8\xff\xe0 not really a picture \xff\xd9"


class Reply:
    def __init__(self, status=200, headers=None, content=b"", body=None):
        self.status_code, self.headers, self.content, self._body = status, headers or {}, content, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code} for url: {URL}")   # requests puts the URL into the message


class Session:
    """Answers per URL (a list is consumed in order, an exception is raised); records every request's headers."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, dict(headers or {})))
        answer = self.answers[url]
        if isinstance(answer, list):
            answer = answer.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def relay_health(online=True, seen=0.3, age=0.4) -> dict:
    return {"camera_id": "cam01", "online": online, "last_seen_s": seen, "last_snapshot_age_s": age,
            "last_health_age_s": seen, "snapshots_received": 46, "health_received": 46,
            "health": {"camera_id": "cam01", "uptime_s": 459, "reset_reason": "usb", "temperature_c": 55.3,
                       "wifi_rssi_dbm": -41, "wifi_connected": True, "heap_free": 41963, "pushes_ok": 91,
                       "pushes_failed": 0, "last_push_age_s": 0}}


def puller() -> pull.Puller:
    return pull.Puller("live01", URL, view_key="vk")


def test_parse_redact_and_health_url():
    assert pull.parse("live01=https://relay.test/snapshot?key=abc, cam02=http://10.0.0.2/snapshot") == {
        "live01": "https://relay.test/snapshot?key=abc", "cam02": "http://10.0.0.2/snapshot"}
    assert pull.parse("http://10.0.0.5/snapshot") == {"live01": "http://10.0.0.5/snapshot"}
    assert pull.redact(URL) == "https://relay.test/snapshot?key=***"
    assert pull.redact("Max retries exceeded with url: /snapshot?camera=cam01&key=SECRET (Caused by X)") == \
        "Max retries exceeded with url: /snapshot?camera=cam01&key=*** (Caused by X)"
    assert pull.health_url(URL) == HEALTH and pull.health_url("http://10.0.0.2/capture") is None


def test_fresh_picture_is_handled_then_304_means_nothing_new(monkeypatch):
    monkeypatch.setattr(config, "PULL_HEALTH", False)
    got, notes = [], []
    sess = Session({URL: [Reply(200, {"ETag": '"cam01-46"', "X-Snapshot-Age-S": "0.7"}, JPEG),
                          Reply(304, {"ETag": '"cam01-46"', "X-Snapshot-Age-S": "4.9"})]})
    p = puller()
    pull.poll_once(p, sess, lambda cam, data: got.append((cam, data)), note=lambda cam, src: notes.append(src))
    assert got == [("live01", JPEG)] and (p.state, p.frames, p.etag, p.last_error) == ("ok", 1, '"cam01-46"', None)
    assert sess.calls[0] == (URL, {"X-View-Key": "vk"})
    pull.poll_once(p, sess, lambda cam, data: got.append((cam, data)), note=lambda cam, src: notes.append(src))
    assert len(got) == 1 and (p.state, p.not_modified, p.errors, p.last_error) == ("not_modified", 1, 0, None)
    assert sess.calls[1] == (URL, {"X-View-Key": "vk", "If-None-Match": '"cam01-46"'})
    assert notes[-1]["state"] == "not_modified" and notes[-1]["last_frame_age_s"] is not None
    assert p.info()["url"] == "https://relay.test/snapshot?key=***" and p.info()["frames"] == 1


def test_no_picture_yet_and_wrong_key_are_told_apart(monkeypatch, capsys):
    monkeypatch.setattr(config, "PULL_HEALTH", False)
    p, got = puller(), []
    pull.poll_once(p, Session({URL: Reply(404)}), lambda c, d: got.append(d))
    assert got == [] and (p.state, p.errors) == ("waiting", 0) and "404" in p.last_error
    pull.poll_once(p, Session({URL: Reply(401)}), lambda c, d: got.append(d))
    assert (p.state, p.errors) == ("error", 1) and "view key refused (401)" in p.last_error and "BIKESLOT_PULL_VIEW_KEY" in p.last_error
    pull.poll_once(p, Session({URL: Reply(401)}), lambda c, d: got.append(d))
    out = capsys.readouterr().out
    assert out.count("view key refused") == 1 and "SECRET" not in out   # each problem logged once, key masked


def test_old_pictures_are_not_used(monkeypatch):
    monkeypatch.setattr(config, "PULL_HEALTH", False)
    monkeypatch.setattr(config, "PULL_MAX_AGE_S", 30.0)
    p, got = puller(), []
    pull.poll_once(p, Session({URL: Reply(200, {"ETag": '"cam01-9"', "X-Snapshot-Age-S": "125.0"}, JPEG)}), lambda c, d: got.append(d))
    assert got == [] and (p.state, p.stale, p.etag) == ("stale", 1, '"cam01-9"') and "125 s old" in p.last_error
    pull.poll_once(p, Session({URL: Reply(304, {"X-Snapshot-Age-S": "130.0"})}), lambda c, d: got.append(d))
    assert got == [] and p.state == "stale"
    pull.poll_once(p, Session({URL: Reply(200, {"ETag": '"cam01-10"', "X-Snapshot-Age-S": "1.0"}, JPEG)}), lambda c, d: got.append(d))
    assert got == [JPEG] and (p.state, p.last_error) == ("ok", None)


def test_exceptions_are_counted_and_the_key_masked(monkeypatch):
    monkeypatch.setattr(config, "PULL_HEALTH", False)
    p = puller()
    pull.poll_once(p, Session({URL: ConnectionError(f"Max retries exceeded with url: {URL}")}), lambda c, d: None)
    assert (p.state, p.errors) == ("error", 1) and "SECRET" not in p.last_error and "key=***" in p.last_error


def test_health_is_read_next_to_the_snapshot_and_decides_online(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PULL_HEALTH", True)
    monkeypatch.setattr(config, "PULL_MAX_AGE_S", 60.0)
    store = Store(tmp_path)
    cam = store.get_or_create("live01")

    def handle(camera_id, data):
        cam.last_frame_at = utcnow()

    p = puller()
    sess = Session({URL: Reply(200, {"ETag": '"cam01-1"', "X-Snapshot-Age-S": "0.5"}, JPEG), HEALTH: Reply(200, body=relay_health())})
    pull.poll_once(p, sess, handle, note=store.note_source)
    assert p.source == {"kind": "relay", "online": True, "camera_id": "cam01", "last_seen_s": 0.3, "snapshot_age_s": 0.4,
                        "temperature_c": 55.3, "wifi_rssi_dbm": -41, "wifi_connected": True, "uptime_s": 459,
                        "pushes_ok": 91, "pushes_failed": 0, "last_push_age_s": 0}
    assert sess.calls[1] == (HEALTH, {"X-View-Key": "vk"})          # the key, but no If-None-Match, on /health
    st = store.current_status(cam)
    assert st.online is True and st.source["kind"] == "relay" and st.source["state"] == "ok" and "offline" not in st.message
    # the relay stops hearing from the camera: offline at once, with the reason, although our last frame is recent
    pull.poll_once(p, Session({URL: Reply(304, {"X-Snapshot-Age-S": "40.0"}),
                               HEALTH: Reply(200, body=relay_health(online=False, seen=40.0, age=40.0))}), handle, note=store.note_source)
    st = store.current_status(cam)
    assert st.online is False and st.message.endswith("camera offline (nothing new at the relay for 40 s)")
    # the relay still says online (health keeps coming) but its newest picture is too old: offline as well
    pull.poll_once(p, Session({URL: Reply(304, {"X-Snapshot-Age-S": "95.0"}), HEALTH: Reply(200, body=relay_health(age=95.0))}),
                   handle, note=store.note_source)
    st = store.current_status(cam)
    assert st.online is False and "95 s old" in st.message and st.source["state"] == "stale"
    # nothing at the relay at all yet
    pull.poll_once(p, Session({URL: Reply(404), HEALTH: Reply(404)}), handle, note=store.note_source)
    assert store.current_status(cam).message.endswith("camera offline (nothing at the relay yet)")
    # a fresh picture again: online, message clean
    pull.poll_once(p, Session({URL: Reply(200, {"ETag": '"cam01-2"', "X-Snapshot-Age-S": "0.2"}, JPEG), HEALTH: Reply(200, body=relay_health())}),
                   handle, note=store.note_source)
    assert store.current_status(cam).online is True


def test_camera_health_and_a_failing_health_endpoint(monkeypatch):
    monkeypatch.setattr(config, "PULL_HEALTH", True)
    # the ESP32's own /health (LAN pull) has no online key: the camera answered, so it is online
    assert pull.summarize_health({"camera_id": "cam01", "temperature_c": 51.0, "uptime_s": 5, "heap_free": 1}) == {
        "kind": "camera", "online": True, "camera_id": "cam01", "last_seen_s": 0.0, "snapshot_age_s": None,
        "temperature_c": 51.0, "uptime_s": 5}
    p = puller()
    pull.poll_once(p, Session({URL: Reply(200, {"ETag": '"x-1"'}, JPEG), HEALTH: RuntimeError("boom")}), lambda c, d: None)
    assert p.state == "ok" and p.source["health_error"] == "RuntimeError: boom"   # the picture still counts
