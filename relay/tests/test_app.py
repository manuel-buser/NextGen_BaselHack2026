import time

import pytest
from fastapi.testclient import TestClient

import app as relay

TOKEN = "t" * 32
KEY = "view-key"
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"


def make_client(view_key=KEY, **kw):
    s = relay.Settings(upload_token=TOKEN, view_key=view_key, online_timeout_s=kw.get("online_timeout_s", 30),
                       max_snapshot_bytes=kw.get("max_snapshot_bytes", 10_000), max_cameras=kw.get("max_cameras", 2))
    return TestClient(relay.create_app(s))


def up(cam="cam01", token=TOKEN, **extra):
    return {"X-Camera-Id": cam, "X-Upload-Token": token, **extra}


@pytest.fixture
def c():
    return make_client()


def test_refuses_to_start_without_token(monkeypatch):
    monkeypatch.delenv("UPLOAD_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        relay.Settings.from_env()
    monkeypatch.setenv("UPLOAD_TOKEN", "short")
    with pytest.raises(RuntimeError):
        relay.Settings.from_env()


def test_healthz_is_open(c):
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_snapshot_roundtrip_with_headers_and_etag(c):
    r = c.post("/ingest/snapshot", content=JPEG, headers=up(**{"X-People-Blurred": "2", "X-Plates-Blurred": "1"}))
    assert r.status_code == 204
    r = c.get("/snapshot", params={"key": KEY})
    assert r.status_code == 200 and r.content == JPEG
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-people-blurred"] == "2" and r.headers["x-plates-blurred"] == "1"
    assert r.headers["x-camera-id"] == "cam01"
    etag = r.headers["etag"]
    assert c.get("/snapshot", params={"key": KEY}, headers={"If-None-Match": etag}).status_code == 304
    c.post("/ingest/snapshot", content=JPEG, headers=up())
    assert c.get("/snapshot", params={"key": KEY}, headers={"If-None-Match": etag}).status_code == 200


def test_upload_needs_token(c):
    assert c.post("/ingest/snapshot", content=JPEG, headers=up(token="wrong")).status_code == 401
    assert c.post("/ingest/snapshot", content=JPEG, headers={"X-Camera-Id": "cam01"}).status_code == 401
    assert c.post("/ingest/health", json={}, headers=up(token="wrong")).status_code == 401


def test_view_needs_key(c):
    c.post("/ingest/snapshot", content=JPEG, headers=up())
    assert c.get("/snapshot").status_code == 401
    assert c.get("/snapshot", params={"key": "nope"}).status_code == 401
    assert c.get("/snapshot", headers={"X-View-Key": KEY}).status_code == 200
    assert c.get("/health").status_code == 401
    assert c.get("/").status_code == 401
    assert c.get("/", params={"key": KEY}).status_code == 200


def test_view_open_without_key_configured():
    c = make_client(view_key="")
    c.post("/ingest/snapshot", content=JPEG, headers=up())
    assert c.get("/snapshot").status_code == 200


def test_rejects_non_jpeg_and_oversize(c):
    assert c.post("/ingest/snapshot", content=b"hello", headers=up()).status_code == 415
    assert c.post("/ingest/snapshot", content=b"\xff\xd8" + b"\x00" * 20_000, headers=up()).status_code == 413


def test_rejects_bad_camera_id_and_too_many_cameras(c):
    assert c.post("/ingest/snapshot", content=JPEG, headers=up(cam="../etc")).status_code == 400
    assert c.post("/ingest/snapshot", content=JPEG, headers=up(cam="")).status_code == 400
    assert c.post("/ingest/snapshot", content=JPEG, headers=up(cam="a")).status_code == 204
    assert c.post("/ingest/snapshot", content=JPEG, headers=up(cam="b")).status_code == 204
    assert c.post("/ingest/snapshot", content=JPEG, headers=up(cam="c")).status_code == 429


def test_health_roundtrip_and_online_flag():
    c = make_client(online_timeout_s=1)
    assert c.get("/health", params={"key": KEY}).status_code == 404
    assert c.post("/ingest/health", json={"uptime_s": 5, "wifi_rssi_dbm": -60}, headers=up()).status_code == 204
    h = c.get("/health", params={"key": KEY}).json()
    assert h["online"] is True and h["health"]["wifi_rssi_dbm"] == -60 and h["health_received"] == 1
    time.sleep(1.2)
    assert c.get("/health", params={"key": KEY}).json()["online"] is False


def test_health_rejects_bad_json(c):
    assert c.post("/ingest/health", content=b"not json", headers=up()).status_code == 400
    assert c.post("/ingest/health", json=[1, 2], headers=up()).status_code == 400


def test_camera_selection(c):
    c.post("/ingest/snapshot", content=JPEG, headers=up(cam="a"))
    c.post("/ingest/snapshot", content=JPEG + b"b", headers=up(cam="b"))
    assert c.get("/snapshot", params={"key": KEY}).headers["x-camera-id"] == "b"   # most recent
    assert c.get("/snapshot", params={"key": KEY, "camera": "a"}).headers["x-camera-id"] == "a"
    assert c.get("/snapshot", params={"key": KEY, "camera": "zzz"}).status_code == 404
    assert {x["camera_id"] for x in c.get("/cameras", params={"key": KEY}).json()} == {"a", "b"}
