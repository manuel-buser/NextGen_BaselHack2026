"""API contract test with FastAPI's TestClient (loads the model once; ~10 s)."""
import pytest
from fastapi.testclient import TestClient

from bikeslot import config

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model weights not downloaded")


@pytest.fixture(scope="module")
def client():
    from bikeslot.api import app

    with TestClient(app) as c:
        yield c


def test_health_and_seeded_golden_camera(client):
    h = client.get("/health").json()
    assert h["ok"] and h["detector"]["loaded"]
    cams = {c["camera_id"]: c for c in client.get("/cameras").json()}
    assert cams["cam01"]["layout_status"] == "confirmed" and cams["cam01"]["slot_count"] == 9


def test_post_frame_raw_and_multipart(client):
    data = (config.GOLDEN_DIR / "bikeparking_cam01_003.png").read_bytes()
    r = client.post("/cameras/cam01/frames", content=data, headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["slot_count"] == 9 and s["detections"]["bicycle"] == 6
    r = client.post("/cameras/cam01/frames", files={"file": ("f.png", data, "image/png")})
    assert r.status_code == 200
    s = r.json()
    assert s["free"] == 3 and s["occupied"] == 6          # 2 identical votes -> states decided
    assert [x["status"] for x in s["slots"]] == ["occupied", "free", "occupied", "free", "occupied",
                                                  "occupied", "occupied", "free", "occupied"]
    assert client.get("/cameras/cam01/preview.jpg").headers["content-type"] == "image/jpeg"
    site = client.get("/sites").json()[0]
    assert site["free"] == 3 and "cam01-S02" in site["free_slot_ids"]


def test_bad_body_and_unknown_camera(client):
    assert client.post("/cameras/camX/frames", content=b"not an image", headers={"Content-Type": "image/jpeg"}).status_code == 400
    assert client.get("/cameras/nope/status").status_code == 404


def test_layout_put_with_pixels_and_delete(client):
    data = (config.GOLDEN_DIR / "bikeparking_cam01_001.png").read_bytes()
    client.post("/cameras/cam03/frames", content=data, headers={"Content-Type": "image/jpeg"})
    st = client.get("/cameras/cam03/status").json()
    assert st["layout_status"] == "none" and "calibrate" in st["message"]
    gold = client.get("/cameras/cam01/layout?pixels=true").json()
    row = gold["rows"][0]
    r = client.put("/cameras/cam03/layout?pixels=true",
                   json={"rows": [row], "status": "confirmed", "frame_size": gold["frame_size"], "site_id": "site01"})
    assert r.status_code == 200, r.text
    lay = r.json()
    assert lay["status"] == "confirmed" and len(lay["slots"]) == 9 and lay["source"] == "admin"
    assert 0.0 <= lay["slots"][0]["polygon"][0][0] <= 1.0
    s = client.post("/cameras/cam03/frames", content=data, headers={"Content-Type": "image/jpeg"}).json()
    s = client.post("/cameras/cam03/frames", content=data, headers={"Content-Type": "image/jpeg"}).json()
    assert s["occupied"] == 6 and s["free"] == 3
    assert client.delete("/cameras/cam03/layout").json()["ok"]
    assert client.get("/cameras/cam03/layout").json()["status"] == "none"


def test_calibrate_without_vlm(client):
    data = (config.GOLDEN_DIR / "bikeparking_cam01_002.png").read_bytes()
    for _ in range(2):
        client.post("/cameras/cam04/frames", content=data, headers={"Content-Type": "image/jpeg"})
    r = client.post("/cameras/cam04/calibrate?use_vlm=false")
    assert r.status_code == 200, r.text
    lay = r.json()
    assert lay["status"] == "proposed" and lay["source"] == "auto"
    assert len(lay["slots"]) >= 6          # 6 bikes visible -> at least 6 clusters
    assert lay["calibration_hint"]["source"] == "local"
