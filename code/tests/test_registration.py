"""Registration: a moved camera's picture is warped back onto the reference frame its slot layout was drawn on."""
import cv2
import numpy as np
import pytest

from bikeslot import config, registration
from bikeslot.schemas import Layout
from bikeslot.slotcls import pipeline, smoothing
from bikeslot.slotcls.crops import slot_crops

DEMO_DIR = config.ROOT / "assets" / "pictures" / "DemoPictures"


@pytest.fixture(scope="module")
def scene():
    """The Memox rack at night, at the camera's resolution class (640 px wide)."""
    return cv2.resize(cv2.imread(str(DEMO_DIR / "memox_01.jpeg")), (640, 360), interpolation=cv2.INTER_AREA)


@pytest.fixture(scope="module")
def ref(scene):
    return registration.make_reference(scene)


def bumped(img, angle=6.0, scale=1.07, shift=(18.0, -12.0), tilt=0.00012):
    """The scene as a bumped camera sees it (rotated, zoomed, shifted, tilted) and the true reference->picture H."""
    h, w = img.shape[:2]
    H = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale), [tilt, 0.0, 1.0]])
    H[:2, 2] += shift
    return cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REPLICATE), H


def jpeg(img, quality=50):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)


def mapping_error(reg, H_true, ref_size, picture_scale=1.0):
    """Max distance (reference px) between grid points and their round trip reference -> picture -> reference."""
    w, h = ref_size
    xs, ys = np.meshgrid(np.linspace(0.2 * w, 0.8 * w, 7), np.linspace(0.25 * h, 0.75 * h, 5))
    pts = np.float32(np.stack([xs.ravel(), ys.ravel()], axis=1))[None]
    in_picture = cv2.perspectiveTransform(pts, H_true) * picture_scale
    back = cv2.perspectiveTransform(in_picture, reg.H)
    return float(np.linalg.norm(back - pts, axis=2).max())


def test_recovers_a_bumped_camera(scene, ref):
    img, H_true = bumped(scene)
    reg = registration.register(ref, jpeg(img))
    assert reg.ok, reg.reason
    assert reg.model == "homography" and reg.inliers >= 50
    assert reg.scale == pytest.approx(1 / 1.07, abs=0.03) and reg.rotation_deg == pytest.approx(6.0, abs=1.0)
    assert mapping_error(reg, H_true, ref.size) < 1.5


def test_registers_a_half_resolution_picture(scene, ref):
    """No fine detail is needed: a 320 px picture of the bumped camera lands as precisely (in reference px)."""
    img, H_true = bumped(scene, angle=-4.0, scale=0.95, shift=(-20.0, 10.0))
    small = jpeg(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA))
    reg = registration.register(ref, small)
    assert reg.ok, reg.reason
    assert reg.scale == pytest.approx(1 / 0.95, abs=0.04)      # evened out for the two resolutions
    assert mapping_error(reg, H_true, ref.size, picture_scale=0.5) < 2.5


def test_warp_puts_the_rack_back_in_place(scene, ref):
    img, _ = bumped(scene)
    aligned, seen = registration.warp(img, registration.register(ref, img), ref)
    assert aligned.shape == scene.shape and seen.shape == scene.shape[:2]
    inner = (slice(60, 300), slice(80, 560))
    before = np.abs(img[inner].astype(int) - scene[inner]).mean()
    after = np.abs(aligned[inner].astype(int) - scene[inner]).mean()
    assert after < 0.35 * before and seen[inner].all()


def test_rejects_other_scenes(scene, ref):
    noise = np.random.default_rng(0).integers(0, 255, scene.shape, dtype=np.uint8)
    assert not registration.register(ref, noise).ok
    flat = np.full(scene.shape, 128, np.uint8)
    reg = registration.register(ref, flat)
    assert not reg.ok and "features" in reg.reason
    on_its_side = cv2.resize(cv2.rotate(scene, cv2.ROTATE_90_CLOCKWISE), (640, 360))
    assert not registration.register(ref, on_its_side).ok
    with pytest.raises(ValueError):
        registration.make_reference(flat)


def test_reference_keeps_no_pixels_and_round_trips(ref, tmp_path):
    path = tmp_path / "cam.ref.npz"
    ref.save(path)
    with np.load(path) as d:
        assert set(d.files) == {"size", "points", "descriptors"}
    back = registration.Reference.load(path)
    assert back.size == ref.size and np.array_equal(back.points, ref.points)
    assert np.array_equal(back.descriptors, ref.descriptors)


class Recorder:
    """Stand-in classifier: slots in ``occupied`` get 0.9, the others 0.1; keeps the crops it was shown."""
    input_size = (96, 96)

    def __init__(self, occupied=()):
        self.occupied = set(occupied)
        self.crops = []

    def predict_proba(self, crops):
        self.crops = list(crops)
        return np.array([0.9 if i in self.occupied else 0.1 for i in range(len(crops))], np.float32)


@pytest.fixture()
def layout():
    lay = Layout.model_validate_json((DEMO_DIR / "memox_01.layout.json").read_text(encoding="utf-8"))
    lay.params.update(k_frames=3, min_frames=1)
    return lay


def test_pipeline_crops_the_warped_picture(scene, ref, layout):
    img, _ = bumped(scene)
    want = slot_crops(layout, scene)
    aligned = pipeline.analyze_frame(layout, img, "c", model=Recorder(), blur_persons=False, reference=ref)
    plain = pipeline.analyze_frame(layout, img, "c", model=Recorder(), blur_persons=False)
    assert aligned.status.registration["ok"] and plain.status.registration is None
    diff = lambda res: np.mean([np.abs(res.crops[k].astype(int) - want[k]).mean() for k in want])
    assert diff(aligned) < 0.4 * diff(plain)
    assert aligned.status.frame_size == [640, 360] and aligned.status.free == len(layout.slots)


def test_unaligned_pictures_cast_no_votes_then_reset(scene, ref, layout):
    trackers = smoothing.make_trackers(layout)
    pipeline.analyze_frame(layout, scene, "c", trackers, model=Recorder({0, 1}), blur_persons=False, reference=ref)
    noise = np.random.default_rng(1).integers(0, 255, scene.shape, dtype=np.uint8)
    once = pipeline.analyze_frame(layout, noise, "c", trackers, model=Recorder(), blur_persons=False,
                                  reference=ref, failures_before=0)
    assert not once.status.registration["ok"] and "not aligned" in once.status.message
    assert once.status.occupied == 2                         # one bad picture changes nothing
    last = pipeline.analyze_frame(layout, noise, "c", trackers, model=Recorder(), blur_persons=False,
                                  reference=ref, failures_before=config.REG_FAIL_RESET - 1)
    assert last.status.unknown == len(layout.slots)          # the camera has looked elsewhere for too long
