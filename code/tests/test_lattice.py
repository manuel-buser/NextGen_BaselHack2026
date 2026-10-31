"""1-D projective lattice: hidden stands, perspective, end slots, no half-pitch lattices."""
from bikeslot import lattice


def _truth(a=120.0, b=150.0, c=0.05, n=9):
    """The 9-bay example: pitch shrinks from 107 px (bay 0->1) to 60 px (bay 7->8)."""
    return [(a * k + b) / (c * k + 1.0) for k in range(n)]


def test_recovers_occluded_stands_under_perspective():
    truth = _truth()
    seen = {0: 0.5, 1: -0.8, 3: 1.2, 5: -0.4, 8: 0.2}            # 4 of 9 stands hidden, +-1.2 px noise
    lat = lattice.fit([truth[k] + d for k, d in seen.items()])
    assert lat is not None
    lat = lat.normalized()
    assert lat.count == 9
    assert max(abs(lat.u(k) - truth[k]) for k in range(9)) < 2.0
    assert abs(lat.c - 0.05) < 0.01
    # the naive median pitch (75 px) would be off by > 40 px at the far end
    assert abs(truth[8] - (truth[0] + 8 * 75.0)) > 40


def test_equal_spacing_is_not_explained_by_a_half_pitch_lattice():
    lat = lattice.fit([100, 200, 300, 400, 500]).normalized()
    assert lat.count == 5 and abs(lat.c) < 1e-6 and abs(lat.pitch(0) - 100) < 1e-6


def test_doubled_stand_is_an_outlier_not_an_extra_slot():
    us = [100, 200, 300, 400, 455, 500, 600]                      # 455: second stirrup drawn in one bay
    lat = lattice.fit(us)
    assert lat.count == 6 and len(lat.strong_in) == 6
    assert 4 not in lat.strong_in


def test_two_stands_plus_bikes_fix_the_pitch():
    lat = lattice.fit([300, 500], weak=[195, 305, 405, 598]).normalized()
    assert lat.count == 2 or lat.count == 3
    assert abs(lat.pitch(0) - 100) < 5


def test_bikes_only_row_counts_empty_slots_between_bikes():
    bikes = [100, 200, 400, 500, 700]                            # slots 2 and 5 empty
    lat = lattice.fit([], weak=bikes).normalized()
    assert lat.count == 7 and abs(lat.pitch(0) - 100) < 5


def test_extend_end_slots_from_bikes():
    truth = _truth()
    lat = lattice.fit(truth[1:8]).normalized()                    # stands 0 and 8 hidden
    assert lat.count == 7
    # a bike at bay 8 (one slot beyond) and a bike leaning 0.4 slot beyond the first stand
    bikes = [truth[8] + 3.0, truth[1] - 0.4 * (truth[2] - truth[1])]
    ext = lattice.extend(lat, bikes)
    assert (ext.n_left, ext.n_right) == (0, 1) and ext.count == 8
    ext = lattice.extend(lat, [truth[0] - 2.0])
    assert (ext.n_left, ext.n_right) == (1, 0)


def test_phase_of_dividers_and_shift():
    lat = lattice.affine(100, 50, 0, 4)
    phi, conc = lattice.phase(lat, [125, 175, 225, 276])
    assert abs(abs(phi) - 0.5) < 0.02 and conc > 0.95
    sh = lat.shifted(2)
    assert sh.n_min == -2 and abs(sh.u(-2) - lat.u(0)) < 1e-9 and abs(sh.u(1) - lat.u(3)) < 1e-9
    b = lat.bounds(0.0)
    assert len(b) == 6 and abs(b[0] - 75) < 1e-9 and abs(b[-1] - 325) < 1e-9


def test_too_few_anchors():
    assert lattice.fit([100]) is None
    assert lattice.fit([]) is None
