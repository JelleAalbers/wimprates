"""Simple tests to see that the results of the computations do not change.

If you do update a computation, you'll have to change the hardcoded reference
values here.

"""
import numpy as np

import wimprates as wr


opts = dict(mw=50, sigma_nucleon=1e-45)


def isclose(x, y):
    assert np.abs(x - y)/x < 1e-5


def test_elastic():
    isclose(wr.rate_wimp_std(1, **opts),
            33.19098343826968)

    # Test vectorized call
    energies = np.linspace(0.01, 40, 100)
    dr = wr.rate_wimp_std(energies, **opts)
    assert dr[0] == wr.rate_wimp_std(0.01, **opts)


def test_lightmediator():
    isclose(wr.rate_wimp_std(1, m_med=1e-3, **opts),
            0.0005502663384403058)


def test_spindependent():
    isclose(wr.rate_wimp_std(1, interaction='SD_n_central', **opts),
            0.00021779266679860948)


def test_migdal():
    isclose(wr.rate_wimp_std(1, detection_mechanism='migdal', **opts),
            0.2610240963512907)


def test_brems():
    isclose(wr.rate_wimp_std(1, detection_mechanism='bremsstrahlung', **opts),
            0.00017062652972332665)


