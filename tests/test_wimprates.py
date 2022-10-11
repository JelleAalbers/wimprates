"""Simple tests to see that the results of the computations do not change.

If you do update a computation, you'll have to change the hardcoded reference
values here.

"""
import numericalunits as nu
import numpy as np

import wimprates as wr
import unittest


class TestBenchmarks(unittest.TestCase):
    opts = dict(mw=50, sigma_nucleon=1e-45)
    def test_elastic(self):
        ref = 33.052499179451

        self.assertAlmostEqual(wr.rate_wimp_std(1, **self.opts), ref)

        # Test numericalunits.reset_units() does not affect results
        nu.reset_units(123)
        self.assertAlmostEqual(wr.rate_wimp_std(1, **self.opts), ref)

        # Test vectorized call
        energies = np.linspace(0.01, 40, 100)
        dr = wr.rate_wimp_std(energies, **self.opts)
        self.assertEqual(dr[0], wr.rate_wimp_std(0.01, **self.opts))


    def test_lightmediator(self):
        self.assertAlmostEqual(wr.rate_wimp_std(1, m_med=1e-3, **self.opts),
                0.0005479703883222002)


    def test_spindependent(self):
        self.assertAlmostEqual(wr.rate_wimp_std(1, interaction='SD_n_central', **self.opts),
                0.00021688396095135553)


    def test_migdal(self):
        self.assertAlmostEqual(wr.rate_wimp_std(1, detection_mechanism='migdal', **self.opts),
                0.2621719215956991)


    def test_brems(self):
        self.assertAlmostEqual(wr.rate_wimp_std(1, detection_mechanism='bremsstrahlung', **self.opts),
                0.00017137557193256555)


    def test_dme(self):
        self.assertAlmostEqual(
            wr.rate_dme(100* nu.eV, 4, 'd',
                        mw=nu.GeV/nu.c0**2, sigma_dme=4e-44 * nu.cm**2)
                * nu.kg * nu.keV * nu.day,
        2.232912243660405e-06)

    def test_halo_scaling(self):
        #check that passing rho multiplies the rate correctly:
        ref = 33.052499179450834
        halo_model = wr.StandardHaloModel(rho_dm=0.3 * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)
        self.assertAlmostEqual(wr.rate_wimp_std(1, halo_model=halo_model, **self.opts), ref)

