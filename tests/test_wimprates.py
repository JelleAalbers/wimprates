"""Simple tests to see that the results of the computations do not change.

If you do update a computation, you'll have to change the hardcoded reference
values here.

"""
import numericalunits as nu
import numpy as np

import wimprates as wr
import unittest


class TestBenchmarks(unittest.TestCase):
    opts = dict(mw=50,
                sigma_nucleon=1e-45,
                )
    def test_elastic(self):
        ref = 30.39515403337126

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
                0.0005039148255734496)


    def test_spindependent(self):
        self.assertAlmostEqual(wr.rate_wimp_std(1, interaction='SD_n_central', **self.opts),
                0.00019944698779638946)


    def test_migdal(self):
        self.assertAlmostEqual(wr.rate_wimp_std(1, detection_mechanism='migdal', **self.opts),
                0.27459766238555017)


    def test_brems(self):
        self.assertAlmostEqual(wr.rate_wimp_std(1, detection_mechanism='bremsstrahlung', **self.opts),
                0.00017949417705393473)


    def test_dme(self):
        self.assertAlmostEqual(
            wr.rate_dme(100* nu.eV, 4, 'd',
                        mw=nu.GeV/nu.c0**2, sigma_dme=4e-44 * nu.cm**2)
                * nu.kg * nu.keV * nu.day,
        2.87553027086139e-06)

    def test_halo_scaling(self):
        #check that passing rho multiplies the rate correctly:
        ref = 30.39515403337113
        halo_model = wr.StandardHaloModel(rho_dm=0.3 * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)
        self.assertAlmostEqual(wr.rate_wimp_std(1,
                                                halo_model=halo_model,
                                                **self.opts,
                                                ), ref)


    def test_v_earth_old(self):
        """
        Check that v_0 = 220 gives 232 km/s for v_earth (old convention)
        https://arxiv.org/pdf/hep-ph/0504010.pdf
        """
        kms = nu.km/nu.s
        v_0_old = 220 * kms
        self.assertAlmostEqual(232,
                               wr.v_earth(t=None, v_0=v_0_old)/kms,
                               )

    def test_average_v_earth(self):
        """
        In v_earth a day is set that should return ~ the annual average, test
        this is true
        """
        kms = nu.km / nu.s
        v_0_default = 238 * kms
        # Lazy way of averaging over one year
        v_earth_average = np.mean(
            [wr.v_earth(t, v_0=v_0_default)
             for t in np.linspace(0, 365.25, int(1e4))]
        ) / kms
        self.assertAlmostEqual(
            v_earth_average, wr.v_earth(t=None, v_0=v_0_default) / kms,
            # places=1 means that we get the same results at the first decimal (fine for 500.0<?>)
            places=1
        )
