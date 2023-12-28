"""Standard halo model: density, and velocity distribution
"""
from datetime import datetime
import numericalunits as nu
import numpy as np
import pandas as pd
import pickle
from scipy.special import erf
import scipy.interpolate as spi
import os


import wimprates as wr
export, __all__ = wr.exporter()


@export
class SolarReflectedDMModel:
    """
        class used to pass a solar reflected dm model to the rate computation
        must contain:

        :param mw -- dark matter mass with nu units assigned
        :param sigma_dme -- dm-electron cross section with nu units assigned
    """

    def __init__(self, mw, sigma_dme,):
        self.mw = mw
        self.sigma_dme = sigma_dme
        self.diff_flux_df = self.load_flux()

    def load_flux(self):
        """ Function to find differential flux for a particular (mass, xsec)
        point.

        Returns dataframe containing differential flux in units of
        [events/(km/s)/cm2/s] as a function of dm speed in units of [km/s].
        """
        fdata = os.path.dirname(os.path.realpath(__file__))
        
        tmp = pickle.load(open(f'{fdata}/data/srdm/consolidated_fluxes.pickle', 'rb'))

        xsec = self.sigma_dme/nu.cm**2
        mass = self.mw/(nu.GeV/nu.c0**2)

        this_name = f'mass{mass:.3e}_xsec{xsec:.3e}'
        df = tmp['flux_bag'][this_name]

        # Assigning units
        speed = df['Speed'].values * (nu.km/nu.s)
        diff_flux = df['Differential Flux'].values * (1/(nu.km/nu.s)/nu.cm**2/nu.s) 
        
        self.f_diff_flux = spi.interp1d(speed, diff_flux, kind='linear',
                           fill_value=0, bounds_error=False)

        self.v_min = min(speed)
        self.v_max = max(speed)

        return df


    def differential_flux(self, v):
        """
        Function to perform some checks on the velocity (actually also not sure
        what for) and return the differential flux in [1/(km/s)/cm2/s] at a
        particular dm speed, v, in [km/s]

        Differential flux computed from DAMASCUS-Sun is equivalent to the
        following for the non-reflected halo dark matter
        dphi/dv = rho_mw/mw * v * f_observed

        See eq 18 and surrounding text in 2102.12483v2 for details
        """

        # Zero if v > v_max
        try:
            len(v)
        except TypeError:
            # Scalar argument
            if v > self.v_max:
                #print('end0 0000000')
                return 0
            else:
                #print('end1 1111111')
                return self.f_diff_flux(v)
        else:
            # Array argument
            #print('end2 2222222')
            return self.f_diff_flux(v)

