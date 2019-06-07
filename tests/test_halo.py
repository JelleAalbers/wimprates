import pandas as pd
from wimprates import j2000, standard_halo_model
import numericalunits as nu
import numpy as np


def test_shm_values():
    halo_model = standard_halo_model()
    assert np.abs(halo_model.v_0 /(nu.km/nu.s) - 220.)<1e-6
    assert np.abs(halo_model.v_esc /(nu.km/nu.s) - 544.)<1e-6

def test_j2000():
    assert j2000(2009, 1, 31.75) == 3318.25


def test_j2000_datetime():
    date = pd.datetime(year=2009, month=1, day=31, hour=18)
    assert j2000(date=date) == 3318.25
