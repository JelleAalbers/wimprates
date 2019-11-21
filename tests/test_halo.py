import pandas as pd
from wimprates import j2000, StandardHaloModel
import numericalunits as nu
import numpy as np


def test_shm_values():
    halo_model = StandardHaloModel()
    assert np.abs(halo_model.v_0 /(nu.km/nu.s) - 220.)<1e-6
    assert np.abs(halo_model.v_esc /(nu.km/nu.s) - 544.)<1e-6

def test_j2000():
    assert j2000(2009, 1, 31.75) == 3318.25


def test_j2000_datetime():
    date = pd.datetime(year=2009, month=1, day=31, hour=18)
    assert j2000(date=date) == 3318.25

def test_j2000_ns_int():
    date = pd.datetime(year=2009, month=1, day=31, hour=18)
    assert j2000(date=pd.to_datetime(date).value) == 3318.25

def test_j2000_ns_array():
    date = pd.datetime(year=2009, month=1, day=31, hour=18)
    dates = np.array([pd.to_datetime(date).value] * 3)
    np.testing.assert_array_equal(j2000(date=dates), np.array([3318.25] * 3))
