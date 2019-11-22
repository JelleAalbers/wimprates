import pandas as pd
from wimprates import j2000, StandardHaloModel, j2000_from_ymd
import numericalunits as nu
import numpy as np


def test_shm_values():
    halo_model = StandardHaloModel()
    assert np.abs(halo_model.v_0 /(nu.km/nu.s) - 220.)<1e-6
    assert np.abs(halo_model.v_esc /(nu.km/nu.s) - 544.)<1e-6


def test_j2000():
    assert j2000_from_ymd(2009, 1, 31.75) == 3318.25


def test_j2000_datetime():
    date = pd.datetime(year=2009, month=1, day=31, hour=18)
    assert j2000(date) == 3318.25


def test_j2000_ns_int():
    date = pd.datetime(year=2009, month=1, day=31, hour=18)
    value = pd.to_datetime(date).value
    assert isinstance(value, int)
    assert j2000(value) == 3318.25


def test_j2000_ns_array():
    # Generate some test cases and compare the two implementations
    years = np.arange(2008, 2020)
    months = np.arange(1, 12 + 1)
    days = np.arange(1, 12 + 1)
    hours = np.arange(1, 12 + 1)

    dates = np.zeros(12)
    j2000_ymd = np.zeros(12)
    for i in range(len(j2000_ymd)):
        print(years[i], months[i], days[i], hours[i])
        date = pd.datetime(year=years[i],
                           month=months[i],
                           day=days[i],
                           hour=hours[i])
        dates[i] = pd.to_datetime(date).value
        # Compute according to j2000_from_ymd
        j2000_ymd[i] = j2000_from_ymd(years[i],
                                      months[i],
                                      days[i] + hours[i] / 24)
    np.testing.assert_array_almost_equal(j2000(dates), j2000_ymd, decimal=6)
