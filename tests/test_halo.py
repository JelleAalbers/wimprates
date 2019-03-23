import pandas as pd
from wimprates import j2000


def test_j2000():
    assert j2000(2009, 1, 31.75) == 3318.25


def test_j2000_datetime():
    date = pd.datetime(year=2009, month=1, day=31, hour=18)
    assert j2000(date=date) == 3318.25
