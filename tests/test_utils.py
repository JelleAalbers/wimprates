import pandas as pd

from wimprates import j2000


def test_j2000():
    assert j2000(2009, 1, 31.75) == 3318.25

def test_j2000_datetime():
    date = pd.datetime('2009-1-31 18:00:00')
    assert j2000(date=date) == 3318.25
