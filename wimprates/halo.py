"""Standard halo model: density, and velocity distribution
"""
import numericalunits as nu
import numpy as np
import pandas as pd
from scipy.special import erf


import wimprates as wr
export, __all__ = wr.exporter()


@export
def rho_dm():
    """Local dark matter density"""
    return 0.3 * nu.GeV/nu.c0**2 / nu.cm**3


@export
def v_0():
    """Most common velocity of WIMPs in the halo,
    relative to galactic center (asymptotic)
    """
    return 220 * nu.km/nu.s


@export
def v_esc():
    """Galactic escape velocity"""
    return 544 * nu.km/nu.s


# J2000.0 epoch conversion (converts datetime to days since epoch)
# Zero of this convention is defined as 12h Terrestrial time on 1 January 2000
# This is similar to UTC or GMT with negligible error (~1 minute).
# See http://arxiv.org/abs/1312.1355 Appendix A for more details
# Test case for 6pm GMT 31st January 2009
#  j2000(2009, 1, 31.75) = 3318.25
#  j2000(date=pd.to_datetime('2009-1-31 18:00:00') = 3318.25
@export
def j2000(year=None, month=None, day_of_month=None, date=None):
    """Convert calendar date in year, month (starting at 1) and
    the (possibly fractional) day of the month relative to midnight UT.
    Either pass year, month and day_of_month or pass pandas datetime object
    via date argument.
    Returns the fractional number of days since J2000.0 epoch.
    """
    if date is not None:
        year = date.year
        month = date.month

        start_of_month = pd.datetime(year, month, 1)
        day_of_month = (date - start_of_month) / pd.Timedelta(1, 'D') + 1

    assert month > 0
    assert month < 13

    y = year if month > 2 else year - 1
    m = month if month > 2 else month + 12

    return (np.floor(365.25 * y)
            + np.floor(30.61 * (m + 1))
            + day_of_month - 730563.5)


@export
def earth_velocity(t):
    """Returns 3d velocity of earth, in the galactic rest frame,
    in galactic coordinates.
    :param t: J2000.0 timestamp

    Values and formula from https://arxiv.org/abs/1209.3339
    Assumes earth circular orbit.
    """
    # e_1 and e_2 are the directions of earth's velocity at t1
    # and t1 + 0.25 year.
    e_1 = np.array([0.9931, 0.1170, -0.01032])
    e_2 = np.array([-0.0670, 0.4927, -0.8676])
    # t1 is the time of the vernal equinox, March 21. Does it matter what
    # year? Precession of equinox takes 25800 years so small effect.
    t1 = j2000(2000, 3, 21)
    # Angular frequency
    omega = 2 * np.pi / 365.25
    phi = omega * (t - t1)

    # Mean orbital velocity of the Earth (Lewin & Smith appendix B)
    v_orbit = 29.79 * nu.km / nu.s

    v_earth_sun = v_orbit * (e_1 * np.cos(phi) + e_2 * np.sin(phi))

    # Velocity of Local Standard of Rest
    v_lsr = np.array([0, 220, 0]) * nu.km/nu.s
    # Solar peculiar velocity
    v_pec = np.array([11, 12, 7]) * nu.km/nu.s

    return v_lsr + v_pec + v_earth_sun


@export
def v_earth(t=None):
    """Return speed of earth relative to galactic rest frame
    :param t: J2000 timestamp or None
    """
    if t is None:
        # Velocity of earth/sun relative to gal. center
        # (eccentric orbit, so not equal to v_0)
        return 232 * nu.km / nu.s
    else:
        return np.sum(earth_velocity(t) ** 2) ** 0.5


@export
def v_max(t=None):
    """Return maximum observable dark matter velocity on Earth."""
    if t is None:
        return v_esc() + v_earth(t)
    else:
        return v_esc() + np.sum(earth_velocity(t) ** 2) ** 0.5


@export
def observed_speed_dist(v, t=None):
    """Observed distribution of dark matter particle speeds on earth
    under the standard halo model.

    See my thesis for derivation ;-)
    If you find a paper where this formula is written out explicitly, please
    let me know. I spent a lot of time looking for this in vain.

    Optionally supply J2000.0 time t to take into account Earth's orbital
    velocity.
    """
    v_earth_t = v_earth(t)

    # Normalization constant, see Lewin&Smith appendix 1a
    _w = v_esc()/v_0()
    k = erf(_w) - 2/np.pi**0.5 * _w * np.exp(-_w**2)

    # Maximum cos(angle) for this velocity, otherwise v0
    xmax = np.minimum(1,
                      (v_esc()**2 - v_earth_t**2 - v**2)
                      / (2 * v_earth_t * v))

    y = (k * v / (np.pi**0.5 * v_0() * v_earth_t)
         * (np.exp(-((v-v_earth_t)/v_0())**2)
            - np.exp(-1/v_0()**2 * (v**2 + v_earth_t**2
                                    + 2 * v * v_earth_t * xmax))))

    # Zero if v > v_max
    try:
        len(v)
    except TypeError:
        # Scalar argument
        if v > v_max(t):
            return 0
        else:
            return y
    else:
        # Array argument
        y[v > v_max(t)] = 0
        return y
