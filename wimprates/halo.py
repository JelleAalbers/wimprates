"""Standard halo model: density, and velocity distribution
"""
from datetime import datetime
import numericalunits as nu
import numpy as np
import pandas as pd
from scipy.special import erf


import wimprates as wr
export, __all__ = wr.exporter()


# J2000.0 epoch conversion (converts datetime to days since epoch)
# Zero of this convention is defined as 12h Terrestrial time on 1 January 2000
# This is similar to UTC or GMT with negligible error (~1 minute).
# See http://arxiv.org/abs/1312.1355 Appendix A for more details
# Test case for 6pm GMT 31st January 2009
#  j2000(2009, 1, 31.75) = 3318.25
#  j2000(date=pd.to_datetime('2009-1-31 18:00:00') = 3318.25
@export
def j2000(date):
    """Returns the fractional number of days since J2000.0 epoch.
    Either pass:
      * An integer or array of integers (date argument), ns since unix epoch
      * datetime.datetime object
      * pd.Timestamp object
    Day of month starts at 1.
    """
    zero = pd.to_datetime('2000-01-01T12:00')
    nanoseconds_per_day = 1e9 * 3600 * 24
    if isinstance(date, datetime):
        # pd.datetime refers to datetime.datetime
        # make it into a pd.Timestamp
        # Timestamp.value gives timestamp in ns
        date = pd.to_datetime(date).value
    elif isinstance(date, pd.Timestamp):
        date = date.value
    return (date - zero.value) / nanoseconds_per_day


@export
def j2000_from_ymd(year, month, day_of_month):
    """"Returns the fractional number of days since J2000.0 epoch.
    :param year: Year
    :param month: Month (January = 1)
    :param day: Day of month (starting from 1), fractional days are
    relative to midnight UT.
    """
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
    t1 = j2000_from_ymd(2000, 3, 21)
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
def v_max(t=None, v_esc=None):
    """Return maximum observable dark matter velocity on Earth."""
    v_esc = 544 * nu.km/nu.s if v_esc is None else v_esc  # default
    # args do not change value when you do a
    # reset_unit so this is necessary to avoid errors
    if t is None:
        return v_esc + v_earth(t)
    else:
        return v_esc + np.sum(earth_velocity(t) ** 2) ** 0.5


@export
def observed_speed_dist(v, t=None, v_0=None, v_esc=None):
    """Observed distribution of dark matter particle speeds on earth
    under the standard halo model.

    See my thesis for derivation ;-)
    If you find a paper where this formula is written out explicitly, please
    let me know. I spent a lot of time looking for this in vain.

    Optionally supply J2000.0 time t to take into account Earth's orbital
    velocity.

    Further inputs: scale velocity v_0 and escape velocity v_esc_value
    """
    v_0 = 220 * nu.km/nu.s if v_0 is None else v_0
    v_esc = 544 * nu.km/nu.s if v_esc is None else v_esc
    v_earth_t = v_earth(t)

    # Normalization constant, see Lewin&Smith appendix 1a
    _w = v_esc/v_0
    k = erf(_w) - 2/np.pi**0.5 * _w * np.exp(-_w**2)  # unitless

    # Maximum cos(angle) for this velocity, otherwise v0
    xmax = np.minimum(1,
                      (v_esc**2 - v_earth_t**2 - v**2)
                      / (2 * v_earth_t * v))
    # unitless

    y = (k * v / (np.pi**0.5 * v_0 * v_earth_t)
         * (np.exp(-((v-v_earth_t)/v_0)**2)
         - np.exp(-1/v_0**2 * (v**2 + v_earth_t**2
                  + 2 * v * v_earth_t * xmax))))
    # units / (velocity)

    # Zero if v > v_max
    try:
        len(v)
    except TypeError:
        # Scalar argument
        if v > v_max(t, v_esc):
            return 0
        else:
            return y
    else:
        # Array argument
        y[v > v_max(t, v_esc)] = 0
        return y


@export
class StandardHaloModel:
    """
        class used to pass a halo model to the rate computation
        must contain:
        :param v_esc -- escape velocity
        :function velocity_dist -- function taking v,t
        giving normalised velocity distribution in earth rest-frame.
        :param rho_dm -- density in mass/volume of dark matter at the Earth
        The standard halo model also allows variation of v_0
        :param v_0
    """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None):
        self.v_0 = 220 * nu.km/nu.s if v_0 is None else v_0
        self.v_esc = 544 * nu.km/nu.s if v_esc is None else v_esc
        self.rho_dm = 0.3 * nu.GeV/nu.c0**2 / nu.cm**3 if rho_dm is None else rho_dm

    def velocity_dist(self, v, t):
        # in units of per velocity,
        # v is in units of velocity
        return observed_speed_dist(v, t, self.v_0, self.v_esc)

