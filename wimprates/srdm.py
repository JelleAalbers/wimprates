"""Standard halo model: density, and velocity distribution
"""
from datetime import datetime
import numericalunits as nu
import numpy as np
import pandas as pd
from scipy.special import erf


import wimprates as wr
export, __all__ = wr.exporter()


# See https://arxiv.org/abs/2105.00599 and references therein
_HALO_DEFAULTS = dict(
    rho_dm = 0.3, # GeV / c2 / cm3
    v_esc = 544,  # km/s
    v_orbit = 29.8,  # km/s
    v_pec = (11.1, 12.2, 7.3),  # km/s
    v_0 = 238,  # km/s
)


@export
def earth_velocity(t, v_0 = None):
    """Returns 3d velocity of earth, in the galactic rest frame,
    in galactic coordinates.
    :param t: J2000.0 timestamp
    :param v_0: Local standard of rest velocity

    Values and formula from https://arxiv.org/abs/1209.3339
    Assumes earth circular orbit.
    """
    if v_0 is None :
        v_0 = _HALO_DEFAULTS['v_0'] * nu.km/nu.s

    # e_1 and e_2 are the directions of earth's velocity at t1
    # and t1 + 0.25 year.
    e_1 = np.array([0.9931, 0.1170, -0.01032])
    e_2 = np.array([-0.0670, 0.4927, -0.8676])
    # t1 is the time of the vernal equinox, March 21. Does it matter what
    # year? Precession of equinox takes 25800 years so small effect.
    t1 = wr.j2000_from_ymd(2000, 3, 21)
    # Angular frequency
    omega = 2 * np.pi / 365.25
    phi = omega * (t - t1)

    # Mean orbital velocity of the Earth (Lewin & Smith appendix B)
    v_orbit = _HALO_DEFAULTS['v_orbit'] * nu.km / nu.s

    v_earth_sun = v_orbit * (e_1 * np.cos(phi) + e_2 * np.sin(phi))

    # Velocity of Local Standard of Rest
    v_lsr = np.array([0, v_0, 0])
    # Solar peculiar velocity
    v_pec = np.array(_HALO_DEFAULTS['v_pec']) * nu.km/nu.s

    return v_lsr + v_pec + v_earth_sun


@export
def v_earth(t=None, v_0=None):
    """Return speed of earth relative to galactic rest frame
    Velocity of earth/sun relative to gal. center (eccentric orbit, so not
    equal to v_0).

    :param t: J2000 timestamp or None
    :param v_0: Local standard of rest velocity
    """
    if t is None:
        # This day (Feb 29 2000) gives ~ the annual average speed
        t = 59.37
    return np.sum(earth_velocity(t, v_0=v_0) ** 2) ** 0.5


@export
def v_max(t=None, v_esc=None, v_0=None):
    """Return maximum observable dark matter velocity on Earth."""
    # defaults
    v_esc = _HALO_DEFAULTS['v_esc'] * nu.km/nu.s if v_esc is None else v_esc
    v_0 = _HALO_DEFAULTS['v_0'] * nu.km / nu.s if v_0 is None else v_0
    # args do not change value when you do a
    # reset_unit so this is necessary to avoid errors
    if t is None:
        return v_esc + v_earth(t, v_0=v_0)
    else:
        return v_esc + np.sum(earth_velocity(t, v_0=v_0) ** 2) ** 0.5


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
    v_0 = _HALO_DEFAULTS['v_0'] * nu.km/nu.s if v_0 is None else v_0
    v_esc = _HALO_DEFAULTS['v_esc'] * nu.km/nu.s if v_esc is None else v_esc
    v_earth_t = v_earth(t, v_0=v_0)

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
        if v > v_max(t, v_esc, v_0=v_0):
            return 0
        else:
            return y
    else:
        # Array argument
        y[v > v_max(t, v_esc, v_0=v_0)] = 0
        return y


@export
class SolarReflectedDMModel:
    """
        class used to pass a solar reflected dm model to the rate computation
        must contain:
        :param v_esc -- escape velocity
        :function velocity_dist -- function taking v,t
        giving normalised velocity distribution in earth rest-frame.
        :param rho_dm -- density in mass/volume of dark matter at the Earth
        The standard halo model also allows variation of v_0
        :param v_0: Local standard of rest velocity
    """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None):
        # CHANGE THESE TO VALUES USED IN DAMASCUS-SUN!
        self.v_0 = _HALO_DEFAULTS['v_0'] * nu.km/nu.s if v_0 is None else v_0
        self.v_esc = _HALO_DEFAULTS['v_esc'] * nu.km/nu.s if v_esc is None else v_esc
        self.rho_dm = _HALO_DEFAULTS['rho_dm'] * nu.GeV/nu.c0**2 / nu.cm**3 if rho_dm is None else rho_dm

    def differential_flux(self, v, t):
        # in units of per velocity,
        # v is in units of velocity
        # Differential Flux from DAMASCUS-Sun
        return observed_speed_dist(v, t, v_0=self.v_0, v_esc=self.v_esc)

