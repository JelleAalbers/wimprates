"""Standard halo model: density, and velocity distribution
Also can specify f(v) distribution function
"""
from datetime import datetime
import numericalunits as nu
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.integrate import dblquad
from scipy.interpolate import CubicSpline
from scipy.fft import rfft
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
    t1 = j2000_from_ymd(2000, 3, 21)
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

    # Normalization constant,see Lewin&Smith appendix 1a (now corrected)
    _w = v_esc/v_0
    k = erf(_w) - 2/np.pi**0.5 * _w * np.exp(-_w**2)  # unitless

    # Maximum cos(angle) for this velocity, otherwise v0
    xmax = np.minimum(1,
                      (v_esc**2 - v_earth_t**2 - v**2)
                      / (2 * v_earth_t * v))
    # unitless

    y = (v / (np.pi**0.5 *k* v_0 * v_earth_t)
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
class dfHalo:
    def __init__(self,v_0,v_esc):
        self.v_0=v_0
        self.v_esc=v_esc
        _w=v_esc/v_0
        k = erf(_w) - 2/np.pi**0.5 * _w * np.exp(-_w**2)
        self.fac=1/(np.pi*np.sqrt(np.pi)*k*v_0**3)
    def value(self,velg):
        vmag2=(velg[0]**2+velg[1]**2+velg[2]**2)
        if(vmag2>self.v_esc**2):
            return 0
        return self.fac*np.exp(-vmag2/(self.v_0**2))
#csth=cos(theta)
def f(phi,csth,v,vel_earth_t,df,e1,e2,e3):
    sinth2=1-csth*csth
    sinth=0
    if(sinth2>0):
        sinth=np.sqrt(sinth2)
    v1=v*csth
    v2=v*sinth*np.cos(phi)
    v3=v*sinth*np.sin(phi)
    vR=v1*e1[0]+v2*e2[0]+v3*e3[0]+vel_earth_t[0]
    vp=v1*e1[1]+v2*e2[1]+v3*e3[1]+vel_earth_t[1]
    vz=v1*e1[2]+v2*e2[2]+v3*e3[2]+vel_earth_t[2]
    #jacobian of velocity space
    jac=v**2
    return jac*df(np.array([vR,vp,vz]))
@export
def observed_speed_distfromdf(v,t=59.37,distF=None,v_0=None,v_esc=None,epsrel=1e-2):
    v_0 = _HALO_DEFAULTS['v_0'] * nu.km/nu.s if v_0 is None else v_0
    v_esc = _HALO_DEFAULTS['v_esc'] * nu.km/nu.s if v_esc is None else v_esc
    if distF is None:
        v_0n = _HALO_DEFAULTS['v_0']*nu.km/nu.s
        v_escn = _HALO_DEFAULTS['v_esc']*nu.km/nu.s
        distF=dfHalo(v_0n,v_escn)
    vel_earth_t = earth_velocity(t,v_0)
    vel_earth_t[0]*=-1
    vel_earth_t[1]*=-1
    vearth=np.sqrt(np.sum(vel_earth_t**2))
    e1=vel_earth_t/vearth
    e2=np.array([vel_earth_t[1],-vel_earth_t[0],0])
    e2/=np.sqrt(np.sum(e2**2))
    e3=np.array([vel_earth_t[2]*vel_earth_t[0],vel_earth_t[1]*vel_earth_t[2],-vel_earth_t[1]**2-vel_earth_t[0]**2])
    e3/=np.sqrt(np.sum(e3**2))
    try:
        len(v)
    except TypeError:
        if(v==0):
            return 0
        csthmax=(v_esc**2-vearth**2-v**2)/(2*v*vearth)
        if(csthmax<-1):
            return 0
        if(csthmax>1):
            csthmax=1
        return dblquad(f,-1,csthmax,0,2*np.pi,args=(v,vel_earth_t,distF,e1,e2,e3),epsrel=epsrel,epsabs=0)[0]
    else:
        result=np.zeros(len(v))
        for i in range(len(v)):
            if(v[i]==0):
                result[i]=0
                continue
            csthmax=(v_esc**2-vearth**2-v[i]**2)/(2*v[i]*vearth)
            if(csthmax<-1):
                result[i]=0
                continue
            if(csthmax>1):
                csthmax=1
            result[i]=dblquad(f,-1,csthmax,0,2*np.pi,args=(v[i],vel_earth_t,distF,e1,e2,e3),epsrel=epsrel,epsabs=0)[0]
        return result
    
    

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
        :param v_0: Local standard of rest velocity
    """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None):
        self.v_0 = _HALO_DEFAULTS['v_0'] * nu.km/nu.s if v_0 is None else v_0
        self.v_esc = _HALO_DEFAULTS['v_esc'] * nu.km/nu.s if v_esc is None else v_esc
        self.rho_dm = _HALO_DEFAULTS['rho_dm'] * nu.GeV/nu.c0**2 / nu.cm**3 if rho_dm is None else rho_dm

    def velocity_dist(self, v, t):
        # in units of per velocity,
        # v is in units of velocity
        return observed_speed_dist(v, t, v_0=self.v_0, v_esc=self.v_esc)
'''
Writes into a file the marginilised distribution function of speed from a given distribution function.
This is done by taking time fourier transform. File is used by HaloModelInterpolatedfromFile.
This way you do not need to calculate the distribution fucntion every time you run it.
distF is the distribution function in the galactic frame f(vR,vz,vphi), normalised so the integral over velocity is one.
v_0 is the circular speed at the Sun's location and v_esc is the escape velocity at the sun.
N is the number of points to evaluate, epsrel is the relative accuracy of the double itnegration
Nf is the number of evaluations in time (so that Nf//2 is maximum fourier coefficient).
File is written speed v in km/s, and then the fourier coefficients in units of s/km. Negative m signifies sine series, and non negative cosine series.
'''
@export
def writeFourcoefs(distF,Filename="FourCoefs.txt",v_0=None,v_esc=None,N=100,epsrel=1e-2,Nf=5):
    v_01=_HALO_DEFAULTS['v_0'] * nu.km/nu.s if v_0 is None else v_0*nu.km/nu.s
    v_esc1=_HALO_DEFAULTS['v_esc']*nu.km/nu.s if v_esc is None else v_esc*nu.km/nu.s
    speedpec=np.sum((np.array(_HALO_DEFAULTS['v_pec']))**2)**0.5 * nu.km/nu.s
    vmax=v_esc1+v_01+speedpec
    vvec=np.linspace(0,vmax,N)
    T=365.25
    Nfc=Nf//2+1
    fv=np.zeros((len(vvec),Nf))
    for j in range(Nf):
        t1=j/Nf*T
        fv[:,j]=observed_speed_distfromdf(vvec,t=t1,distF=distF,v_0=v_01,v_esc=v_esc1,epsrel=epsrel)*nu.km/nu.s
    a=np.zeros((len(vvec),2*Nfc-1))
    TxtHead="Fourier coefficients with time of DM speed distribution\n v(km/s) m"
    for j in range(2*Nfc-1):
        if j<Nfc:
            TxtHead+=" "+str(-(Nfc-j-1))
        else:
            TxtHead+=" "+str(j-Nfc+1)
    for i in range(len(vvec)):
        complA=rfft(fv[i,:])
        a[i,Nfc-1]=complA[0].real/Nf
        for j in range(Nfc-1):
            a[i,j+Nfc]=1/Nf*complA[j+1].real
            a[i,Nfc-j-2]=1/Nf*complA[j+1].imag
    np.savetxt(Filename,np.concatenate((np.array([vvec*nu.s/nu.km]).T,a),axis=1),header=TxtHead)
@export
class HaloModelInterpolated:
    """
        class which, from a given DF, samples the speed of the dark matter halo relative to Earth
        distF is distribution function f(vR,vz,vphi) normalised so int f(v)d^3v=1.
        rho_dm is DM density
        v_0 is circular speed at Earth
        v_esc is escape speed
        N is number of points to evaluate the speed at for interpolating
        Nf is number of fourier coefficients
        epsrel is relative accuracy of integral
    """
    def __init__(self,distF,rho_dm=None,v_0=None,v_esc=None,N=100,Nf=5,epsrel=1e-2):
        self.distF=distF
        self.v_0= _HALO_DEFAULTS['v_0'] * nu.km/nu.s if v_0 is None else v_0
        self.v_esc=_HALO_DEFAULTS['v_esc']*nu.km/nu.s if v_esc is None else v_esc
        self.rho_dm=_HALO_DEFAULTS['rho_dm'] * nu.GeV/nu.c0**2 / nu.cm**3 if rho_dm is None else rho_dm
        speedpec=np.sum((np.array(_HALO_DEFAULTS['v_pec']))**2)**0.5 * nu.km/nu.s
        vmax=self.v_esc+self.v_0+speedpec
        vvec=np.linspace(0,vmax,N)
        self.vvec=vvec
        T=365.25
        Nfc=Nf//2+1
        fv=np.zeros((Nf,len(vvec)))*(1+1j)
        for j in range(Nf):
            t1=j/Nf*T
            fv[j,:]=observed_speed_distfromdf(vvec,t=t1,distF=distF,v_0=v_0,v_esc=v_esc,epsrel=epsrel)
        a=np.zeros((len(vvec),Nfc))*(1+1j)
        for i in range(len(vvec)):
            complA=rfft(fv[:,i].real)
            a[i,:]=1/Nf*complA
        self.a=a
        self.interpola=CubicSpline(vvec,a)
        self.Nt=Nfc
        self.T=T
    def velocity_dist(self,v,t):
        t=59.37 if t is None else t
        th=2*np.pi/self.T*t
        vmax=v_max(t,self.v_esc,self.v_0)
        try:
            len(v)
        except TypeError:
            if(v>=vmax):
                return 0
            Acompl=self.interpola(v)
            res=Acompl[0].real
            for i in range(1,self.Nt-1):
                res+=2*(np.cos(i*th)*Acompl[i].real-np.sin(i*th)*Acompl[i].imag)
            return res
        else:
            result=np.zeros(len(v))
            for i in range(len(v)):
                if(v[i]>=vmax):
                    result[i]=0
                else:
                    Acompl=self.interpola(v[i])
                    result[i]=Acompl[0].real
                    for j in range(1,self.Nt):
                        result[i]+=2*(np.cos(j*th)*Acompl[j].real-np.sin(j*th)*Acompl[j].imag)
            return result
    def write_to_file(self,Filename="FourCoefs.txt"):
        TxtHead="Fourier coefficients with time of DM speed distribution\n v(km/s) m"
        for j in range(2*self.Nt-1):
            if j<self.Nt:
                TxtHead+=" "+str(-(self.Nt-j-1))
            else:
                TxtHead+=" "+str(j-self.Nt+1)
        a=np.zeros((len(self.vvec),2*self.Nt-1))
        for i in range(len(self.vvec)):
            a[i,self.Nt-1]=self.a[i,0].real*nu.km/nu.s
            for j in range(self.Nt-1):
                a[i,j+self.Nt]=self.a[i,j+1].real*nu.km/nu.s
                a[i,self.Nt-j-2]=self.a[i,j+1].imag*nu.km/nu.s
        np.savetxt(Filename,np.concatenate((np.array([self.vvec*nu.s/nu.km]).T,a),axis=1),header=TxtHead)
@export
class HaloModelInterpolatedFromFile:
    """
        class which, from a given file, samples the speed of the dark matter halo relative to Earth
        rho_dm is DM density
        v_0 is circular speed at Earth
        v_esc is escape speed
    """
    def __init__(self,Filename="FourCoefs.txt",rho_dm=None,v_0=None,v_esc=None):
        self.v_0= _HALO_DEFAULTS['v_0'] * nu.km/nu.s if v_0 is None else v_0
        self.v_esc=_HALO_DEFAULTS['v_esc']*nu.km/nu.s if v_esc is None else v_esc
        self.rho_dm=_HALO_DEFAULTS['rho_dm'] * nu.GeV/nu.c0**2 / nu.cm**3 if rho_dm is None else rho_dm
        T=365.25
        self.T=T
        va=np.loadtxt(Filename)
        vvec=va[:,0].real*nu.km/nu.s
        Nfc=(np.shape(va)[1]+1)//2
        a=np.ones((len(vvec),Nfc))*(1+1j)
        a[:,0]=va[:,Nfc]*nu.s/nu.km
        for j in range(Nfc-1):
            a[:,j+1]=(va[:,j+Nfc+1]+va[:,Nfc-j-1]*1j)*nu.s/nu.km
        self.interpola=CubicSpline(vvec,a)
        self.Nt=Nfc
    def velocity_dist(self,v,t):
        t=59.37 if t is None else t
        th=2*np.pi/self.T*t
        vmax=v_max(t,self.v_esc,self.v_0)
        try:
            len(v)
        except TypeError:
            if(v>=vmax):
                return 0
            Acompl=self.interpola(v)
            res=Acompl[0].real
            for i in range(1,self.Nt-1):
                res+=2*(np.cos(i*th)*Acompl[i].real-np.sin(i*th)*Acompl[i].imag)
            if(res<0):
                res=0
            return res
        else:
            result=np.zeros(len(v))
            for i in range(len(v)):
                if(v[i]>=vmax):
                    result[i]=0
                else:
                    Acompl=self.interpola(v[i])
                    result[i]=Acompl[0].real
                    for j in range(1,self.Nt):
                        result[i]+=2*(np.cos(j*th)*Acompl[j].real-np.sin(j*th)*Acompl[j].imag)
                    if result[i]<0:
                        result[i]=0
            return result
