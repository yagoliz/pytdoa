from dataclasses import dataclass

import jax.numpy as jnp

# Speed of light
SPEED_OF_LIGHT = 299792458

# Geodetic class
@dataclass
class geoC:
    # Basic Earth model constants
    WGS84_A: float = 6378137.0
    WGS84_F: float = 1.0 / 298.257223563
    WGS84_B: float = WGS84_A * (1 - WGS84_F)
    WGS84_ECC_SQ: float = 1 - WGS84_B * WGS84_B / (WGS84_A * WGS84_A)
    WGS84_ECC: float = jnp.sqrt(WGS84_ECC_SQ)

    # Average radius for a spherical Earth
    SPHERICAL_R: float = 6371e3
    CIRCUMFERENCE: float = 2 * jnp.pi * SPHERICAL_R

    # Some derived values
    wgs84_ep: float = jnp.sqrt((WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B ** 2)
    wgs84_ep2_b: float = wgs84_ep ** 2 * WGS84_B
    wgs84_e2_a: float = WGS84_ECC_SQ * WGS84_A


def dist3fromllh(llh0, llh1):
    """
    Computing 3D distance from latitude, longitude and altitude

    Parameters:
    llh0: numpy array of shape(N,3)
    llh1: numpy array of shape(N,3)

    Returns:
    jnp.array(N,3): 3D distance
    """

    p0 = llh2ecef(llh0)
    p1 = llh2ecef(llh1)

    return ecef_distance(p0, p1)


def ecef_distance(p0, p1):
    """
    3D distance from ECEF coordinates

    Parameters:
    p0: numpy array of shape(N,3)
    p1: numpy array of shape(N,3)

    Returns:
    jnp.array(N,3): 3D distance
    """

    return jnp.sqrt(jnp.sum((p0 - p1) ** 2, axis=1))

    


def llh2ecef(llh):
    """
    Compute ECEF distance from latitude, longitude and altitude

    Parameters:
    llh: numpy array of shape(N,3)

    Returns:
    jnp.array(N,3): ECEF distance
    """

    lat = deg2rad(llh[:, 0])
    lon = deg2rad(llh[:, 1])
    alt = llh[:, 2]

    slat = jnp.sin(lat)
    slon = jnp.sin(lon)
    clat = jnp.cos(lat)
    clon = jnp.cos(lon)

    d = jnp.sqrt(1 - (slat * slat * geoC.WGS84_ECC_SQ))
    rn = geoC.WGS84_A / d

    x = (rn + alt) * clat * clon
    y = (rn + alt) * clat * slon
    z = (rn * (1 - geoC.WGS84_ECC_SQ) + alt) * slat

    return jnp.vstack((x, y, z)).T


def ecef2llh(ecef):
    """
    Compute latitude, longitude and altitude from ECEF distance

    Parameters:
    ecef: numpy array of shape(N,3)

    Returns:
    jnp.array(N,3): latitude, longitude, altitude
    """

    x = ecef[:, 0]
    y = ecef[:, 1]
    z = ecef[:, 2]

    lon = jnp.arctan2(y, x)

    p = jnp.sqrt(x ** 2 + y ** 2)
    th = jnp.arctan2(geoC.WGS84_A * z, geoC.WGS84_B * p)
    lat = jnp.arctan2(
        z + geoC.wgs84_ep2_b * jnp.sin(th) ** 3, p - geoC.wgs84_e2_a * jnp.cos(th) ** 3
    )

    N = geoC.WGS84_A / jnp.sqrt(1 - geoC.WGS84_ECC_SQ * jnp.sin(lat) ** 2)
    alt = p / jnp.cos(lat) - N

    return jnp.vstack((rad2deg(lat), rad2deg(lon), alt)).T


def rad2deg(radian):
    return radian * 180.0 / jnp.pi


def deg2rad(angle):
    return angle * jnp.pi / 180.0


def havdist(ll0, ll1):
    lat0 = deg2rad(ll0[:, 0])
    lon0 = deg2rad(ll0[:, 1])

    lat1 = deg2rad(ll1[:, 0])
    lon1 = deg2rad(ll1[:, 1])

    hav = haversine(lat1 - lat0) + jnp.cos(lat0) * jnp.cos(lat1) * haversine(lon1 - lon0)
    return 2 * geoC.WGS84_A * jnp.arcsin(jnp.sqrt(hav))


def haversine(theta):
    return jnp.sin(theta / 2) ** 2


def latlon2xy(lat, lon, ref_lat, ref_lon):
    y = (lat - ref_lat)/360 * geoC.CIRCUMFERENCE
    x = (lon - ref_lon)/360 * jnp.cos(ref_lat*jnp.pi/180) * geoC.CIRCUMFERENCE

    return jnp.hstack((x.reshape(-1,1), y.reshape(-1,1)))


def xy2latlon(x, y, ref_lat, ref_lon):
    lat = (y * 360 / geoC.CIRCUMFERENCE) + ref_lat
    lon = ((x * 360) / (geoC.CIRCUMFERENCE * jnp.cos(ref_lat*jnp.pi/180))) + ref_lon

    return jnp.hstack((lat.reshape(-1,1), lon.reshape(-1,1)))