#!/usr/bin/env python3

import itertools
import json

import numpy as np
import pandas as pd
from scipy.signal import resample

import jax.numpy as jnp
from jax.scipy.optimize import minimize

from geodesy import geodesy
from ltess import ltess
from mlat import exact, lls, nlls
from spec_load import spec_load
from tdoa import tdoa


def correct_fo(signal, PPM, fRS, fUS, samplingRate=2e6):
    """
    Correct frequency offset of IQ signal

    Parameters:
    signal: np array with IQ signal
    PPM: Clock drift

    Returns:
    np array with corrected signal
    """

    phi = PPM * 1e-6
    Ts = 1 / samplingRate

    t = Ts / (1 + phi) * np.arange(len(signal))
    up = np.size(np.arange(0, np.max(t), Ts))
    down = np.size(t)
    chunk_length = len(signal) // 3

    # First chunk
    c_0 = signal[0:chunk_length]
    t_0 = t[0:chunk_length]
    c_0_corrected = c_0 * np.exp(t_0 * (-1j * 2 * np.pi * phi * fRS))

    # Second chunk
    c_1 = signal[chunk_length : chunk_length * 2]
    t_1 = t[chunk_length : chunk_length * 2]
    c_1_corrected = c_1 * np.exp(t_1 * (-1j * 2 * np.pi * phi * fUS))

    # Third chunk
    c_2 = signal[chunk_length * 2 :]
    t_2 = t[chunk_length * 2 :]
    c_2_corrected = c_2 * np.exp(t_2 * (-1j * 2 * np.pi * phi * fRS))

    # Resampling phase
    signal_corrected = np.concatenate((c_0_corrected, c_1_corrected, c_2_corrected))
    return resample(signal_corrected, up)


def correct_fo_ltess(signal, PPM, fS=806e6, samplingRate=1.92e6):
    """
    Correct frequency offset of IQ signal (LTE frequency)

    Parameters:
    signal: np array with IQ signal
    PPM: Clock drift

    Returns:
    np array with corrected signal
    """

    phi = PPM * 1e-6
    Ts = 1 / samplingRate

    t = Ts / (1 + phi) * np.arange(len(signal))
    up = np.size(np.arange(0, np.max(t), Ts))
    down = np.size(t)

    signal = signal * np.exp(t * (-1j * 2 * np.pi * phi * fS))

    # Resampling phase
    return resample(signal, up)


def pytdoa(config):
    """
    Obtain the position

    Parameters:
    config: dictionary with the configuration

    Returns:
    np.array(3,): latitude, longitude, altitude of the transmitter
    """

    # Load configuration
    # Unknown transmitter
    fUS_MHz = config["transmitters"]["unknown"]["freq"]
    fUS = fUS_MHz * 1e6

    # Reference transmitter
    fRS_MHz = config["transmitters"]["reference"]["freq"]
    fRS = fRS_MHz * 1e6
    rs_lat = config["transmitters"]["reference"]["coord"][0]
    rs_lon = config["transmitters"]["reference"]["coord"][1]
    rs_alt = config["transmitters"]["reference"]["height"]
    rs_llh = np.array([rs_lat, rs_lon, rs_alt]).reshape(1, 3)

    # TDOA estimation
    try:
        corr_type = config["config"]["corr_type"]
    except:
        corr_type = "dphase"

    try:
        interpol = config["config"]["interpol"]
    except:
        interpol = 1

    # Sensor configurations
    sr_tdoa = config["config"]["sample_rate"]
    sr_ltess = config["config"]["sample_rate_ltess"]
    sensors = pd.DataFrame(config["sensors"])
    sensors[['latitude','longitude']] = sensors.coordinates.to_list()
    sensors = sensors.drop(['coordinates'],axis=1)
    directory = config["config"]["folder"]
    filenum = config["config"]["filenum"]
    NUM_SENSORS = len(sensors)

    # Correct the drift of the RTL-SDRs (requires knowing the drift in PPM)
    correct = config["config"]["correct"]

    # Return the accurate position from NLLS
    try:
        accurate = config["config"]["accurate"]
    except:
        accurate = False

    # Arrays to hold important things
    distances_rs = np.empty(0)

    # Load data and prepare it
    sensors_ecef = np.empty(0)
    sensors_llh = np.empty(0)
    samples = {}
    for (i, sensor) in sensors.iterrows():
        # Computing the distance to the Ref tX
        sensor_llh = np.array(
            [sensor["latitude"], sensor["longitude"], sensor["height"]]
        ).reshape(1, 3)
        dist = geodesy.dist3fromllh(rs_llh, sensor_llh)
        sensors.at[i,'ref_dist'] = dist

        # Compute ECEF coordinates per sensor
        ecef = geodesy.llh2ecef(sensor_llh).squeeze()
        sensors.at[i,'x'] = ecef[0]
        sensors.at[i,'y'] = ecef[1]
        sensors.at[i,'z'] = ecef[2]


        # Loading IQ data
        sname = sensor["name"]
        fname_tdoa = f"{directory}/{sname}/E{filenum}-{int(fRS_MHz)}_{int(fUS_MHz)}-localization.dat"
        fname_ltess = f"{directory}/{sname}/E{filenum}-ltess.dat"

        tdoa_iq = spec_load(fname_tdoa)
        ltess_iq = spec_load(fname_ltess)

        if correct:
            # Estimate Clock drift using the LTESS-Track tool
            (PPM, _, _) = ltess.ltess(ltess_iq, resample_factor=60)
            # Clock correction
            samples[sname] = correct_fo(tdoa_iq, PPM, fRS, fUS, samplingRate=sr_tdoa)
        else:
            samples[sname] = tdoa_iq

    # Get combinations and compute TDOAs per pair
    combinations = itertools.combinations(np.arange(len(sensors)), 2)

    combination_list = np.empty((0,2), dtype=int)
    tdoa_list = np.empty(0)

    for combination in combinations:
        i, j = combination[0], combination[1]

        combination_list = np.vstack((combination_list,[combination[0],combination[1]]))

        name_i = sensors.iloc[i]["name"]
        name_j = sensors.iloc[j]["name"]
        rx_diff = sensors.iloc[i]["ref_dist"] - sensors.iloc[j]["ref_dist"]

        tdoa_ij = tdoa.tdoa(samples[name_i], samples[name_j], rx_diff, interpol=interpol, corr_type=corr_type)
        tdoa_list = np.append(tdoa_list, [tdoa_ij["tdoa_m_i"]])

    # Optimization part
    sensors_llh = sensors[['latitude','longitude','height']].to_numpy()
    reference_c = np.mean(sensors_llh, axis=0)

    # We start by a simple Linear Optimization
    # We need to convert latitude,longitude pairs to their 2D representations
    sensors_xyz = geodesy.latlon2xy(sensors_llh[:,0], sensors_llh[:,1], reference_c[0], reference_c[1])

    if NUM_SENSORS > 3:
        result_lls = lls.lls(sensors_xyz, tdoa_list[:NUM_SENSORS-1])
    else:
        result_lls = exact.fang(sensors_xyz, tdoa_list[:NUM_SENSORS-1])
    
    result = geodesy.xy2latlon(result_lls[0], result_lls[1], reference_c[0], reference_c[1]).squeeze()

    # Non-Linear optimization if flag is set
    if accurate:
        sensors_ecef = geodesy.llh2ecef(sensors[['latitude','longitude','height']].to_numpy())
        sensors_mean = np.mean(sensors_ecef, axis=0)

        sensors_ecef = sensors_ecef - sensors_mean

        # Let's create our lambda for the optimization
        sensors_ecef_jnp = jnp.asarray(sensors_ecef)
        tdoas_jnp = jnp.asarray(tdoa_list)
        combinations_jnp = jnp.asarray(combination_list)
        optimfun = lambda X: nlls.nlls(X, sensors_ecef_jnp, tdoas_jnp, combinations_jnp)

        # Minimization routing
        X0 = geodesy.llh2ecef(np.append(result, reference_c[2]).reshape(1,3)) - sensors_mean
        X0_jnp = jnp.asarray(X0).reshape(-1,)
        res = minimize(optimfun, X0_jnp, method='BFGS')[0]

        res_np = np.array(res, copy=False).reshape(-1,3)
        result_accurate = geodesy.ecef2llh(res_np + sensors_mean).squeeze()

    else:
        result_accurate = result

    return np.array([result_accurate[0], result_accurate[1]])


###############################################################################
# MAIN definition
if __name__ == "__main__":

    with open(".config/config_fang.json") as f:
        config = json.load(f)

    position = pytdoa(config)

    print("Result: ", position.tolist())
