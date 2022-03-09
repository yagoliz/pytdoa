#!/usr/bin/env python3

# Compute positions using the TDOA/MLAT algorithm
# Author: Yago Lizarribar
# Email: yago.lizarribar@imdea.org


################################################################################
# Imports
import itertools

import logging
import logging.config
import numpy as np
import pandas as pd
from scipy.signal import resample
import scipy.optimize as optimize

from pytdoa.geodesy import geodesy
from pytdoa.ltess import ltess
from pytdoa.mlat import exact, lls, nlls
from pytdoa.spec_load import spec_load
from pytdoa.tdoa import tdoa

logger = logging.getLogger("PYTDOA")

################################################################################
# PYTDOA functions

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


def linoptim(sensors, tdoas):
    """
    Obtain the position by linear methods

    Parameters:
    sensors: DataFrame with 'latitude', 'longitude', 'height' parameters
    tdoas: array of tdoa values of shape(n-1,1)

    Returns:
    np.array([lat,lon]) with the resulting latitude and longitude
    """

    sensors_llh = sensors[["latitude", "longitude", "height"]].to_numpy()
    reference_c = np.mean(sensors_llh, axis=0)

    sensors_xyz = geodesy.latlon2xy(
        sensors_llh[:, 0], sensors_llh[:, 1], reference_c[0], reference_c[1]
    )

    if sensors.shape[0] > 3:
        res = lls.lls(sensors_xyz, tdoas)
    else:
        res = exact.fang(sensors_xyz, tdoas)

    return geodesy.xy2latlon(res[0], res[1], reference_c[0], reference_c[1]).squeeze()


def brutefoptim(
    sensors,
    tdoas,
    combinations,
    ltrange=2,
    lnrange=2,
    step=0.05,
    epsilon=1e-4,
    maxiter=10,
    workers=1,
):
    """
    Obtain the position by brute force

    Parameters:
    sensors: DataFrame with 'latitude', 'longitude', 'height' parameters
    tdoas: array of tdoa values of shape(n-1,1)
    combinations: list with sensor combinations when computing tdoa pairs

    Returns:
    np.array([lat,lon, height]) with the resulting latitude and longitude
    """

    sensors_llh = sensors[["latitude", "longitude", "height"]].to_numpy()
    X0 = np.mean(sensors_llh, axis=0)
    altitude = X0[2]

    sensors_ecef = geodesy.llh2ecef(
        sensors[["latitude", "longitude", "height"]].to_numpy()
    )
    sensors_mean = np.mean(sensors_ecef, axis=0)

    optimfun = lambda X: nlls.nlls_llh(
        X, altitude, sensors_ecef - sensors_mean, sensors_mean, tdoas, combinations
    )

    Xr = np.array([X0[0], X0[1]])
    F_prev = None
    lt = ltrange
    ln = lnrange
    st = step
    Ns = int(2 * ltrange / step)
    for i in range(maxiter):
        llrange = (slice(Xr[0] - lt, Xr[0] + lt, st), slice(Xr[1] - ln, Xr[1] + ln, st))

        summary = optimize.brute(
            optimfun, llrange, full_output=True, finish=None, workers=workers
        )

        # We update all the values for the next iteration
        if F_prev is None:
            Xr = summary[0]
            F_prev = summary[1]

            lt = lt * 0.1
            ln = ln * 0.1
            st = 2 * lt / Ns
        else:
            Xr = summary[0]
            F = summary[1]

            if np.abs((F - F_prev) / F) < epsilon:
                return Xr

            F_prev = F

            lt = lt * 0.1
            ln = ln * 0.1
            st = 2 * lt / Ns

    logger.warning("Reached maximum number of iterations")
    return Xr


def nonlinoptim(sensors, tdoas, combinations):
    """
    Obtain the position by non linear methods

    Parameters:
    sensors: DataFrame with 'latitude', 'longitude', 'height' parameters
    tdoas: array of tdoa values of shape(n-1,1)

    Returns:
    np.array([lat,lon, height]) with the resulting latitude and longitude
    """

    sensors_ecef = geodesy.llh2ecef(
        sensors[["latitude", "longitude", "height"]].to_numpy()
    )
    sensors_mean = np.mean(sensors_ecef, axis=0)

    sensors_ecef = sensors_ecef - sensors_mean
    optimfun = lambda X: nlls.nlls(X, sensors_ecef, tdoas, combinations)

    # Minimization routine
    X0 = np.zeros(shape=(3, 1))

    summary = optimize.minimize(optimfun, X0, method="BFGS")

    res = np.array(summary.x, copy=False).reshape(-1, 3)
    return geodesy.ecef2llh(res + sensors_mean).squeeze()


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

    # Sensor configurations
    sr_tdoa = config["config"]["sample_rate"]
    sr_ltess = config["config"]["sample_rate_ltess"]
    sensors = pd.DataFrame(config["sensors"])
    sensors[["latitude", "longitude"]] = sensors.coordinates.to_list()
    sensors = sensors.drop(["coordinates"], axis=1)
    directory = config["config"]["folder"]
    filenum = config["config"]["filenum"]
    NUM_SENSORS = len(sensors)

    # TDOA estimation
    corr_type = config["config"].get("corr_type", "dphase")
    interpol = config["config"].get("interpol", 1)
    bw_rs = config["transmitters"]["reference"].get("bw", sr_tdoa)
    bw_us = config["transmitters"]["unknown"].get("bw", sr_tdoa)

    # Design the filter taps for all chunks
    taps_rs, taps_us = None, None
    if bw_rs < sr_tdoa:
        taps_rs = tdoa.design_filt(bw_rs, sr_tdoa)
        logger.info(f"Filter for reference signal of bandwidth {bw_rs:.2f} created")

    if bw_us < sr_tdoa:
        taps_us = tdoa.design_filt(bw_us, sr_tdoa)
        logger.info(f"Filter for targe signal of bandwidth {bw_us:.2f} created")

    # Correct the drift of the RTL-SDRs (requires knowing the drift in PPM)
    correct = config["config"]["correct"]

    # Return the accurate position from NLLS
    method = config["config"].get("method", "linear")

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
        sensors.at[i, "ref_dist"] = dist

        # Compute ECEF coordinates per sensor
        ecef = geodesy.llh2ecef(sensor_llh).squeeze()
        sensors.at[i, "x"] = ecef[0]
        sensors.at[i, "y"] = ecef[1]
        sensors.at[i, "z"] = ecef[2]

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

    combination_list = np.empty((0, 2), dtype=int)
    tdoa_list = np.empty(0)

    for combination in combinations:
        i, j = combination[0], combination[1]

        combination_list = np.vstack(
            (combination_list, [combination[0], combination[1]])
        )

        name_i = sensors.iloc[i]["name"]
        name_j = sensors.iloc[j]["name"]
        rx_diff = sensors.iloc[i]["ref_dist"] - sensors.iloc[j]["ref_dist"]

        tdoa_ij = tdoa.tdoa(
            samples[name_i],
            samples[name_j],
            rx_diff,
            interpol=interpol,
            corr_type=corr_type,
            taps_rs=taps_rs,
            taps_us=taps_us,
        )
        tdoa_list = np.append(tdoa_list, [tdoa_ij["tdoa_m_i"]])

    # Optimization part
    if method == "linear":
        result = linoptim(sensors, tdoa_list[: NUM_SENSORS - 1])

    elif method == "brute":
        result = brutefoptim(sensors, tdoa_list, combination_list)

    elif method == "nonlinear":
        result = nonlinoptim(sensors, tdoa_list, combination_list)

    else:
        raise RuntimeError("Unsupported optimization method")

    return np.array([result[0], result[1]])