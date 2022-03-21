#!/usr/bin/env python3

#   Copyright (C) IMDEA Networks Institute 2022
#   This program is free software: you can redistribute it and/or modify
#
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see http://www.gnu.org/licenses/.
#
#   Authors: Yago Lizarribar <yago.lizarribar [at] imdea [dot] org>
#


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
from pytdoa.util import generate_heatmap, generate_hyperbola

from pytdoa.geodesy.geodesy import SPEED_OF_LIGHT, latlon2xy

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


def nonlinoptim(sensors, tdoas, combinations, llh=None):
    """
    Obtain the position by non linear methods

    Parameters:
    sensors: DataFrame with 'latitude', 'longitude', 'height' parameters
    tdoas: array of tdoa values of shape(n-1,1)
    combinations: array with combinations per sensor
    LLH0: Initial guess for latitude, longitude and altitude

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
    # If no initial point is given we start at the center
    if llh is None:
        X0 = np.zeros(shape=(3, 1))
    else:
        X0 = (geodesy.llh2ecef(llh.reshape(-1,3)) - sensors_mean).reshape(3,1)

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

    ###########################################################################
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
    interpol = config["config"].get("interp", 1)
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

    # Additional calculations
    add_heatmap = config["config"].get("generate_heatmap", False)
    add_hyperbola = config["config"].get("generate_hyperbola", False)

    ###########################################################################
    # Get and correct the samples per sensor
    # Load data and prepare it
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

    ###########################################################################
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

    ###########################################################################
    # Optimization part
    # There are several options to perform the optimization
    result = {}
    result["linear"] = np.empty((0))
    result["accurate"] = np.empty((0))

    # "linear": will perform only optimization using either the exact solution or LLS
    if method == "linear":
        target = linoptim(sensors, tdoa_list[:NUM_SENSORS-1])
        result["linear"] = target

    # "brute": will use brute force to estimate the solution
    elif method == "brute":
        target = brutefoptim(sensors, tdoa_list, combination_list)
        result["accurate"] = target

    # "nonlinear": will use NLLS to estimate the solution
    elif method == "nonlinear":
        target = nonlinoptim(sensors, tdoa_list, combination_list)
        result["accurate"] = target
    
    # "both": will first obtain an initial solution using linear methods and
    #         then will perform NLLS with the starting point given before
    elif method == "both":
        target_linear = linoptim(sensors, tdoa_list[:NUM_SENSORS-1])
        result["res_linear"] = target_linear

        altitude = np.mean(sensors["height"].to_numpy())
        target = nonlinoptim(sensors, tdoa_list, combination_list, llh=np.append(target_linear,altitude))
        result["res_accurate"] = target
    else:
        raise RuntimeError("Unsupported optimization method")

    ###########################################################################
    # Additional calculations like heatmap and hyperbolas
    rxs_llh = sensors[["latitude","longitude","height"]].to_numpy()
    ref_llh = np.mean(rxs_llh, axis=0)

    rxs_xyz = geodesy.latlon2xy(rxs_llh[:,0], rxs_llh[:,1], ref_llh[0], ref_llh[1])

    # Heatmap calculation
    if add_heatmap:
        xlim = 2000
        ylim = 2000
        target_xy = geodesy.latlon2xy(target[0], target[1], ref_llh[0], ref_llh[1])
        xrange = (target_xy[0,0]-xlim, target_xy[0,0]+xlim)
        yrange = (target_xy[0,1]-ylim, target_xy[0,1]+ylim)
        heatmap = generate_heatmap(tdoa_list, rxs_xyz, xrange, yrange, combination_list, step=10.0)

        cal_llh = geodesy.xy2latlon(heatmap[0], heatmap[1], ref_llh[0], ref_llh[1])

        result["hm"] = np.hstack((cal_llh, heatmap[2].reshape(-1,1)))
    else:
        result["hm"] = np.empty((0))

    # Hyperbola calculation
    if add_hyperbola:
        t = np.arange(0,2,0.001)
        hyperbolas = np.zeros((2*(NUM_SENSORS-1), len(t)*4))
        perturbation = 0.5 / sr_tdoa / interpol * SPEED_OF_LIGHT
        for i in range(NUM_SENSORS-1):
            si = combination_list[i,0]
            sj = combination_list[i,1]
            hyperbola_1 = generate_hyperbola(tdoa_list[i]+perturbation,rxs_xyz[si,:], rxs_xyz[sj,:], t)
            hyperbola_2 = generate_hyperbola(tdoa_list[i]-perturbation,rxs_xyz[si,:], rxs_xyz[sj,:], t)
            hyperbola_c = np.hstack((hyperbola_1, np.fliplr(hyperbola_2)))
            hyperbola_llh = geodesy.xy2latlon(hyperbola_c[0,:], hyperbola_c[1,:], ref_llh[0], ref_llh[1])
            hyperbolas[2*i:2*(i+1),:] = hyperbola_llh.T

        result["hyperbolas"] = hyperbolas

    else:
        result["hyperbolas"] = np.emtpy((0))

    
    return result

