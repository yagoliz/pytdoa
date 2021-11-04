#!/usr/bin/env python3

import itertools
import json

import numpy as np
from scipy.signal import resample

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
    us_lat = config["transmitters"]["unknown"]["coord"][0]
    us_lon = config["transmitters"]["unknown"]["coord"][1]
    us_alt = config["transmitters"]["unknown"]["height"]

    # Reference transmitter
    fRS_MHz = config["transmitters"]["reference"]["freq"]
    fRS = fRS_MHz * 1e6
    rs_lat = config["transmitters"]["reference"]["coord"][0]
    rs_lon = config["transmitters"]["reference"]["coord"][1]

    # Sensor configurations
    sr_tdoa = config["config"]["sample_rate"]
    sr_ltess = config["config"]["sample_rate_ltess"]
    sensors = config["sensors"]
    directory = config["config"]["folder"]
    filenum = config["config"]["filenum"]
    NUM_SENSORS = len(sensors)

    # Arrays to hold important things
    distances_rs = np.zeros((NUM_SENSORS, 1))

    # Load data and prepare it
    samples = {}
    for sensor in sensors:
        sname = sensor["name"]
        fname_tdoa = f"{directory}/{sname}/E{filenum}-{int(fRS_MHz)}_{int(fUS_MHz)}-localization.dat"
        fname_ltess = f"{directory}/{sname}/E{filenum}-ltess.dat"

        tdoa_iq = spec_load(fname_tdoa)
        ltess_iq = spec_load(fname_ltess)

        # Estimate Clock drift using the LTESS-Track tool
        (PPM, delta_f, confidence) = ltess.ltess(ltess_iq, resample_factor=60)

        # Clock correction
        samples[sname] = correct_fo(tdoa_iq, PPM, fRS, fUS, samplingRate=sr_tdoa)

    # Get combinations and compute TDOAs per pair
    combinations = itertools.combinations(np.arange(len(sensors)))
    tdoa_list = []

    

    return np.array([us_lat, us_lon, us_alt])


if __name__ == "__main__":

    with open("config.json") as f:
        config = json.load(f)

    position = pytdoa(config)

    print("Result: ", position.tolist())
