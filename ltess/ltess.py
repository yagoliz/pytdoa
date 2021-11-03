import os

import numpy as np

from scipy import signal
from sklearn.preprocessing import MinMaxScaler

from ltess.foc.pssdrift import *

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Load Zadoof sequencies
def get_zadoof_seqs(filename):
    f = open(filename, "rb")
    bdata = np.fromfile(f, "<f4")
    cdata = np.vectorize(complex)(
        bdata[range(0, len(bdata), 2)], bdata[range(1, len(bdata), 2)]
    )
    return cdata


def ltess(
    samples,
    fc=806e6,
    fs=1.92e6,
    resample_factor=20,
    pss_step=9600,
    search_window=150,
    preamble=20,
    aux_buffer_size=1024,
):
    """
    Determine the clock drift with raw IQ data

    Parameters:
    samples: IQ samples

    Returns:
    Tuple with drift (PPM), change in frequency (delta_f) and confidence value
    """

    # load zadoof sequences (in time)
    try:
        cwd = os.getcwd()
        Z_sequences = np.array(
            [
                get_zadoof_seqs(f"{cwd}/ltess/lte/25-Zadoff.bin"),
                get_zadoof_seqs(f"{cwd}/ltess/lte/29-Zadoff.bin"),
                get_zadoof_seqs(f"{cwd}/ltess/lte/34-Zadoff.bin"),
            ]
        )
    except FileNotFoundError:
        Z_sequences = np.array(
            [
                get_zadoof_seqs("/usr/share/pyltesstrack/lte/25-Zadoff.bin"),
                get_zadoof_seqs("/usr/share/pyltesstrack/lte/29-Zadoff.bin"),
                get_zadoof_seqs("/usr/share/pyltesstrack/lte/34-Zadoff.bin"),
            ]
        )

    # Get drift by analyzing the PSS time of arrival
    [PPM, delta_f, confidence] = get_drift(
        samples,
        Z_sequences,
        preamble,
        pss_step,
        search_window,
        resample_factor,
        fs,
        debug_plot=False,
    )

    print(
        "[LTESSTRACK] Local oscilator error: %.8f PPM - [%.2f Hz], confidence=%.3f"
        % (PPM, delta_f, confidence)
    )

    return (PPM, delta_f, confidence)