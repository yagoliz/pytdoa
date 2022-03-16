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
#   Authors: Roberto Calvo-Palomino <roberto.calvo [at] imdea [dot] org>
#   Modified by: Yago Lizarribar <yago.lizarribar [at] imdea [dot] org>
#

import logging
import os
from pathlib import Path

import numpy as np


from pytdoa.ltess.foc.pssdrift import *

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger("LTESS")

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
        mainpath = Path(__file__).resolve().parent
        Z_sequences = np.array(
            [
                get_zadoof_seqs(f"{mainpath}/lte/25-Zadoff.bin"),
                get_zadoof_seqs(f"{mainpath}/lte/29-Zadoff.bin"),
                get_zadoof_seqs(f"{mainpath}/lte/34-Zadoff.bin"),
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

    logger.info(
        "Local oscilator error: %.8f PPM - [%.2f Hz], confidence=%.3f"
        % (PPM, delta_f, confidence)
    )

    return (PPM, delta_f, confidence)
