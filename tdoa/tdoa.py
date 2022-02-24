from random import sample
import numpy as np
from scipy.signal import correlate, correlation_lags, detrend
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy.interpolate import interp1d

from geodesy.geodesy import SPEED_OF_LIGHT as c


def correlate_arrays(s1, s2, normalize=True):
    """
    Given 2 signals, computes the cross correlation after applying normalization
    """
    if normalize:
        s1 = (s1 - np.mean(s1)) / (np.std(s1) * len(s1))
        s2 = (s2 - np.mean(s2)) / (np.std(s2))

    acor = correlate(s1, s2, mode="full")
    lags = correlation_lags(len(s1), len(s2), mode="full")

    return (acor, lags)


def correlate_iq(s1, s2, method="dphase"):
    """
    Compute cross correlation of 2 signals based on correlation method
    """

    if method == "iq":
        (acor, lags) = correlate_arrays(s1, s2)
        acor = np.abs(acor)

    elif method == "abs":
        a1 = np.abs(s1)
        a2 = np.abs(s2)

        (acor, lags) = correlate_arrays(a1, a2)

    elif method == "dphase":
        # Obtain the phase derivative
        d1 = np.diff(np.unwrap(np.angle(s1)), prepend=[0.0])
        d2 = np.diff(np.unwrap(np.angle(s2)), prepend=[0.0])

        # Remove linear trends from time signal
        d1 = detrend(d1)
        d2 = detrend(d2)

        (acor, lags) = correlate_arrays(d1, d2)

    else:
        raise RuntimeError("Unsupported Correlation Type")

    return (acor, lags)


def design_filt(bandwidth, fs, ripple=60, width_percent=0.05, cutoff_percent=0.05):

    # Obtain the passband & stopband frequencies
    cutoff = (1 + cutoff_percent) * bandwidth / 2
    width = width_percent * bandwidth / 2

    # First we get an estimation of the optimal number of taps
    numtaps, beta = kaiserord(ripple, width / (0.5 * fs))
    return firwin(numtaps, cutoff, window=("kaiser", beta), fs=fs)


def filter_iq(signal, taps):
    # Use lfilter function to filter our signal
    # Since the second parameter is set to 1, we will have a FIR filter
    return lfilter(taps, 1.0, signal)


def tdoa(
    s1,
    s2,
    rx_diff,
    taps_rs=None,
    taps_us=None,
    samples_per_frequency=1000000,
    guard=0.7,
    sample_rate=2e6,
    interpol=1,
    corr_type="dphase",
    report=1,
):
    """
    Computes the TDOA between 2 signals using a Reference Receiver for synchronization
    """

    ## Signal preparation
    # Compute the points where we'll take the slices of the 3 parts of the signal
    center = round(0.5 * samples_per_frequency)
    left = center - round(0.7 * guard * samples_per_frequency)
    right = center + round(0.7 * guard * samples_per_frequency)

    # Valid samples
    valid = 20

    # Signal 1 slicing
    s11 = s1[left:right]
    s12 = s1[samples_per_frequency + left : samples_per_frequency + right]
    s13 = s1[2 * samples_per_frequency + left : 2 * samples_per_frequency + right]

    # Signal 2 slicing
    s21 = s2[left:right]
    s22 = s2[samples_per_frequency + left : samples_per_frequency + right]
    s23 = s2[2 * samples_per_frequency + left : 2 * samples_per_frequency + right]

    # Filtering
    # Reference signals
    if taps_rs != None:
        s11 = filter_iq(s11, taps_rs)
        s13 = filter_iq(s13, taps_rs)

        s21 = filter_iq(s21, taps_rs)
        s23 = filter_iq(s23, taps_rs)

        print(f"[INFO] TDOA: Reference signals filtered")
    else:
        print("[INFO] TDOA: No filter applied to reference signals")

    # Target frequency
    if signal_bandwidth_rs < sample_rate:
        s12 = filter_iq(s12, taps_us)
        s22 = filter_iq(s22, taps_us)

        print(f"[INFO] TDOA: Target signals filtered")
    else:
        print("[INFO] TDOA: No filter applied to target signals")

    ## Correlations
    # First chunk correlation
    [acor1, lags1] = correlate_iq(s11, s21, method=corr_type)

    midx1 = np.argmax(acor1)
    mcor1 = acor1[midx1]
    mlag1 = lags1[midx1]

    # We upsample the acor result
    if interpol > 1:
        N = (2 * valid) * interpol
        lags1_interpolated = np.linspace(
            lags1[midx1 - valid], lags1[midx1 + valid] - 1, N, endpoint=False
        )

        # Create interpolator object
        interpolant = interp1d(
            lags1[midx1 - valid : midx1 + valid],
            acor1[midx1 - valid : midx1 + valid],
            kind="cubic",
        )
        acor1_interpolated = interpolant(lags1_interpolated)

        midx1_interpolated = np.argmax(acor1_interpolated)
        mcor1_interpolated = acor1_interpolated[midx1_interpolated]
        mlag1_interpolated = lags1_interpolated[midx1_interpolated]
    else:
        midx1_interpolated = midx1
        mcor1_interpolated = mcor1
        mlag1_interpolated = mlag1

    # Second chunk correlation
    [acor2, lags2] = correlate_iq(s12, s22, method=corr_type)
    midx2 = np.argmax(acor2)
    mcor2 = acor2[midx2]
    mlag2 = lags2[midx2]

    if interpol > 1:
        lags2_interpolated = np.linspace(
            lags2[midx2 - valid], lags2[midx2 + valid] - 1, N, endpoint=False
        )

        # Create interpolator object
        interpolant = interp1d(
            lags2[midx2 - valid : midx2 + valid],
            acor2[midx2 - valid : midx2 + valid],
            kind="cubic",
        )
        acor2_interpolated = interpolant(lags2_interpolated)

        midx2_interpolated = np.argmax(acor2_interpolated)
        mcor2_interpolated = acor2_interpolated[midx2_interpolated]
        mlag2_interpolated = lags2_interpolated[midx2_interpolated]
    else:
        midx2_interpolated = midx2
        mcor2_interpolated = mcor2
        mlag2_interpolated = mlag2

    # Third chunk correlation
    [acor3, lags3] = correlate_iq(s13, s23, method=corr_type)
    midx3 = np.argmax(acor3)
    mcor3 = acor3[midx3]
    mlag3 = lags3[midx3]

    if interpol > 1:
        lags3_interpolated = np.linspace(
            lags3[midx3 - valid], lags3[midx3 + valid] - 1, N, endpoint=False
        )

        # Create interpolator object
        interpolant = interp1d(
            lags3[midx3 - valid : midx3 + valid],
            acor2[midx3 - valid : midx3 + valid],
            kind="cubic",
        )
        acor3_interpolated = interpolant(lags3_interpolated)

        midx3_interpolated = np.argmax(acor3_interpolated)
        mcor3_interpolated = acor3_interpolated[midx3_interpolated]
        mlag3_interpolated = lags3_interpolated[midx3_interpolated]
    else:
        midx3_interpolated = midx3
        mcor3_interpolated = mcor3
        mlag3_interpolated = mlag3

    # Now compute the TDOAs
    if abs(mlag1 - mlag3) > 2:
        print("[WARNING] Delay between Reference Chunks is greater than 2 samples")

    mlag = (mlag1 + mlag3) / 2
    mlag_interpolated = (mlag1_interpolated + mlag3_interpolated) / 2
    rx_diff_samples = rx_diff / c * sample_rate

    tdoa_s = mlag2 - mlag + rx_diff_samples
    tdoa_s_interpolated = mlag2_interpolated - mlag_interpolated + rx_diff_samples
    tdoa_s_2 = mlag2 - mlag1 + rx_diff_samples
    tdoa_s_2_interpolated = mlag2_interpolated - mlag1_interpolated + rx_diff_samples
    tdoa_m = tdoa_s / sample_rate * c
    tdoa_m_interpolated = tdoa_s_interpolated / sample_rate * c
    tdoa_m_2 = tdoa_s_2 / sample_rate * c
    tdoa_m_2_interpolated = tdoa_s_2_interpolated / sample_rate * c

    if report > 0:
        print(" ")
        print("[INFO] TDOA: CORRELATION RESULTS")
        print(
            f"\tRaw Delay 1 (ref) in samples (Regular / Upsampled): {mlag1}/{mlag1_interpolated:.1f}. Reliability (0-1): {mcor1:.2f}/{mcor1_interpolated:.2f}"
        )
        print(
            f"\tRaw Delay 2 (unk) in samples (Regular / Upsampled): {mlag2}/{mlag2_interpolated:.1f}. Reliability (0-1): {mcor2:.2f}/{mcor2_interpolated:.2f}"
        )
        print(
            f"\tRaw Delay 3 (chk) in samples (Regular / Upsampled): {mlag3}/{mlag3_interpolated:.1f}. Reliability (0-1): {mcor3:.2f}/{mcor3_interpolated:.2f}"
        )
        print(
            f"\tMerged Delay (1 & 3) in samples (Regular / Upsampled): {mlag}/{mlag_interpolated:.1f}"
        )

        print("[INFO] TDOA: REFERENCE TRANSMITTER")
        print(f"\tDistance to Reference TX [m]: {rx_diff}")
        print(f"\tDistance to Reference TX [samples]: {rx_diff_samples}")

        print("[INFO] TDOA: UNKNOWN TRANSMITTER")
        print("------ Regular")
        print(f"\tTDOA to Unknown TX (Merged) [m]: {tdoa_s}")
        print(f"\tTDOA to Unknown TX (Merged) [samples]: {tdoa_m:.2f}")
        print(f"\tTDOA to Unknown TX (Unmerged) [m]: {tdoa_s_2}")
        print(f"\tTDOA to Unknown TX (Unmerged) [samples]: {tdoa_m_2:.2f}")
        print("------ Upsampled")
        print(f"\tTDOA to Unknown TX (Merged) [m]: {tdoa_s_interpolated:.1f}")
        print(f"\tTDOA to Unknown TX (Merged) [samples]: {tdoa_m_interpolated:.2f}")
        print(f"\tTDOA to Unknown TX (Unmerged) [m]: {tdoa_s_2_interpolated:.1f}")
        print(f"\tTDOA to Unknown TX (Unmerged) [samples]: {tdoa_m_2_interpolated:.2f}")
        print(
            f"\t- Reliability (Minimum correlation value): {np.min([mcor1, mcor2, mcor3])}"
        )
        print(
            f"\t- Reliability Upsampled (Minimum correlation value): {np.min([mcor1_interpolated, mcor2_interpolated, mcor3_interpolated]):.2f}"
        )

    # Return dict with most important results
    return {
        "tdoa_s": tdoa_s,
        "tdoa_s_2": tdoa_s_2,
        "tdoa_m": tdoa_m,
        "tdoa_m_2": tdoa_m_2,
        "tdoa_s_i": tdoa_s_interpolated,
        "tdoa_s_2_i": tdoa_s_2_interpolated,
        "tdoa_m_i": tdoa_m_interpolated,
        "tdoa_m_2_i": tdoa_m_2_interpolated,
        "corr_val": np.min([mcor1, mcor2, mcor3]),
    }
