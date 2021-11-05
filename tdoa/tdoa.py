
import numpy as np
from scipy.signal import correlate, correlation_lags, detrend

from geodesy.geodesy import SPEED_OF_LIGHT as c

def correlate_arrays(s1, s2, normalize=True):
    """
    Given 2 signals, computes the cross correlation after applying normalization
    """
    if normalize:
        s1 = (s1 - np.mean(s1))/(np.std(s1)*len(s1))
        s2 = (s2 - np.mean(s2))/(np.std(s2))

    acor = correlate(s1, s2, mode="full")
    lags = correlation_lags(len(s1), len(s2), mode="full")
    
    return (acor, lags)


def correlate_iq(s1, s2, method="dphase"):
    """
    Compute cross correlation of 2 signals based on correlation method
    """

    if method == "iq":
        pass

    elif method == "abs":
        pass

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


def tdoa(
    s1,
    s2,
    rx_diff,
    signal_bandwidth=2e6,
    samples_per_frequency=1000000,
    guard=0.7,
    sample_rate=2e6,
    interpol=1,
    corr_type="dphase",
    report=1
):
    """
    Computes the TDOA between 2 signals using a Reference Receiver for synchronization
    """

    ## Signal preparation
    # Compute the points where we'll take the slices of the 3 parts of the signal
    center = round(0.5 * samples_per_frequency)
    left = center - round(0.5*guard*samples_per_frequency)
    right = center + round(0.5*guard*samples_per_frequency)
    span = right - left

    # Signal 1 slicing
    s11 = s1[left:right]
    s12 = s1[samples_per_frequency+left:samples_per_frequency+right]
    s13 = s1[2*samples_per_frequency+left:2*samples_per_frequency+right]

    # Signal 2 slicing
    s21 = s2[left:right]
    s22 = s2[samples_per_frequency+left:samples_per_frequency+right]
    s23 = s2[2*samples_per_frequency+left:2*samples_per_frequency+right]

    ## Correlations
    # First, we upsample all signals
    if interpol > 1:
        x = np.linspace(0, span, span*interpol, endpoint=False)
        xp = np.arange(0,span)

        s11 = np.interp(x,xp,s11)
        s12 = np.interp(x,xp,s12)
        s13 = np.interp(x,xp,s13)
        
        s21 = np.interp(x,xp,s21)
        s22 = np.interp(x,xp,s22)
        s23 = np.interp(x,xp,s23)

    # First chunk correlation
    [acor1, lags1] = correlate_iq(s11, s21, method=corr_type)
    midx1 = np.argmax(acor1)
    mcor1 = acor1[midx1]
    mlag1 = lags1[midx1]/interpol

    # Second chunk correlation
    [acor2, lags2] = correlate_iq(s12, s22, method=corr_type)
    midx2 = np.argmax(acor2)
    mcor2 = acor2[midx2]
    mlag2 = lags2[midx2]/interpol

    # Third chunk correlation
    [acor3, lags3] = correlate_iq(s13, s23, method=corr_type)
    midx3 = np.argmax(acor3)
    mcor3 = acor3[midx3]
    mlag3 = lags3[midx3]/interpol

    # Now compute the TDOAs
    if abs(mlag1 - mlag3) > 2:
        print("[WARNING] Delay between Reference Chunks is greater than 2 samples")
    
    mlag = (mlag1 + mlag3) / 2
    rx_diff_samples = rx_diff / c * sample_rate

    tdoa_s = mlag2 - mlag + rx_diff_samples
    tdoa_s_2 = mlag2 - mlag1 + rx_diff_samples
    tdoa_m = tdoa_s / sample_rate * c
    tdoa_m_2 = tdoa_s_2 / sample_rate * c

    if report > 0:
        print(' ')
        print('CORRELATION RESULTS')
        print(f'Raw Delay 1 (ref) in samples: {mlag1}. Reliability (0-1): {mcor1}')
        print(f'Raw Delay 2 (unk) in samples: {mlag2}. Reliability (0-1): {mcor2}')
        print(f'Raw Delay 3 (chk) in samples: {mlag3}. Reliability (0-1): {mcor3}')
        print(f'Merged Delay (1 & 3) in samples: {mlag}')

        print('REFERENCE TRANSMITTER')
        print(f'Distance to Reference TX [m]: {rx_diff}')
        print(f'Distance to Reference TX [samples]: {rx_diff_samples}')

        print('UNKNOWN TRANSMITTER')
        print(f'TDOA to Unknown TX (Merged) [m]: {tdoa_s}')
        print(f'TDOA to Unknown TX (Merged) [samples]: {tdoa_m}')
        print(f'TDOA to Unknown TX (Unmerged) [m]: {tdoa_s_2}')
        print(f'TDOA to Unknown TX (Unmerged) [samples]: {tdoa_m_2}')
        print(f'Reliability (Minimum correlation value): {np.min([mcor1, mcor2, mcor3])}')

    # Return dict with most important results
    return {
        'tdoa_s': tdoa_s,
        'tdoa_s_2': tdoa_s_2,
        'tdoa_m': tdoa_m,
        'tdoa_m_2': tdoa_m_2,
        'corr_val': np.min([mcor1, mcor2, mcor3])
    }

