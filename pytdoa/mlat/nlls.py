import numpy as np

from pytdoa.geodesy import geodesy


def nlls(X, positions, tdoas, combinations):
    """
    Solve TDOA equations using the Non-Linear Least Squares approach
    The solutions contain the ecef coordinates of the estimated
    transmitter position
    ---
    """

    # Compute all distances to the sensor
    d = geodesy.ecef_distance(positions, X)

    si = combinations[:, 0]
    sj = combinations[:, 1]

    t = d[si] - d[sj]

    err = np.square(tdoas - t)
    F = np.sum(err)

    return F


def nlls_llh(X, height, positions, positions_mean, tdoas, combinations):
    """
    Solve TDOA equations using the Non-Linear Least Squares approach
    The solutions contain the ecef coordinates of the estimated
    transmitter position
    ---
    """

    # In this case, X contains the latitude, longitude and height distribution
    X_ecef = geodesy.llh2ecef(np.append(X, height).reshape(1, 3)) - positions_mean

    # Compute all distances to the sensor
    d = geodesy.ecef_distance(positions, X_ecef)

    si = combinations[:, 0]
    sj = combinations[:, 1]

    t = d[si] - d[sj]

    err = np.square(tdoas - t)
    F = np.sum(err)

    return F
