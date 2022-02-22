import jax.numpy as jnp

from geodesy import geodesy_jnp as geodesy

def nlls(X, positions, tdoas, combinations):
    """
    Solve TDOA equations using the Non-Linear Least Squares approach
    The solutions contains the latitude and the longitude of the estimated
    transmitter position
    ---
    """

    # Compute all distances to the sensor
    d = geodesy.ecef_distance(positions, X)

    si = combinations[:,0]
    sj = combinations[:,1]

    t = d[si] - d[sj]

    err = jnp.square(tdoas - t)
    F = jnp.sum(err)

    return F
