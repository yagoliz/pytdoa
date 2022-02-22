import numpy as np

def lls(positions, tdoas):
    """
    Solve TDOA equations using the Linear Least Squares approach
    The solutions contains the latitude and the longitude of the estimated
    transmitter position
    ---
    """

    (A, b) = getMatrices(positions, tdoas)
    result = np.linalg.lstsq(A,b)[0]
    return (result[1].real, result[2].real)


def getMatrices(positions, tdoas):
    # Initializing our dear variables
    A = np.zeros((len(tdoas),3))
    b = np.zeros((len(tdoas),1))

    for i in range(len(tdoas)):
        # System matrix
        A[i,0] = -tdoas[i]
        A[i,1] = positions[0,0] - positions[i+1,0]
        A[i,2] = positions[0,1] - positions[i+1,1]

        # Solutions
        b[i] = 0.5 * (tdoas[i]**2 + np.linalg.norm(positions[0,:])**2 - np.linalg.norm(positions[i+1,:])**2)
    
    return (A,b)
