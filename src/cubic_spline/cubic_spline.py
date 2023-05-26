import numpy as np
from numba import njit
# from .tools import *
from cubic_spline import tools


'''Interpolation for arrays
    - cubic_index: gridsearch
    - cubic_coef: spline coefficients
    - cubic_apply: interpolated values
    - cubic_y: all steps at once 

    We observe (x, y) and fit spline coefficients (a, b) so we can obtain (xq, yq).
'''

def cubic_index(x, xq, robust=False):
    """ Solves for the index value that left brackets each query point

    One of the following cases must be true:
        1) either x or xq has only one row (common, because xq is usually an exogenous grid)
        2) x and xq have the same number of rows, i.e. if x is (s,n), then xq is (s,nq), there are 
            no constraints on n and nq

    Parameters
    ----------
    x  : array (n) or (s,n), NumPy array of observed, increasing x values
    xq : array (nq) or (s,nq), NumPy array of desired query points
    robust : bool (optional), chooses search method, default exploits monotonicity in gridsearch,
        if xq is non-monotonic, use robust=True for binary search method
    
    Returns
    ----------
    xqi : array (nq) or (s,nq), array of integer values for the nearest left index to each point in xq
    
    """

    if x.ndim > 1 and x.shape[0] > 1 and xq.ndim > 1 and xq.shape[0] >1 and x.shape[0] != xq.shape[0]:
        raise ValueError('x and xq must have the same number of states (rows) if both have more than 1 state')

    if xq.ndim == 0:
        xqi = tools.point_index(x, xq)
        return xqi
    else:
        xqi = tools.array_index(x, xq, robust)
        return xqi.astype(int)
    

def cubic_coef(x, y):
    """Calculates cubic spline coefficients
    
    The restrictions on the shape of x and y are the same as those on x and xq in cubic_index

    Parameters
    ----------
    x  : array (n) or (s,n), NumPy array of observed, increasing x values
    y  : array (n) or (s,n), NumPy array of observed y values
    
    Returns
    ----------
    a  : array (n-1) or (s,n-1), NumPy array of coefficients
    b  : array (n-1) or (s,n-1), NumPy array of coefficients
    
    """

    if x.ndim > 1 and x.shape[0] > 1 and y.ndim > 1 and  y.shape[0] > 1 and x.shape[0] != y.shape[0]:
        raise ValueError('x and y must have the same number of states (rows) if both have more than 1 state')

    coef_a, coef_b = tools.cubic_coef_solver(x, y)

    if x.ndim > 1 or y.ndim > 1:
        a = np.delete(coef_a, -1, axis = 1)
        b = np.delete(coef_b, -1, axis = 1)
    elif x.ndim == 1 and y.ndim == 1:
        a = coef_a[:-1]
        b = coef_b[:-1]

    return a, b


def cubic_apply(x, xq, y, xqi, a, b):
    """ Using outputs from above equations, calculates yq, the interpolated points
        corresponding to xq

    Parameters
    ----------
    x   : array (n) or (s,n), NumPy array of observed, increasing x values
    xq  : array (nq) or (s,nq), NumPy array of query points of interest 
    y   : array (n) or (s,n), NumPy array of y values that correspond to x 
    xqi : array (nq) or (s,nq), integer NumPy array of left bracketing index values obtained with cubic_index
    a   : array (n-1) or (s,n-1), NumPy array of coefficients from cubic_coef
    b   : array (n-1) or (s,n-1), NumPy array of coefficients from cubic_coef
    
    Returns
    ----------
    yq  : array (nq) or (s,nq), NumPy array of interpolated query points
    
    """
    if xq.ndim == 0:
        yq = tools.cubic_apply_point(x, xq, y, xqi, a, b)
    else:
        yq = tools.cubic_apply_array(x, xq, y, xqi, a, b)
    
    return yq


def cubic_y(x, xq, y, robust=False):
    """ Applies above functions in one line

    Parameters
    ----------
    x   : array (n) or (s,n), NumPy array of observed, increasing x values
    xq  : array (nq) or (s,nq), NumPy array of query points
    y   : array (n) or (s,n), NumPy array of y values corresponding to the x input
    robust : bool (optional), chooses gridsearch method, see cubic_index for more details
    
    Returns
    ----------
    yq  : array (nq) or (s,nq), NumPy array of interpolated query points
    
    """
    xqi = cubic_index(x, xq, robust)
    a, b = cubic_coef(x, y)
    yq = cubic_apply(x, xq, y, xqi, a, b)
    return yq


'''Njitted interpolation of single point
    - cubic_vfi: know spline coefs. Useful in VFI.
    - cubic_egm: know spline coefs + position on grid. Useful in DC-EGM.
'''


@njit
def cubic_vfi(x, xq, y, a, b):
    """ Efficiently solves for interpolated value of a point given spline coefficients
    
    Parameters
    ----------
    x   : array (n), NumPy array of observed, increasing values
    xq  : scalar, query point of interest
    y   : array (n), NumPy array of y values corresponding to x
    a   : array (n-1), NumPy coefficient array from cubic_coef
    b   : array (n-1), NumPy coefficient array from cubic_coef
    
    Returns
    ----------
    yq  : scalar, interpolated value for the query point xq
    
    """
    xi = 0
    x_high = x[1]
    while xi < x.shape[0] - 2:
        if x_high >= xq:
            break
        xi += 1
        x_high = x[xi + 1]
    
    t = (xq - x[xi]) / (x[xi + 1] - x[xi])
    yq = (1 - t) * y[xi] + t * y[xi + 1] + t * (1 - t) * ( (1 - t) * a[xi] + t * b[xi])
    
    return yq


@njit
def cubic_egm(xq, x_low, x_high, y_low, y_high, aq, bq):
    """ Efficiently solves for interpolated value when the location of xq on the grid is known

    Parameters
    ----------
    xq     : scalar, query point of interest
    x_low  : scalar, gridpoint to the left of xq
    x_high : scalar, gridpoint to the right of xq
    y_low  : scalar, y value corresponding to x_low
    y_high : scalar, y value corresponding to x_high
    aq     : scalar, a coefficient corresponding to the index of x_low
    bq     : scalar, b coefficient corresponding to the index of x_low

    Returns
    ----------
    yq     : scalar, interpolated value for the query point xq
    
    
    """
    t = (xq - x_low) / (x_high - x_low)
    yq = (1 - t) * y_low + t * y_high + t * (1 - t) * ((1 - t) * aq + t * bq)
    return yq
