import numpy as np
from numba import njit

import sequence_jacobian as sj
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import Continuous1D, ExogenousMaker
import cubic_spline as cs


'''Top level block construction'''


def build_block():
    tfp_stage = ExogenousMaker(markov_name='z_markov', index=0, name='tfp-shock')
    hetinputs = [make_grids, preprod]
    prod_stage = Continuous1D(backward=['V'], policy='k', f=vfi, name ='production', hetoutputs=[profit])
    return StageBlock([tfp_stage, prod_stage], name='firm', backward_init=init_firm, hetinputs=hetinputs)


'''Hetinputs
- functions that generate multidimensional inputs to core block
'''


def make_grids(rho_z, sd_z, n_z, min_k, max_k, n_k):
    """Markov process for TFP, capital grid."""
    z_grid_raw, z_dist, z_markov = sj.grids.markov_rouwenhorst(rho_z, sd_z, n_z)
    k_grid = sj.grids.agrid(max_k, n_k, min_k)
    return z_grid_raw, z_dist, z_markov, k_grid


def preprod(k_grid, z_grid_raw, w, alpha, drs, tfp, delta, re):
    """Precompute policies that don't depend on investment."""
    df = 1 / (1 + re)
    z_grid = z_grid_raw * tfp                       # scale tfp
    l_pre = (w / ((1 - alpha) * drs * z_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** (alpha * drs))) ** (1 / ((1 - alpha) * drs - 1)) # labor demand
    y_pre = z_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** (alpha * drs) * l_pre ** ((1 - alpha) * drs)  # production
    pi = y_pre - w * l_pre                                  # revenue minus labor cost
    mpk = alpha * drs * y_pre / k_grid[np.newaxis, :]   # marginal profit
    k_na = (1 - delta) * k_grid                     # capital of non-adjusters
    kqi_na = cs.cubic_index(k_grid, k_na)           # grid search for non-adjusters
    return df, z_grid, l_pre, y_pre, pi, mpk, k_na, kqi_na


def profit(y_pre, l_pre):
    y = y_pre
    l = l_pre
    return y, l


def init_firm(y_pre, l_pre, w, k_grid, z_grid, delta):
    """Provide initial guess for backward iteration."""
    i_guess = delta * k_grid[np.newaxis, :] * np.ones_like(z_grid)[:, np.newaxis]
    V = (y_pre - w * l_pre - i_guess) / 0.05
    Vk = np.empty_like(V)
    Vk[..., 1:-1] = (V[..., 2:] - V[..., :-2]) / (k_grid[2:] - k_grid[:-2])
    Vk[..., 0] = (V[..., 1] - V[..., 0]) / (k_grid[1] - k_grid[0])
    Vk[..., -1] = (V[..., -1] - V[..., -2]) / (k_grid[-1] - k_grid[-2])
    return V, Vk


'''Value function iteration'''


def vfi(V, k_grid, pi, df, delta, phi, xibar, min_k, max_k, tol_bellman, k_na):
    # discounting
    W = df * V

    # non-adjuster
    W_na = cs.cubic_y(k_grid, k_na, W)  # x, xq, y
    V_na = pi + W_na

    # adjuster
    a, b = cs.cubic_coef(k_grid, W)
    V_ad, k_ad = solve_bellman(W, k_grid, pi, delta, phi, a, b, min_k, max_k, tol_bellman)
    i_ad = k_ad - (1 - delta) * k_grid[np.newaxis, :]

    # upper envelope
    p_ad, p_na, xi_exp = uniform_cost(V_na, V_ad, xibar)
    V = p_na * V_na + p_ad * (V_ad - xi_exp)
    k = p_na * k_na[np.newaxis, :] + p_ad * k_ad
    i = p_ad * i_ad

    return V, k, i, i_ad, p_ad, p_na, xi_exp


@njit
def solve_bellman(W, k_grid, pi, delta, phi, a, b, min_k, max_k, tol_bellman):
    """Return value function and capital policy."""
    n_z, n_k = W.shape
    k = np.empty_like(W)
    V = np.empty_like(W)

    for iz in range(n_z):
        for ik in range(n_k):
            k_ = k_grid[ik]
            k[iz, ik] = golden_section(obj_bellman, min_k, max_k, args=(k_grid, W[iz, :], delta, k_, pi[iz, ik], phi, a[iz, :], b[iz, :]), tol=tol_bellman)
            V[iz, ik] = -obj_bellman(k[iz, ik], k_grid, W[iz, :], delta, k_, pi[iz, ik], phi, a[iz, :], b[iz, :])

    return V, k


@njit
def obj_bellman(k, k_grid, W, delta, k_, pi, phi, a, b):
    """Evaluate Bellman for capital choice k."""
    W_interp = cs.cubic_vfi(k_grid, k, W, a, b) # x, xq, y
    i = k - (1 - delta) * k_
    value = pi - i - adjust(i, k_, phi) + W_interp
    return -value


@njit
def adjust(i, k_, phi):
    """Adjustment cost (without fixed cost)."""
    return phi / 2 * i ** 2 / k_


def uniform_cost(V_na, V_ad, xibar):
    """Return adjustment choice probabilities and fixed cost threshold on the grid."""
    xihat = V_ad - V_na
    xihat[xihat < 0] = 0
    xihat[xihat > xibar] = xibar
    p_ad = xihat / xibar
    p_na = 1 - p_ad
    xi_exp = xihat / 2
    return p_ad, p_na, xi_exp


"""Golden section search from Jeppe Druedahl"""


@njit
def golden_section(obj, a, b, args=(), tol=1e-6):
    """ golden section search optimizer
    
    Args:
        obj (callable): 1d function to optimize over
        a (double): minimum of starting bracket
        b (double): maximum of starting bracket
        args (tuple): additional arguments to the objective function
        tol (double, optional): tolerance
    Returns:
        (float): optimization result
    
    """
    
    inv_phi = (np.sqrt(5) - 1) / 2    # 1/phi                                                                                                                
    inv_phi_sq = (3 - np.sqrt(5)) / 2 # 1/phi^2     
        
    # a. distance
    dist = b - a
    if dist <= tol: 
        return (a + b) / 2

    # b. number of iterations
    n = int(np.ceil(np.log(tol/dist)/np.log(inv_phi)))

    # c. potential new mid-points
    c = a + inv_phi_sq * dist
    d = a + inv_phi * dist
    yc = obj(c, *args)
    yd = obj(d, *args)

    # d. loop
    for _ in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            dist = inv_phi * dist
            c = a + inv_phi_sq * dist
            yc = obj(c, *args)
        else:
            a = c
            c = d
            yc = yd
            dist = inv_phi * dist
            d = a + inv_phi * dist
            yd = obj(d, *args)

    # e. return
    if yc < yd:
        return (a + d) / 2
    else:
        return (c + b) / 2