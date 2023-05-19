import numpy as np
from numba import njit, guvectorize


'''Gridsearch'''


@guvectorize(['void(float64[:], float64, int64[:])'], '(n),()->()')
def point_index(x, xq, xqi):
    xi = 0
    x_high = x[1]
    while xi < x.shape[0] - 2:
        if x_high >= xq:
            break
        xi += 1
        x_high = x[xi + 1]
    
    xqi[0] = xi


@guvectorize(['void(float64[:], float64[:], boolean, float64[:])'], '(n),(nq),()->(nq)')
def array_index(x, xq, robust, xqi):
    if robust:
        n = len(x)
        nq = len(xq)

        for iq in range(nq):
            if xq[iq] < x[0]:
                ilow = 0
            elif xq[iq] > x[-2]:
                ilow = n-2
            else:
                # start binary search
                # should end with ilow and ihigh exactly 1 apart, bracketing variable
                ihigh = n-1
                ilow = 0
                while ihigh - ilow > 1:
                    imid = (ihigh + ilow) // 2
                    if xq[iq] > x[imid]:
                        ilow = imid
                    else:
                        ihigh = imid

            xqi[iq] = ilow    
    else:
        nxq, nx = xq.shape[0], x.shape[0]

        xi = 0
        x_high = x[1]
        for xqi_cur in range(nxq):
            xq_cur = xq[xqi_cur]
            while xi < nx - 2:
                if x_high >= xq_cur:
                    break
                xi += 1
                x_high = x[xi + 1]

            xqi[xqi_cur] = xi


'''Solve for spline coefficients.'''

# Thomas algorithm from Claudio Bellei
@njit
def sparse_solver(a, b, c, d):
    nf = len(a)
    for it in range(1, nf):
        mc = a[it]/b[it-1]
        b[it] = b[it] - mc*c[it-1]
        d[it] = d[it] - mc*d[it-1]

    xc = a
    xc[-1] = d[-1]/b[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (d[il]-c[il]*xc[il+1])/b[il]

    return xc


@guvectorize(['void(float64[:],float64[:],float64[:],float64[:])'], '(n),(n)->(n),(n)')
def cubic_coef_solver(x, y, coef_a, coef_b):
    d = np.zeros(len(x))
    d[0] = 3 * (y[1] - y[0]) / (x[1] - x[0])**2
    d[-1] = 3 * (y[-1] - y[-2]) / (x[-1] - x[-2])**2
    d[1:-1] = 3 * ( (y[1:-1] - y[:-2]) / (x[1:-1] - x[:-2])**2 + (y[2:] - y[1:-1]) / (x[2:] - x[1:-1])**2 )

    b = np.zeros_like(d)
    b[0] = 2 / (x[1] - x[0])
    b[-1] = 2 / (x[-1] - x[-2])
    b[1:-1] = 2 * (1 / (x[1:-1] - x[:-2]) + 1 / (x[2:] - x[1:-1]))

    a = np.zeros_like(d)
    a[1:] = 1 / (x[1:] - x[:-1])

    c = np.zeros_like(d)
    c[:-1] = 1 / (x[1:] - x[:-1])

    k = sparse_solver(a, b, c, d)

    for i in range(len(k) - 1):
        coef_a[i] = k[i] * (x[i+1] - x[i]) - (y[i+1] - y[i])
        coef_b[i] = -k[i+1] * (x[i+1] - x[i]) + (y[i+1] - y[i])


'''Apply spline coefficients'''


@guvectorize(['void(float64[:], float64[:], float64[:], int64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n),(nq),(m),(m)->(nq)')
def cubic_apply_array(x, xq, y, xqi, a, b, yq):
    for j in range(len(xq)):
        t = (xq[j] - x[xqi[j]]) / (x[xqi[j] + 1] - x[xqi[j]])
        yq[j] = (1 - t) * y[xqi[j]] + t * y[xqi[j] + 1] + t * (1 - t) * ( (1 - t) * a[xqi[j]] + t * b[xqi[j]])


@guvectorize(['void(float64[:], float64, float64[:], int64, float64[:], float64[:], float64[:])'], '(n),(),(n),(),(m),(m)->()')
def cubic_apply_point(x, xq, y, xqi, a, b, yq):
    t = (xq - x[xqi]) / (x[xqi + 1] - x[xqi])
    yq[0] = (1 - t) * y[xqi] + t * y[xqi + 1] + t * (1 - t) * ( (1 - t) * a[xqi] + t * b[xqi])
