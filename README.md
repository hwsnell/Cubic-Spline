# Cubic Spline Interpolation

This package carries out cubic spline interpolation. The intended use of this package is to solve dynamic decision problems of heterogeneous agents in conjunction with the [Sequence-Space Jacobian (SSJ) package](https://github.com/shade-econ/sequence-jacobian). 

The approach differs from the `CubicSpline` method of SciPy in that our package is compatible with Numba commands which increase performance in large problems. However, for now, the package is limited to the natural boundary condition.

In this package, we use basic spline interpolation of exactly degree 3, the cubic spline. We follow the methods found here: https://en.wikipedia.org/wiki/Spline_interpolation. We use the natural boundary condition, which is extends a straight line from the end points at the same slope as that end point. In formal terms, the second derivative at the end points is equal to 0. We make this choice to efficiently solve the interpolation problem. The natural boundary gives a tridiagonal matrix when the constraints are put in matrix form, and this allows for the Thomas algorithm to quickly solve the matrix problem. In SciPy, we can carry out the same interpolation and get very nearly the same results by running

```
interpolate.CubicSpline(x, y, bc_type='natural')
```

## Installation

`cubic_spline` runs on NumPy and Numba. The pacakge was developed with Python 3.9 with NumPy version 1.21 and Numba version 0.55. To install run the following command

```
pip install cubic-spline
```


## Using the Pacakge

`cubic_spline.py` and `tools.py` are the necessary files for carrying out cubic interpolation. Users will normally only need to interact with `cubic_spline` as it contains the high level wrapper functions. The other files, `vfi_demo` and `firm_engine`, provide an application of this cubic interpolation package to a heterogeneous firm optimization problem. `firm_engine` requires the SSJ pacakge to be installed. `vfi_demo` is a notebook that walks through the canonical lumpy investment problem with value function iteration making use of the `cubic_spline` methods.
