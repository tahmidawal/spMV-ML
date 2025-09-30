import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass

import poisson_utils.basis
import poisson_utils.tensor





@dataclass
class Refel:
    p_order: int
    N_1d: int

    # (p_order + 1) Guass-Lobatto quadarature points & weights in 1D
    gll_x_1d: np.ndarray
    gll_w_1d: np.ndarray

    # (p_order + 1) Guass quadarature points & weights in 1D
    gauss_x_1d: np.ndarray
    gauss_w_1d: np.ndarray

    # (p_order + 1)^3 Guass quadarature in 1D
    gauss_w_3d: np.ndarray

    # Transformation matrix from Guass-Lobatto points to Gauss points in 1D
    gll_to_gauss_1d: np.ndarray


    # Derivative of Lagrange interpolants at the gauss quad points
    # ( N_1d x N_1d )
    # d_gauss_1d[i,j] = lagrange_j' (xGauss_i)
    d_gauss_1d: np.ndarray


    def __init__(self, p_order):
        self.p_order = p_order
        self.N_1d = p_order +1

        self.gll_x_1d, self.gll_w_1d = poisson_utils.basis.gll(0, 0, self.p_order)
        self.gauss_x_1d, self.gauss_w_1d = poisson_utils.basis.gauss(0, 0, self.p_order)

        self.gauss_w_3d = (self.gauss_w_1d[:, None, None] * self.gauss_w_1d[None, :, None] * self.gauss_w_1d[None, None, :]).flatten()

        # 1D Vandermonde matrix of Legendre polynomials at GLL points 
        # Vr[i, j] = L_i(xGLL_j)
        Vr = np.zeros((self.N_1d, self.N_1d))
        # Its drivative
        gradVr = np.zeros((self.N_1d, self.N_1d))

        # 1D Vandermonde matrix of Legendre polynomials at gauss points 
        # Vr[i, j] = L_i(xGauss_j)
        Vg = np.zeros((self.N_1d, self.N_1d))
        # Its drivative
        gradVg = np.zeros((self.N_1d, self.N_1d))

        for i in range(self.N_1d):
            Vr[i, :]     = poisson_utils.basis.polynomial(self.gll_x_1d, 0, 0, i)
            gradVr[i, :] = poisson_utils.basis.gradient(self.gll_x_1d, 0, 0, i)

            Vg[i, :]     = poisson_utils.basis.polynomial(self.gauss_x_1d, 0, 0, i)
            gradVg[i, :] = poisson_utils.basis.gradient(self.gauss_x_1d, 0, 0, i)
        
        self.gll_to_gauss_1d = np.linalg.solve(Vr, Vg).transpose()
        self.d_gauss_1d = np.linalg.solve(Vr, gradVg).transpose()

