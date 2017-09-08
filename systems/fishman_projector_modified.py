#pylint: disable=C0103
"""
This Program tests the FEM Cell code Provided by

FEM_Cell_Legendre for setting up and solving the box potential

"""

from __future__ import (division, print_function)
import numpy as np
from scipy.special import erf
from FEM.core.fem_kicked import FEMKicked

class Fishman(FEMKicked):
    """ kicked fishman map FEM code
    """
    def __init__(self, kappa, hbar, max_order, N, x_max, x_b=1.0, x_f=1.2):
        """
        """
        self.kappa = kappa
        self.x_max = x_max
        x_grid = np.linspace(-x_max, x_max, N)
        FEMKicked.__init__(self, hbar, max_order, x_grid, theta=0.0,
                           dirichlet_bound_conds=True)
        self.x_b = x_b # bump location
        self.x_f = x_f # fixed point location
        # set the perturbation strength
        f1 = self.x_f * self.kappa * np.sqrt(np.pi/8.0) / 2.0
        f2 = np.exp(-8.0 * self.x_b * (2.0 * self.x_f - self.x_b))
        f3 = 1.0 - np.exp(-32.0 * self.x_f * self.x_b)
        self.eps = f1 * f2 / f3

    def V_fct(self, x):
        sqrt_8 = np.sqrt(8.0)
        xr = sqrt_8 * (x-self.x_b)
        xl = sqrt_8 * (x+self.x_b)
        return -self.kappa * np.exp(- 8.0 * x**2 ) / 16.0 \
               -self.eps * erf(xr) + self.eps * erf(xl)

    def potential_oscillation_length(self):
        """
        """
        h = 2.0 * np.pi * self.hbar
        return h * 8.0 * np.exp(0.5)/self.kappa
