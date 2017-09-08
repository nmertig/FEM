#pylint: disable=C0103
"""
This Program tests the FEM Cell code Provided by

FEM_Cell_Legendre for setting up and solving the box potential

"""

from __future__ import (division, print_function)
import numpy as np
from FEM.core.fem_kicked import FEMKicked

class Fishman(FEMKicked):
    """
    """
    def __init__(self, kappa, hbar, max_order, scaling_radius, N_int, x_max,
                 N_ext, theta=np.pi/8.0, dirichlet_bound_conds=False):
        """
        """
        self.kappa = kappa
        self.x_max = x_max

        xl = np.linspace(-x_max, -scaling_radius, N_ext+1)
        xc = np.linspace(-scaling_radius, scaling_radius, N_int+1)
        xr = np.linspace(scaling_radius, x_max, N_ext+1)
        x_grid = np.concatenate((np.concatenate((xl[:-1], xc)), xr[1:]))

        ml = np.ones(N_ext, dtype=bool)
        mc = np.zeros(N_int, dtype=bool)
        mr = np.ones(N_ext, dtype=bool)
        mask = np.concatenate((np.concatenate((ml, mc)), mr))

        FEMKicked.__init__(self, hbar, max_order, x_grid, theta=theta,
                           dirichlet_bound_conds=dirichlet_bound_conds,
                           exterior_scaling=True,
                           exterior_scaling_xmin=-scaling_radius,
                           exterior_scaling_xmax=scaling_radius,
                           exterior_scaling_mask=mask)

    def V_fct(self, x):
        """
        """
        return -self.kappa * np.exp(- 8.0 * x**2 ) / 16.0

    def potential_oscillation_length(self):
        """
        """
        h = 2.0 * np.pi * self.hbar
        return h * 8.0 * np.exp(0.5)/self.kappa
