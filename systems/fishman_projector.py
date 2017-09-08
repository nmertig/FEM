#pylint: disable=C0103
"""
This Program tests the FEM Cell code Provided by

FEM_Cell_Legendre for setting up and solving the box potential

"""

from __future__ import (division, print_function)
import numpy as np
from FEM.core.fem_kicked import FEMKicked

class Fishman(FEMKicked):
    """ kicked fishman map FEM code
    """
    def __init__(self, kappa, hbar, max_order, N, x_max):
        """
        """
        self.kappa = kappa
        self.x_max = x_max
        x_grid = np.linspace(-x_max, x_max, N)
        FEMKicked.__init__(self, hbar, max_order, x_grid, theta=0.0,
                           dirichlet_bound_conds=True)

    def V_fct(self, x):
        """
        """
        return -self.kappa * np.exp(- 8.0 * x**2 ) / 16.0

    def potential_oscillation_length(self):
        """
        """
        h = 2.0 * np.pi * self.hbar
        return h * 8.0 * np.exp(0.5)/self.kappa
