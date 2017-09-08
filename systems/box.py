#pylint: disable=C0103
"""
This system represents the box potential
"""

from __future__ import (division, print_function)
import numpy as np
from FEM.core.fem import FEM

class Box_Potential(FEM):
    """ H = - hbar*2 \Laplace on the domain x = [0,1]
    """

    def __init__(self, hbar, N_cell, max_order, dirichlet_bound_conds=True):
        """
        """
        x_grid = np.linspace(0, 1, N_cell+1)
        V_fct = None
        FEM.__init__(self, hbar, max_order, x_grid,
                     dirichlet_bound_conds=dirichlet_bound_conds)

    def V_fct(self, q):
        """
        """
        return 0.0*q

    def E(self, n):
        """Analytic Spectrum
        """
        return 0.5 * (np.pi * (n+1) * self.hbar)**2

    def psi(self, x, n=0):
        """Analytic Eigenfunction
        """
        return np.sin(np.pi * (n+1) * x) * np.sqrt(2.0)

