#pylint: disable=C0103
"""
This class represents the harmonic oscillator based on the fem method
"""

from __future__ import (division, print_function)
import numpy as np
from numpy.polynomial import Hermite
from scipy.special import gamma

from FEM.core.fem import FEM

class HarmonicOscillator(FEM):
    """ H = - hbar*2 \Laplace on the domain x = [0,1]
    """

    def __init__(self, hbar, N_cell, max_order, L=10,
                 dirichlet_bound_conds=True):
        """
        """
        x_grid = np.linspace(-L, L, N_cell+1)
        FEM.__init__(self, hbar, max_order, x_grid,
                     dirichlet_bound_conds=dirichlet_bound_conds)
    def V_fct(self, x):
        """
        """
        return x**2/2.0

    def E(self, n):
        """Analytic Spectrum
        """
        return (n + 0.5) * self.hbar

    def psi(self, x, n=0):
        """Analytic Eigenfunction
        """
        coeff = np.zeros([n+1])
        coeff[-1] = 1
        H_n = Hermite(coeff)
        pre_fac = 1.0 / np.sqrt(2**n * gamma(n+1) * np.sqrt(np.pi))
        return pre_fac * H_n(x) * np.exp(-0.5 * x**2)
