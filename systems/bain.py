#pylint: disable=C0103
"""
Here we define the system according to the potential function

V(q) = 7.5 * q**2 * exp(-q)

This is the most famous benchmark case
"""

from __future__ import (division, print_function)
import numpy as np
from FEM.core.fem import FEM

class Bain(FEM):
    """ H = - hbar*2 \Laplace on the domain x = [0,1]
    """

    def __init__(self, hbar, max_order, N1, L1, N2, L2, theta=0.25*np.pi,
                 dirichlet_bound_conds=True):
        """
        """
        x_l = np.linspace(0.0, L1, N1+1)
        x_r = np.linspace(L1, L2, N2+1)
        x_grid = np.concatenate((x_l, x_r[1:]))
        FEM.__init__(self, hbar, max_order, x_grid, theta=theta)

    def V_fct(self, x):
        """
        """
        return 7.5 * x**2 * np.exp(-x)

    #def order_evecs(self):
        #""" here we order the states R according to their overlap with psi
        #"""
        #i_sort = np.argsort(-self.evals.imag)
        #self.evals = self.evals[i_sort]
        #self.R = self.R[:,i_sort]
        #self.L = self.L[:,i_sort]

    #def embedd_evecs(self):
        #"""
        #"""
        #tmp = np.zeros((self.dim, self.dim-1), dtype=complex)
        #tmp[1:,:] = self.R
        #self.R = tmp

    ## #################################################################
    ## numerical stuff
    ## #################################################################
    #def normalize_evecs(self):
        #""" Reduces the states to the points on the FEM grid """
        #dq = self.x_grid[1:] - self.x_grid[:-1]
        ## normalization of right eigenvectors
        #for l in xrange(self.dim-1):
            ## l-th state on the element grid
            #psi_l = self.R[:,l][(self.max_order-1):][::self.max_order]
            #f = np.zeros(len(psi_l)+1, dtype=float)
            #f[1:] = abs(psi_l)**2
            #F = (f[1:] + f[:-1])/2.0
            #norm_l = np.sum(F * dq)
            #self.R[:,l] *= 1.0 / np.sqrt(norm_l)

    #def impose_boundary_condition(self):
        #"""
        #"""
        #self.T = self.T[1:,1:]
        #self.V = self.V[1:,1:]
        #self.O = self.O[1:,1:]
        #self.H = self.H[1:,1:]
