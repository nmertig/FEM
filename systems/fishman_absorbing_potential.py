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
    """ H = - hbar*2 \Laplace on the domain x = [0,1]
    """
    def __init__(self, kappa, hbar, max_order, N, x_max,
                 qabs, amp, width, poly_order,
                 dirichlet_bound_conds=False):
        """
        """
        self.kappa = kappa
        self.x_max = x_max
        x_grid = np.linspace(-x_max, x_max, N)
        FEMKicked.__init__(self, hbar, max_order, x_grid, theta=0.0,
                           dirichlet_bound_conds=dirichlet_bound_conds)
        # parameters of the absorbing potential
        self.qabs = qabs
        self.amp = amp
        self.width = width
        self.poly_order = poly_order

    def set_amplitude(self, eps=10**(-15)):
        """ sets the amplitude of the absorbing potential, such that the
            corresponding projector reaches 'eps' (usually set to be numerical
            accuracy) at the upper grid boundary
        """
        print("amplitude was", self.amp)
        self.amp =\
            - (2.0 * self.hbar * self.amp / self.F(self.x_max)) * np.log(eps)
        print("reset amplitude to", self.amp)


    def V_fct(self, x):
        """
        """
        return -self.kappa * np.exp(- 8.0 * x**2 ) / 16.0 - 1j*self.F(x)

    def F(self, x):
        """ symmetrized version of F_
        """
        return self.F_(x) + self.F_(-x)

    def F_(self, x):
        """ integrals of the error-function like step function
        """
        x = (x-self.qabs) / self.width
        pre = self.width ** self.poly_order
        if self.poly_order == 0:
            F_ = (1.0 + erf(x)) / 2.0
        if self.poly_order == 1:
            F_ =  (x * (1.0 + erf(x)) + np.exp(-x**2)/np.sqrt(np.pi)) / 2.0
        if self.poly_order == 2:
            F_ = ((2.0 * x**2 + 1.0) * (1.0 + erf(x)) +
                   2.0 * x * np.exp(-x**2)/np.sqrt(np.pi)) / 4.0
        if self.poly_order == 3:
            F_ = ((2.0 * x**3 + 3.0 * x) * (1.0 + erf(x)) +
                  (2.0 + 2.0 * x**2) * np.exp(-x**2)/np.sqrt(np.pi)) / 4.0
        return pre * self.amp * F_

    def projector_characteristics(self):
            # ------------------------------------------------------------
            q_lattice = self.x_grid
            proj_diag = np.exp(-self.F(q_lattice) / (2.0 * self.hbar))
            # extract effective range
            q_ = q_lattice[np.where(proj_diag>(1.0-10**(-15)))]
            if len(q_)>1:
                print("state unaffected in", min(q_), max(q_))
            else:
                print("maximal projector point is", max(proj_diag))
            # ------------------------------------------------------------
            # absorbing range
            q_ = q_lattice[np.where(proj_diag>10**(-15))]
            if len(q_)>1:
                print("state absorbed beyond", min(q_), max(q_))
            else:
                print("states transmit accros boundary")

    def potential_oscillation_length(self):
        """
        """
        h = 2.0 * np.pi * self.hbar
        return h * 8.0 * np.exp(0.5)/self.kappa