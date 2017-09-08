#pylint: disable=C0103
"""
In this section we provide the code which sets up Matrix elements of
a time dependent schroedinger equation on a single FEM cell. The Schroedinger
equation is:

i hbar dt psi = [-heff^2/2 * dx^2 + V(x)] psi

The FEM code follows the idea of Elander.

In one cell [s_i,s_{i+1}] the function is described by X_{n}(s) such that
(i) X_{0}(s_{i}) = 1 and X_{1}(s_{i+1}) = 1
while for other n
(ii) X_{n}(s_{i}) = X_{n}(s_{i+1}) = 0

The main quantities computed here, are the Overlap Matrix, the Laplace Matrix
and the Potential Matrix. While the first can easily be computed analytically,
the latter are computed Quadrature integration.

The functions X are based on Legendre-Polynomials as follows:

polynomial of max_order

- X_{0} = (P_{0} - P_{1})/2
- X_{n} = P_{n+1} - P_{(n+1)%2}
- X_{max_order} = (P_{0} + P_{1})/2

The test's for FEM cell legendre are in Test_FEM_Cell_Legendre

"""

from __future__ import (division, print_function)
import numpy as np
from numpy.polynomial import Legendre as L

from matplotlib import pylab as plt
from time import time

from FEM.core.legendre_quad import LegendreQuad
from FEM.core.legendre_quad_2d import LegendreQuad2d

class FEM_Cell_Legendre(object):
    """
    """
    def __init__(self, max_order=6):
        """
        """
        self.max_order=max_order  # Maximal order of Legendre Polynomial
        self.Int_1d = LegendreQuad()
        self.Int_2d = LegendreQuad2d()

    # legendre polynomial and projection onto it in a domain ----------------
    def P(self, n, domain=[-1,1], window=[-1,1]):
        """ returns the hermite Polynomial of order n
        """
        coef = np.zeros(n+1, dtype=int)
        coef[-1]  = 1
        return L(coef, domain=domain, window=window)

    def project_psi_on_P(self, psi, n, domain=[-1,1], degree=None):
        """
        """
        P = self.P(n, domain=domain)
        fct = lambda x: P(x) * psi(x)
        if degree is None:
            degree = max(self.max_order + 10, 30)
        Pn2_inv = (n + 0.5) * 2.0 / (domain[1]-domain[0])
        return self.Int_1d.Integrate(fct, domain, degree) * Pn2_inv

    # the basis functions used on the finite element set -------------------
    def coef(self, m):
        """
        """
        if m == 0:
            return np.array([0.5,-0.5])
        elif m == self.max_order:
            return np.array([0.5, 0.5])
        else:
            m += 1
            coef = np.zeros(m+1, dtype=int)
            coef[-1]  = 1
            coef[m%2] = -1
            return coef

    def Xsi(self, m, domain=[-1,1], window=[-1,1]):
        """ Xsi(x) = P_m(x) - P_{m%2}(x)
        """
        return L(self.coef(m), domain=domain, window=window)

    def dXsi(self, m, domain=[-1,1], window=[-1,1], n=1):
        """ d^(n)Xis = d^(n)P_m(x) - d^(n)P_{m%2}(x)
        """
        return L(self.coef(m), domain=domain, window=window).deriv(n)

    # Finite Element Matrix representation of Overlap, Laplace, and Potential
    def Overlap_matrix(self, smin=-1.0, smax=1.0):
        """ This determines the overlap \int_{smin}^{smax} X_{n} X_{m}
        for functions with domain adjusted to [smin, smax]
        """
        Overlap_matrix = np.zeros((self.max_order + 1, self.max_order + 1),
                                  dtype=complex)
        # set up the diagonal terms first
        for i in range(self.max_order + 1):
            if i == 0:
                Overlap_matrix[i,i] = 2.0 / 3.0
            elif i == self.max_order:
                Overlap_matrix[i,i] = 2.0 / 3.0
            else:
                m = i+1
                Overlap_matrix[i,i] = 2.0 / (2.0 * m + 1)\
                                    + 2.0 / (2.0 * (m%2) + 1.0)
        # set up the upper right and lower left corner
        Overlap_matrix[0,-1] = 1.0/3.0
        Overlap_matrix[-1,0] = 1.0/3.0
        # set up the upper row and the first column
        for i in range(self.max_order-1):
            if (i%2) == 0:
                Overlap_matrix[0,1+i] = -1.0
                Overlap_matrix[1+i,0] = -1.0
            else:
                Overlap_matrix[0,1+i] = 1.0/3.0
                Overlap_matrix[1+i,0] = 1.0/3.0
        # set up the last column and the lower row
        for i in range(self.max_order-1):
            if (i%2) == 0:
                Overlap_matrix[i+1,-1] = -1.0
                Overlap_matrix[-1,i+1] = -1.0
            else:
                Overlap_matrix[-1,i+1] = -1.0/3.0
                Overlap_matrix[i+1,-1] = -1.0/3.0
        for i in range(self.max_order):
            for j in range(self.max_order):
                if ((i==0) or (i==self.max_order) or
                    (j==0) or (j==self.max_order) or
                    (i==j)):
                    pass
                else:
                    if (i%2) == (j%2):
                        if (i+1)%2 == 0:
                            Overlap_matrix[i,j] = 2.0
                        else:
                            Overlap_matrix[i,j] = 2.0/3.0

        # scaling the domain of the finite element, leads to a modification
        # of the above overlap matrix
        Overlap_matrix *= (smax - smin) / 2.0

        return Overlap_matrix

    def Laplace_matrix(self, smin=-1.0, smax=1.0):
        """ This determines the overlap \int_{smin}^{smax} dX_{n} dX_{m}
        for functions with domain adjusted to [smin, smax]
        """
        Laplace_matrix = np.zeros((self.max_order + 1, self.max_order + 1),
                                  dtype=complex)
        # set up top left and lower right
        Laplace_matrix[0,0] = 1.0/2.0
        Laplace_matrix[-1,-1] = 1.0/2.0
        # set up the upper right and lower left corner
        Laplace_matrix[0,-1] = -1.0/2.0
        Laplace_matrix[-1,0] = -1.0/2.0
        # set up the main body
        for i in range(self.max_order-1):
            i += 1
            for j in range(self.max_order-1):
                j += 1
                if (i%2) == (j%2):
                    m_min = min(i, j) + 1
                    Laplace_matrix[i,j] = m_min * (m_min + 1)
                    if i%2 == 0:
                        Laplace_matrix[i,j] += -2

        # scaling the domain of the finite element, leads to a modification
        # of the above Laplace matrix
        Laplace_matrix *= 2.0 / (smax - smin)

        return -Laplace_matrix

    # ----------------------------------------------------------------------
    # Matrix computed from 1D integration, e.g., potential
    def Potential_matrix(self, V, smin=-1.0, smax=1.0, degree=None):
        """ This determines the overlap \int_{smin}^{smax} X_{n} V X_{m}
        for functions with domain adjusted to [smin, smax]
        """
        # Use Legendre Quadrature integration
        Potential_matrix = np.zeros((self.max_order + 1, self.max_order + 1),
                                    dtype=complex)
        # iterate through the matrix and exploit symmetry
        for m in range(self.max_order + 1):
            Xsi_m = self.Xsi(m, domain=[smin, smax])
            for n in range(m + 1):
                Xsi_n = self.Xsi(n, domain=[smin, smax])
                # define the integran f = Xsi_m * V * Xsi_n
                f = lambda x: Xsi_m(x) * V(x) * Xsi_n(x)
                # Use Legendre Quadrature integration
                if degree is None:
                    degree = max(self.max_order + 10, 30)
                    print('We set the degree as:', degree)
                V_mn = self.Int_1d.Integrate(f, [smin, smax], degree)
                Potential_matrix[m,n] = V_mn
                # exploit symmetry
                if n != m:
                    Potential_matrix[n,m] = V_mn
        return Potential_matrix

    # ----------------------------------------------------------------------
    # Matrix computed from 2D integration, e.g., propagator
    def K_matrix(self, K, smin=-1.0, smax=1.0, etamin=-1.0, etamax=1.0,
                 degree=None):
        """ \int_{xsimin}^{xsismax} dxsi \int_{smin}^{smax} ds
            X_{n}(xsi) K(xsi, s)  X_{m}(s)
        """
        K_matrix = np.zeros((self.max_order + 1, self.max_order + 1),
                            dtype=complex)
        # iterate through the matrix and exploit symmetry
        for m in range(self.max_order + 1):
            Xsi_m = self.Xsi(m, domain=[etamin, etamax])
            for n in range(self.max_order + 1):
                Xsi_n = self.Xsi(n, domain=[smin, smax])
                # define the integran f = Xsi_m(xsi) * K(xsi,s) * Xsi_n(s)
                f = lambda y, x : Xsi_m(y) * K(y, x) * Xsi_n(x)
                # Use Legendre Quadrature integration
                if degree is None:
                    degree = max(self.max_order + 10, 30)
                    print('We set the degree as:', degree)
                K_mn = self.Int_2d.Integrate(f, [etamin, etamax], [smin, smax],
                                             degree, degree)
                K_matrix[m,n] = K_mn
        return K_matrix

# ########################################################################
# the code below is capable of determining the matrix elements numerically
# it is very slow and should not be used. Its only purpose is testing of the
# code above

    def Overlap_matrix_numeric(self, smin=-1.0, smax=1.0):
        """ Same as Overlap Matrix but numerical, i.e., slow!
        """
        Overlap_matrix = np.zeros((self.max_order + 1, self.max_order + 1),
                                  dtype=complex)
        for i in range(self.max_order + 1):
            Xsi_i = self.Xsi(i, domain=[smin, smax])
            for l in range(self.max_order + 1):
                Xsi_l = self.Xsi(l, domain=[smin, smax])
                Overlap_matrix[i,l] = (smax-smin) * (Xsi_i * Xsi_l).coef[0]
        return Overlap_matrix

    def Laplace_matrix_numeric(self, smin=-1.0, smax=1.0):
        """ Same as Laplace Matrix but numerical, i.e., slow!
        """
        Laplace_matrix = np.zeros((self.max_order + 1, self.max_order + 1),
                                  dtype=complex)
        for i in range(self.max_order + 1):
            dXsi_i = self.dXsi(i, domain=[smin, smax])
            for l in range(self.max_order + 1):
                dXsi_l = self.dXsi(l, domain=[smin, smax])
                Laplace_matrix[i,l] = (smax-smin) * (dXsi_i * dXsi_l).coef[0]
        return -Laplace_matrix

