#pylint: disable=C0103
"""
This is the FEM mother class for kicked systems along an exterior scaling path
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt
from scipy.linalg import eig, inv, eigh
from FEM.core.fem import FEM
from scipy.special import erf

class FEMKicked(FEM):
    """ H = - hbar*2 \Laplace on the domain x = [0,1]
    """
    def __init__(self, hbar, max_order, x_grid, theta=0.0,
                 dirichlet_bound_conds=True,
                 exterior_scaling_width=0.2,
                 exterior_x0=2.0):
        """
        """
        FEM.__init__(self, hbar, max_order, x_grid, theta=theta,
                     dirichlet_bound_conds=dirichlet_bound_conds)

        self.exterior_scaling_width=exterior_scaling_width
        self.exterior_x0=exterior_x0
        # additional matrices
        self.U = None
        self.UT = None
        self.UV_half = None

    # #####################################################
    # scaling path for smooth exterior scaling

    def Fm1(self, x):
        """ a normalized Gaussian
        """
        return np.exp(-x**2)/np.sqrt(2.0)

    def F0(self, x):
        """ smooth step function (integral of Fm1)
        """
        return (1.0 + erf(x)) / 2.0

    def F1(self, x):
        """ smooth interpolation between a:
        -- a function equal to zero to the left
        -- a function which behaves like 'x', i.e., linearly to the right
        """
        return x * self.F0(x) + self.Fm1(x) / 2.0

    def scaling_path(self, x):
        """ this function contains the scaling path
        """
        w  = self.exterior_scaling_width
        yr = (x - self.exterior_x0) / w
        yl =(-x - self.exterior_x0) / w
        ret = x \
            + (np.exp(1j*self.theta) - 1.0) * (w * self.F1(yr))\
            - (np.exp(1j*self.theta) - 1.0) * (w * self.F1(yl))
        return ret

    def der_scaling_path(self, x):
        """ this is the parameter derivative of the scaling path
        """
        w  = self.exterior_scaling_width
        yr = (x - self.exterior_x0) / w
        yl =(-x - self.exterior_x0) / w
        ret = x \
            + (np.exp(1j*self.theta) - 1.0) * self.F0(yr)\
            + (np.exp(1j*self.theta) - 1.0) * self.F0(yl)
        return ret

    # this defines the propagator functions -------------------------
    def exp_V_half(self, x):
        """ position space propagator of the half Kick potential
        """
        return np.exp(-1j * self.V_fct_theta(x) / (2.0 * self.hbar))

    def K_free(self, y, x, dt=1.0):
        """ this is the complex scaled propagator function without kick
        """
        alpha = np.exp(-1j*np.pi/4.0) / np.sqrt(2.0*self.hbar*dt)

        x_ = self.scaling_path(x)
        y_ = self.scaling_path(y)

        return alpha * self.der_scaling_path(x) / np.sqrt(np.pi) \
                     * np.exp(-alpha**2 * (x_-y_)**2)

    def K_full(self, y, x, dt=1.0):
        """ this is the complex scaled propagator function without kick
        """
        return self.exp_V_half(y)*self.K_free(y,x,dt=1.0)*self.exp_V_half(x)

    # matrix computation routines ------------------------------------
    def compute_evecs(self):
        """
        """
        if self.dirichlet_bound_conds:
            O = self.O[1:-1,1:-1]
            U = self.U[1:-1,1:-1]
            evals, L, R = eig(U, b=O, left=True, right=True)
            # embedd the results
            self.evals = np.concatenate((evals, np.array([0.0, 0.0])))
            self.R = np.zeros((self.dim, self.dim), dtype=complex)
            self.R[1:-1,:-2] = R
            self.L = np.zeros((self.dim, self.dim), dtype=complex)
            self.R[1:-1,:-2] = L
        else:
            self.evals, self.L, self.R = eig(self.U, b=self.O, left=True,
                                             right=True)

    def order_evecs(self):
        """ here we order the states R according to their decay rate
        """
        i_sort = np.argsort(-abs(self.evals))
        self.evals = self.evals[i_sort]
        self.R = self.R[:,i_sort]
        self.L = self.L[:,i_sort]

    # #####################################################################
    # generateU ------------------------------------------------------
    def generateU(self, method=0, degree2D=None, eps=10**(-15)):
        """ this method generates the time evolution operator
        """
        # method 0: same as setup_U_int
        # disadvantage: slow
        # advantage: no inversion of O needed
        # for the moment no fast alternative exists
        if method == 0:
            self.setup_U_int(degree=degree2D, eps=eps)

    # #####################################################################
    # Matrix setups
    # #####################################################################

    # from propagator integration -----------------------------------------
    def setup_U_int(self, degree=None, eps=10**(-15)):
        """ Sets up the Propagator Matrix of the finite element representation.
        """
        max_order = self.max_order
        U = np.zeros((self.dim, self.dim), dtype=complex)
        # iterating through the FEM cells for 2-dimensional integrals
        for i in range(self.N_cell):
            print(i+1, "/", self.N_cell)
            smin = self.x_grid[i]
            smax = self.x_grid[i+1]
            for j in range(self.N_cell):
                etamin = self.x_grid[j]
                etamax = self.x_grid[j+1]
                #full propagator -------------------------------------
                if max(abs(self.K_full(etamin, smin)),
                       abs(self.K_full(etamin, smax)),
                       abs(self.K_full(etamax, smin)),
                       abs(self.K_full(etamax, smax))) < eps:
                    pass
                else:
                    U[j*max_order:(j+1)*max_order+1,
                        i*max_order:(i+1)*max_order+1] +=\
                    self.FEM.K_matrix(self.K_full, smin=smin, smax=smax,
                                        etamin=etamin, etamax=etamax,
                                        degree=degree)
        self.U = U

    # ####################################################################
    # Note: These routines may become useful later. For the moment they are
    # just left in the code. But they may be exploited for testing

    def setup_UV_half_int(self, degree=None):
        """ Sets up the half-Kick Matrix of the FE representation
        """
        max_order = self.max_order
        UV_half = np.zeros((self.dim, self.dim), dtype=complex)
        # iterating through the FEM cells for one dimensional integrals
        for i in range(self.N_cell):
            smin = self.x_grid[i]
            smax = self.x_grid[i+1]
            UV_mat = self.FEM.Potential_matrix(self.exp_V_half, smin=smin,
                                               smax=smax, degree=degree)
            UV_half[i*max_order:(i+1)*max_order+1,
                    i*max_order:(i+1)*max_order+1] += UV_mat
        self.UV_half = UV_half

    def setup_UT_int(self, degree=None, eps=10**(-15)):
        """ Sets up the complex scaled Propagator of the free particle
        in the finite element representation.
        """
        raise NotImplementedError

        max_order = self.max_order
        UT = np.zeros((self.dim, self.dim), dtype=complex)
        # iterating through the FEM cells for 2-dimensional integrals
        for i in range(self.N_cell):
            smin = self.x_grid[i]
            smax = self.x_grid[i+1]
            for j in range(self.N_cell):
                etamin = self.x_grid[j]
                etamax = self.x_grid[j+1]
                ##free propagator -------------------------------------
                #if max(abs(self.K_free(etamin, smin)),
                       #abs(self.K_free(etamin, smax)),
                       #abs(self.K_free(etamax, smin)),
                       #abs(self.K_free(etamax, smax))) < eps:
                    #pass
                #else:
                UT[j*max_order:(j+1)*max_order+1,
                    i*max_order:(i+1)*max_order+1] +=\
                self.FEM.K_matrix(self.K_free, smin=smin, smax=smax,
                                    etamin=etamin, etamax=etamax,
                                    degree=degree)
        self.UT = UT