#pylint: disable=C0103
"""
This is the FEM mother class.

From here Kicked and Autonomous are derived.

The hard and important work is performed by FEM_Cell_Legendre.

FEM_Cell Legendre is imported as a module
"""

from __future__ import (division, print_function)
import numpy as np
from scipy.linalg import eig
from FEM.core.fem_cell_legendre import FEM_Cell_Legendre

class FEM(object):
    """ H = - hbar*2 \Laplace on the domain x = [0,1]
    """
    def __init__(self, hbar, max_order, x_grid, theta=0.0,
                 dirichlet_bound_conds=True):
        """
        """
        self.hbar = hbar
        self.max_order = max_order
        self.FEM = FEM_Cell_Legendre(max_order=max_order)
        self.x_grid = x_grid
        self.N_cell = len(x_grid) - 1
        # angle for complex scaling
        self.theta = theta
        self.dim = max_order * self.N_cell + 1
        self.dirichlet_bound_conds = dirichlet_bound_conds

        # to be filled by setup
        self.O = None
        self.T = None
        self.V = None
        self.H = None

    # the scaling path ----------------------------------------------------
    def scaling_path(self, x):
        """ this function contains the scaling path
        """
        return x * np.exp(1j*self.theta)

    # Kinetic Energy and Potential ----------------------------------------
    def T_fct(self, x):
        """ kinetic energy
        """
        raise x**2 / 2.0

    def T_fct_theta(self, x):
        """ scaled kinetic energy
        """
        return self.T_fct(x * np.exp(-1j*self.theta))

    def V_fct(self, x):
        """ Potential as defined for the unscaled system
        """
        raise NotImplementedError

    def V_fct_theta(self, x):
        """ Potential after including the complex scaling Transformation
        """
        return self.V_fct(self.scaling_path(x))

    # #################################################################
    # q space representation of wave functions
    # #################################################################
    def psi_from_coeff(self, x, m=0, coeff=None):
        """ This constructs the right wavefunction from coefficients
        """
        psi = np.zeros(len(x), dtype=complex)
        if coeff is None:
            coeff = self.R[:,m] # read out coeffs
        # iterate through the grid
        for i in range(self.N_cell):
            smin = self.x_grid[i]
            smax = self.x_grid[i+1]
            if i == 0:
                indeces = np.where((smin<=x)*(x<=smax))
            else:
                indeces = np.where((smin<x)*(x<=smax))
            for m in range(self.max_order + 1):
                Xsi_m = self.FEM.Xsi(m, domain=[smin, smax])
                psi[indeces] += coeff[i*self.max_order+m] * Xsi_m(x[indeces])
        return psi

    def coeff_from_psi(self, psi, degree=None):
        """ This constructs the coefficients for the right wave function
        """
        coeff = np.zeros(self.dim, dtype=complex)
        coeff[::self.max_order] = psi(self.x_grid)
        # iterating through the cell
        for i in range(self.N_cell):
            smin = self.x_grid[i]
            smax = self.x_grid[i+1]
            for m in range(1, self.max_order):
                coeff[i*self.max_order+m] =\
                    self.FEM.project_psi_on_P(psi, m+1, domain=[smin,smax],
                                              degree=degree)
        return coeff

    # #################################################################
    # setup of matrices
    # #################################################################
    def setup_O(self):
        """ Sets up the Overlapmatrix of the finite element representation.
        """
        max_order = self.max_order
        O = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.N_cell):
            smin = self.x_grid[i]
            smax = self.x_grid[i+1]
            O_mat = self.FEM.Overlap_matrix(smin=smin, smax=smax)
            O[i*max_order:(i+1)*max_order+1,i*max_order:(i+1)*max_order+1] +=\
                O_mat
        self.O = O

    # #################################################################
    # setup of matrices (for autonomous problems)
    # #################################################################
    def setup_T(self):
        """ Sets up the finite element representation of the overlap matrix.
        """
        max_order = self.max_order
        T = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.N_cell):
            smin = self.x_grid[i]
            smax = self.x_grid[i+1]
            L = self.FEM.Laplace_matrix(smin=smin, smax=smax)
            L *= np.exp(-1j*2.0*self.theta)
            # put the resulting matrix
            T[i*max_order:(i+1)*max_order+1,i*max_order:(i+1)*max_order+1] +=\
                -(0.5 * self.hbar**2) * L
        self.T = T

    def setup_V(self, degree=None, nonzero=True):
        """ Sets up the finite element representation of the overlap matrix.
        """
        max_order = self.max_order
        V = np.zeros((self.dim, self.dim), dtype=complex)
        if nonzero:
            for i in range(self.N_cell):
                smin = self.x_grid[i]
                smax = self.x_grid[i+1]
                Pot_mat = self.FEM.Potential_matrix(self.V_fct_theta,
                                                    smin=smin, smax=smax,
                                                    degree=degree)
                V[i*max_order:(i+1)*max_order+1,
                  i*max_order:(i+1)*max_order+1] += Pot_mat
        self.V = V

    def setup_H(self):
        """ Full matrix representation
        """
        self.H = self.T + self.V

    # #################################################################
    # numerical stuff
    # #################################################################
    def compute_evecs(self):
        """
        """
        if self.dirichlet_bound_conds:
            O = self.O[1:-1,1:-1]
            H = self.H[1:-1,1:-1]
            evals, L, R = eig(H, b=O, left=True, right=True)
            # embedd
            self.evals = np.concatenate((evals, np.array([0.0, 0.0])))
            self.R = np.zeros((self.dim, self.dim), dtype=complex)
            self.R[1:-1,:-2] = R
            self.L = np.zeros((self.dim, self.dim), dtype=complex)
            self.R[1:-1,:-2] = L
        else:
            self.evals, self.L, self.R =\
                eig(self.H, b=self.O, left=True, right=True)

    def order_evecs(self):
        """ here we order the states R according to their overlap with psi
        """
        if self.dirichlet_bound_conds:
            i_sort = np.argsort(self.evals[:-2])
            i_sort = np.concatenate((i_sort,np.array([self.dim-2,self.dim-1])))
        else:
            i_sort = np.argsort(self.evals)
        self.evals = self.evals[i_sort]
        self.R = self.R[:,i_sort]
        self.L = self.L[:,i_sort]

    def normalize_R(self):
        """ normalize the absolut square value of right eigenvectors R
        Should be |Psi|^(2) = coeff^{\dag} O coeff
        """
        for l in xrange(len(self.R[0,:])):
            # l-th state on the element grid
            vec = self.R[:,l]
            norm_l = np.dot(vec.conjugate(), np.dot(self.O, vec))
            if abs(norm_l) > 10**(-15):
                self.R[:,l] *= 1.0 / np.sqrt(norm_l)

    # #################################################################
    # normalization for coefficients
    # #################################################################
    def normalize_RL(self, R, L):
        """ Normalizes a set of left and right eigenvectors such that:
        - right vectors are normalized
        - the scalar product <L_j,R_i> = delta_{j,i}
        """
        N = len(R[0,:])
        # 1) normalization of right eigenvectors
        for l in xrange(N):
            norm = np.dot(R[:,l].conjugate(), R[:,l])
            R[:,l] = R[:,l] / np.sqrt(norm)

        # 2) normalization of left eigenvectors to <L_k|R_l> = \delta_{l,k}
        for j in xrange(N):
            norm = np.dot(L[j,:], R[:,j])
            L[j,:] = L[j,:] / norm
        return R, L

    def test_normalization_RL(self, R, L):
        """ Check results of normalize_RL
        """
        N = len(R[0,:])

        # 1) check orthogonormality between left and right eigenvectors
        tmp = np.dot(L, R)
        err2 = tmp - np.diag(np.ones(N))
        if abs(err2).max() > 10**(-12):
            print("Biorthogonality errors of left and right eigenvectors",\
                   abs(err2).max())

        # 1) check completeness relation R L = 1
        unity = np.dot(R, L)
        diff = abs(unity - np.diag(np.ones(N))).max()
        if diff > 10**(-12):
            print('RL decomposition correct up to', diff)

    def normalize_RL_to_O(self, R, L):
        """ Normalizes a set of left and right eigenvectors such that:
        - right vectors are normalized
        - the scalar product <L_j,R_i> = delta_{j,i}
        """
        N = len(R[0,:])
        # 1) normalization of right eigenvectors
        for l in xrange(N):
            norm = np.dot(R[:,l].conjugate(), R[:,l])
            R[:,l] = R[:,l] / np.sqrt(norm)

        # 2) normalization of left eigenvectors to <L_k|R_l> = \delta_{l,k}
        for j in xrange(N):
            norm = np.dot(L[j,:], np.dot(self.O, R[:,j]))
            L[j,:] = L[j,:] / norm
        return R, L

    def test_normalization_RL_to_O(self, R, L):
        """ Check results of normalize_RL
        """
        N = len(R[0,:])

        # 1) check orthogonormality between left and right eigenvectors
        tmp = np.dot(L, np.dot(self.O, R))
        err2 = tmp - np.diag(np.ones(N))
        if abs(err2).max() > 10**(-12):
            print("Biorthogonality errors of left and right eigenvectors",\
                   abs(err2).max())

        # 1) check completeness relation R L O = 1
        unity = np.dot(R,np.dot(L, self.O))
        diff = abs(unity - np.diag(np.ones(N))).max()
        if diff > 10**(-12):
            print('RLO decomposition correct up to', diff)

        # 1) check completeness relation R L O = 1
        unity = np.dot(np.dot(self.O, R), L)
        diff = abs(unity - np.diag(np.ones(N))).max()
        if diff > 10**(-12):
            print('ORL decomposition correct up to', diff)