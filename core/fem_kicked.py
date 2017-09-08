#pylint: disable=C0103
"""
This is the FEM mother class for kicked systems
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt
from scipy.linalg import eig, inv, eigh
from FEM.core.fem import FEM
from FEM.extras.runge_kutta_2 import RungeKutta

class FEMKicked(FEM):
    """ H = - hbar*2 \Laplace on the domain x = [0,1]
    """
    def __init__(self, hbar, max_order, x_grid, theta=0.0,
                 dirichlet_bound_conds=True,
                 exterior_scaling=False,
                 exterior_scaling_xmin=-1.0,
                 exterior_scaling_xmax=1.0,
                 exterior_scaling_mask=None):
        """
        """
        FEM.__init__(self, hbar, max_order, x_grid, theta=theta,
                     dirichlet_bound_conds=dirichlet_bound_conds,
                     exterior_scaling=exterior_scaling,
                     exterior_scaling_xmin=exterior_scaling_xmin,
                     exterior_scaling_xmax=exterior_scaling_xmax,
                     exterior_scaling_mask=exterior_scaling_mask)
        # additional matrices
        self.U = None
        self.UT = None
        self.UV_half = None

    # this defines the propagator functions -------------------------
    def exp_V_half(self, x):
        """ position space propagator of the half Kick potential
        """
        return np.exp(-1j * self.V_fct_theta(x) / (2.0 * self.hbar))

    def K_free(self, y, x, dt=1.0):
        """ this is the complex scaled propagator function without kick
        """
        alpha = np.exp(1j*(self.theta-np.pi/4.0)) / np.sqrt(2.0*self.hbar*dt)
        return alpha * np.exp(- alpha**2 * (x-y)**2) / np.sqrt(np.pi)

    def K_full(self, y, x, dt=1.0):
        """ this is the complex scaled propagator function without kick
        """
        return self.exp_V_half(y)*self.K_free(y,x,dt=1.0)*self.exp_V_half(x)

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

    # generateU ------------------------------------------------------
    def generateU(self, method=0, degree=None, degree2D=None, eps=10**(-15),
                  Oinv_fast=False, Oinv_max_eval=None, Oinv_show_modes=False,
                  show_matrix=False):
        """ this method generates the time evolution operator
        """
        mask = np.ones((self.dim, self.dim))
        if self.dirichlet_bound_conds:
            mask[0,:]  = 0.0
            mask[-1,:] = 0.0
            mask[:,0]  = 0.0
            mask[:,-1] = 0.0

        # method 0: set up from propagation parts, i.e. UV_half UT UV_half
        # advantage: much faster then full integration
        # disadvantage: requires the inverse of O
        if method == 0:
            self.setup_O()
            self.setup_Oinv(fast=Oinv_fast, max_eval=Oinv_max_eval,
                            show_modes=Oinv_show_modes)
            self.setup_UT_int_fast(degree=degree2D, eps=10**(-15))
            self.setup_UV_half_int(degree=degree)

            # prepare the full propagation matrix ----------
            UV_half = np.dot(self.Oinv*mask, self.UV_half*mask)
            UT = np.dot(self.Oinv*mask, self.UT*mask)
            self.U = np.dot(self.UV_half*mask, np.dot(UT, UV_half))

            # short check of the boundary conditions
            if show_matrix:
                fig = plt.figure(1)
                ax1 = fig.add_subplot(131)
                ax1.imshow(abs(UV_half))
                ax2 = fig.add_subplot(132)
                ax2.imshow(abs(UT))
                ax3 = fig.add_subplot(133)
                ax3.imshow(abs(mask))
                plt.show()

        # method 1: same as setup_U_int
        # disadvantage: slow
        # advantage: no inversion of O needed
        if method == 1:
            self.setup_U_int()

    def setup_Oinv(self, fast=False, max_eval=None, show_modes=False):
        """ Inverts the Overlapmatrix of the finite element representation.
        """
        if fast: # This is a very quick solution but not stable
            # one known problem is the occurence of modes localizing on the left
            # and right edge respectively
            self.Oinv = inv(self.O)

        else: # hence it is safer to diagonalize the matrix and determine
            # the modes which need to be excluded explicitely
            # an important distinction is the case with and without
            # dirichlet boundary conditions
            if not self.dirichlet_bound_conds:
                # Note: for the non exterior case O is symmetric and we could
                # use eigh, this is not done here
                evals, R, L = eig(self.O, left=True, right=True)
                # sort
                i_sort = np.argsort(abs(evals))[::-1]
                evals = evals[i_sort]
                R = R[:,i_sort]
                L = L[:,i_sort]
                # swap the state index
                L = L.transpose().conjugate()
                R, L = self.normalize_RL(R, L)
                self.test_normalization_RL(R, L)
                tmpO = np.dot(R, np.dot(np.diag(evals), L))
                print('-----------------------')
                print('representation quality of O', abs(tmpO-self.O).max())

                # set up the inverse matrix -----------------------------
                evals_inverse = np.zeros(self.dim, dtype=complex)
                # excluding the two leading edge states
                evals_inverse[:-2] = 1.0/evals[:-2]
                self.Oinv = np.dot(R, np.dot(np.diag(evals_inverse), L))

            else: # this takes care of the inversion with dirichlet boundconds
                O = self.O[1:-1,1:-1]
                # Note: for the non exterior case O is symmetric and we could
                # use eigh, this is not done here
                evals, R, L = eig(O, left=True, right=True)
                # sort
                i_sort = np.argsort(abs(evals))[::-1]
                evals = evals[i_sort]
                R = R[:,i_sort]
                L = L[:,i_sort]
                # swap the state index
                L = L.transpose().conjugate()
                # Note: for the non exterior case O is symmetric and we could
                # use eigh, this is not done here
                R, L = self.normalize_RL(R, L)
                self.test_normalization_RL(R, L)
                tmpO = np.dot(R, np.dot(np.diag(evals), L))
                print('-----------------------')
                print('representation quality of O', abs(tmpO-O).max())

                # set up the inverse matrix -----------------------------
                evals_inverse = np.zeros(self.dim)
                # excluding the two leading edge states
                evals_inverse = 1.0/evals
                Oinv = np.dot(R, np.dot(np.diag(evals_inverse), L))
                # embedd this
                self.Oinv = np.zeros((self.dim, self.dim), dtype=complex)
                self.Oinv[1:-1,1:-1] = Oinv


        # check the exclusion process
        if show_modes:
            if self.dirichlet_bound_conds: # embedding of the results
                evals_inverse = np.concatenate((evals_inverse,
                                                np.array([0.0,0.0])))
                evals = np.concatenate((evals, np.array([0.0,0.0])))
                R_ = np.zeros((self.dim, self.dim), dtype=complex)
                R_[1:-1,:-2] = R
                R = R_

            fig = plt.figure(1)
            x_plot = self.x_grid

            ax1 = fig.add_subplot(221)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.plot(np.arange(len(evals_inverse)),
                     abs(evals_inverse), 'ko', label=None)
            ax1.plot(np.arange(len(evals_inverse)),
                     abs(1.0/evals), 'r+', label=None)

            i_state=0
            ax2 = fig.add_subplot(222)
            ax2.set_xscale('linear')
            ax2.set_yscale('linear')
            ax2.plot(x_plot, abs(R[::self.max_order,i_state]), 'k-')

            ax3 = fig.add_subplot(223)
            ax3.set_xscale('linear')
            ax3.set_yscale('linear')
            ax3.plot(np.arange(self.dim), R[:,i_state].real, 'ko')

            ax4 = fig.add_subplot(224)
            ax4.set_xscale('linear')
            ax4.set_yscale('linear')
            ax4.plot(np.arange(self.dim), R[:,i_state].imag, 'ko')

            def onclick(event):
                x, y = event.xdata, event.ydata
                i_state = int(x)
                ax2.lines = []
                ax2.plot(x_plot, abs(R[::self.max_order,i_state]), 'k-')
                ax3.lines = []
                ax3.plot(np.arange(self.dim), R[:,i_state].real, 'ko')
                ax4.lines = []
                ax4.plot(np.arange(self.dim), R[:,i_state].imag, 'ko')
                plt.draw()

            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

    # #####################################################################
    # Matrix setups
    # #####################################################################
    # from propagator integration -----------------------------------------
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
            # set the phase factor of complex scaling
            if self.exterior_scaling:
                if self.exterior_scaling_mask[i]:
                    UV_mat *= np.exp(1j*self.theta)
            UV_half[i*max_order:(i+1)*max_order+1,
                    i*max_order:(i+1)*max_order+1] += UV_mat
        self.UV_half = UV_half

    def setup_U_int(self, degree=None, eps=10**(-15)):
        """ Sets up the Propagator Matrix of the finite element representation.
        """
        if self.exterior_scaling:
            print("This method does not work for exterior complex scaling")
            raise NotImplementedError
        max_order = self.max_order
        U = np.zeros((self.dim, self.dim), dtype=complex)
        # iterating through the FEM cells for 2-dimensional integrals
        for i in range(self.N_cell):
            print(i)
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

    def setup_UT_int(self, degree=None, eps=10**(-15)):
        """ Sets up the complex scaled Propagator of the free particle
        in the finite element representation.
        """
        if self.exterior_scaling:
            print("This method does not work for exterior complex scaling")
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
                #free propagator -------------------------------------
                if max(abs(self.K_free(etamin, smin)),
                       abs(self.K_free(etamin, smax)),
                       abs(self.K_free(etamax, smin)),
                       abs(self.K_free(etamax, smax))) < eps:
                    pass
                else:
                    UT[j*max_order:(j+1)*max_order+1,
                       i*max_order:(i+1)*max_order+1] +=\
                    self.FEM.K_matrix(self.K_free, smin=smin, smax=smax,
                                        etamin=etamin, etamax=etamax,
                                        degree=degree)
        self.UT = UT

    def setup_UT_int_fast(self, degree=None, eps=10**(-15)):
        """ Sets up the complex scaled Propagator of the free particle
        in the finite element representation.
        """
        if self.exterior_scaling:
            print("This method does not work for exterior complex scaling")
            raise NotImplementedError
        max_order = self.max_order
        UT = np.zeros((self.dim, self.dim), dtype=complex)
        # iterating through the FEM cells for 2-dimensional integrals
        for d in range(self.N_cell):
            etamin = self.x_grid[0]
            etamax = self.x_grid[1]
            smin = self.x_grid[d]
            smax = self.x_grid[d+1]
            #free propagator -------------------------------------
            if max(abs(self.K_free(etamin, smin)),
                    abs(self.K_free(etamin, smax)),
                    abs(self.K_free(etamax, smin)),
                    abs(self.K_free(etamax, smax))) < eps:
                pass
            else:
                K_nm = self.FEM.K_matrix(self.K_free, smin=smin, smax=smax,
                                         etamin=etamin, etamax=etamax,
                                         degree=degree)
                # distribute along the d-th off-diagonal
                for k in range(self.N_cell-d):
                    j = k     # row index
                    i = k + d # column index
                    UT[j*max_order:(j+1)*max_order+1,
                       i*max_order:(i+1)*max_order+1] += K_nm
                    if d > 0: # exploit the symmetry for the lower half
                        UT[i*max_order:(i+1)*max_order+1,
                        j*max_order:(j+1)*max_order+1] += K_nm.transpose()
        self.UT = UT

    # setting up the matrices U, UT, UV via time evolution --------------
    def setup_UT_te(self):
        """
        -- this method sets up UT from time-evolution
        -- this is the only method available for exterior_scaling
        """
        self.setup_O()
        self.setup_T()
        # diagonalizing T
        ET, LT, RT = eig(self.T, b=self.O, left=True, right=True)
        LT = LT.transpose().conjugate()
        exp_T = np.exp(-1j*ET / self.hbar)
        # order according to absolute value:
        i_sort = np.argsort(-abs(exp_T))
        exp_T = exp_T[i_sort]
        RT = RT[:,i_sort]
        LT = LT[i_sort,:]
        # normalize RL to O and test the decomposition
        RT, LT = self.normalize_RL_to_O(RT, LT)
        # test the quality of the decomposition -------------------------
        # we exclude directions of evals below 10**(-15) by hand
        max_mode = len(np.where(abs(exp_T)>10**(-15))[0])
        ET_red = ET[:max_mode]
        RT_red = RT[:,:max_mode]
        LT_red = LT[:max_mode,:]
        # 1) test of orthogonality on the reduced space
        unity = np.dot(LT_red, np.dot(self.O, RT_red))
        ortho_error = abs(unity - np.diag(np.ones(max_mode))).max()
        print("Orthogonality errors", ortho_error)
        # 1) test difference between the full and the reduced te-operator
        UT_red = np.dot(RT_red, np.dot(np.diag(exp_T[:max_mode]),
                                               np.dot(LT_red, self.O)))
        UT = np.dot(RT, np.dot(np.diag(exp_T), np.dot(LT, self.O)))
        print("Propagator error", abs(UT_red - UT).max())
        self.UT = UT


    # FIXME: The part below is not very stable and just kept here for my
    # personal memory. The problem was that the computation of UV_half from
    # a diagonalization of V is not quite accurate. using the setup from
    # integration is recommended instead
    def setup_U_te(self, degree=None, reduced=False):
        """ sets up UT, UV_half and U via time-evolution method
        since the UV_half method is highly inaccurate it's use is not
        recommended.
        """
        self.setup_O()
        print("------------------------------------------------")
        print("set up of UT")
        self.setup_T()
        # diagonalizing T
        ET, LT, RT = eig(self.T, b=self.O, left=True, right=True)
        LT = LT.transpose().conjugate()
        exp_T = np.exp(-1j*ET / self.hbar)
        # order according to absolute value:
        i_sort = np.argsort(-abs(exp_T))
        exp_T = exp_T[i_sort]
        RT = RT[:,i_sort]
        LT = LT[i_sort,:]
        # normalize and test decomposition
        RT, LT = self.normalize_RL_to_O(RT, LT)
        self.test_normalization_RL_to_O(RT, LT)
        # put forth UT
        self.UT = np.dot(RT, np.dot(np.diag(exp_T), np.dot(LT, self.O)))

        # define the reduced eigenspace space of T --------------
        # here the key point is that T has many eigenvectors which decay
        # extremely fast. We want to rempve them by explicit projection
        # the label _red denotes an operator in the reduced eigenbasis of T
        print("maximal eigenvalue of UT:", abs(exp_T).max())
        print("minimal eigenvalue of UT:", abs(exp_T).min())
        if reduced:
            max_mode = len(np.where(abs(exp_T)>10**(-15))[0])
            print("necessary directions", max_mode)
        else:
            # include all states
            max_mode = len(ET)
        UT_red = np.diag(exp_T[:max_mode])

        # set up UV ---------------------------------------------
        print("------------------------------------------------")
        print("set up of UV")
        self.setup_V(degree=degree)
        # transform V to relevant space of T
        V_in_eigenspace_T = np.dot(LT, np.dot(self.V, RT))
        V_red = V_in_eigenspace_T[0:max_mode,0:max_mode]

        # diagonalization of V_red, note: the above trafo eliminates O
        EV_red, LV_red, RV_red = eig(V_red, left=True, right=True)
        LV_red = LV_red.transpose().conjugate()
        exp_V_half_red = np.exp(-1j*EV_red / (2.0 * self.hbar))
        # normalize and test decomposition
        RV_red, LV_red = self.normalize_RL(RV_red, LV_red)
        self.test_normalization_RL(RV_red, LV_red)

        # important note: the modes of V may grow, because the imaginary part
        # of the complex rotated potential is not necessarily negative
        print("maximal eigenvalue of UV_half:", abs(exp_V_half_red).max())
        print("minimal eigenvalue of UV_half:", abs(exp_V_half_red).min())

        # construct half kick evolution on reduced eigenspace of T
        UV_half_red = np.dot(RV_red, np.dot(np.diag(exp_V_half_red), LV_red))
        # construct half kick evolution on full coefficient space
        UV_half = np.zeros((len(exp_T), len(exp_T)), dtype=complex)
        UV_half[:max_mode, :max_mode] = UV_half_red
        # projecting from eigenspace of T to the coefficient space
        self.UV_half = np.dot(RT, np.dot(UV_half, np.dot(LT, self.O)))

        # set up U ---------------------------------------------
        print("------------------------------------------------")
        print("set up of U")
        # finally we set up the full U, first in the reduced eigenspace of T
        U_red = np.dot(UV_half_red, np.dot(UT_red, UV_half_red))
        # and lift it to the full coefficient space
        U = np.zeros_like(self.UT)
        U[:max_mode, :max_mode] = U_red.copy()
        # projecting from eigenspace of T to the coefficient space
        self.U = np.dot(RT, np.dot (U, np.dot(LT, self.O)))

        # we further provide its eigenvectors and eigenvalues from the reduced
        # space
        evals_red, L_red, R_red = eig(U_red, left=True, right=True)
        L_red = L_red.transpose().conjugate()
        # normalize and test decomposition
        R_red, L_red = self.normalize_RL(R_red, L_red)
        self.test_normalization_RL(R_red, L_red)

        # check the eigenvalues
        print("maximal eigenvalue of U:", abs(evals_red).max())
        print("minimal eigenvalue of U:", abs(evals_red).min())

        i_sort = np.argsort(-abs(evals_red))
        evals_red = evals_red[i_sort]
        R_red = R_red[:,i_sort]
        L_red = L_red[i_sort,:]

        # construct solution on full coefficient space
        self.evals = np.zeros(len(exp_T), dtype=complex)
        self.evals[:max_mode] = evals_red
        # right eigenvectors
        R = np.diag(np.ones(len(exp_T), dtype=complex))
        R[:max_mode, :max_mode] = R_red
        self.R = np.dot(RT, R)
        # left eigenvectors
        L = np.diag(np.ones(len(exp_T), dtype=complex))
        L[:max_mode, :max_mode] = L_red
        self.L = np.dot(L, np.dot(LT, self.O))
        self.test_normalization_RL(self.R, self.L)
