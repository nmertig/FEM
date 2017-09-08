#pylint: disable=C0103
"""
This Program tests the FEM Cell code Provided by

FEM_Cell_Legendre for setting up and solving the box potential

"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt
from time import time

from FEM.core.convenient_function import recommended_N
from FEM.systems.fishman import Fishman
from FEM.extras.gaussian_wave_packet import GWP as gwp

# testing stuff ---------------------------------------------------------

class test_fem_via_fishman(object):
    """
    """
    def __init__(self, FM, GWP, qmin=-2.0, qmax=2.0, pmin=-1.0, pmax=1.0,
                 Nq=10, Np=7, degree=50, dergee2d=50):
        """
        """
        self.degree = degree
        self.degree2d = degree2d
        self.FM = FM
        self.GWP = GWP
        # setting up the phase space grid
        self.Nq=Nq
        self.Np=Np
        self.qmin=qmin
        self.qmax=qmax
        self.pmin=pmin
        self.pmax=pmax
        self.q_grid = np.linspace(qmin, qmax, Nq)
        self.p_grid = np.linspace(pmin, pmax, Np)

    # ####################################################################
    def test_operators_from_time_evolution(self, t=1, with_plot=True):
        """ test converted half-kick evolved wave packet vs analytic result
        """
        # first setting up all operators
        self.FM.setup_U_te(degree=self.degree, reduced=False)

        # then checking the free-time evolution matrix ----------------------
        error_matrix = np.zeros((self.Nq, self.Np), dtype=float)
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)

        for iq in range(self.Nq):
            print(iq)
            for ip in range(self.Np):
                self.GWP.q0 = self.q_grid[iq]
                self.GWP.p0 = self.p_grid[ip]
                # get coefficient
                psi = lambda x: self.GWP.psi(x,t=0.0)
                coeff = self.FM.coeff_from_psi(psi, degree=self.degree)
                # propagate coefficients
                for i in range(t):
                    coeff = np.dot(self.FM.UT, coeff)
                # numerical function
                psi_x = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff)
                # exact function
                psi_x_exact = self.GWP.psi(x_plot, t=t)
                error_matrix[iq, ip] = abs(psi_x_exact - psi_x).max()

        print('Error of free evolution:', error_matrix.max())
        if with_plot:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(111)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.imshow(error_matrix[:,::-1].transpose(), interpolation='none')
            plt.show()

        # now checking the half kick behavior ------------------------------
        error_matrix = np.zeros((self.Nq, self.Np), dtype=float)
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)

        for iq in range(self.Nq):
            print(iq)
            for ip in range(self.Np):
                self.GWP.q0 = self.q_grid[iq]
                self.GWP.p0 = self.p_grid[ip]
                # get coefficient
                psi = lambda x: self.GWP.psi(x,t=0.0)
                coeff = self.FM.coeff_from_psi(psi, degree=self.degree)
                # propagate coefficients
                coeff = np.dot(self.FM.UV_half, coeff)
                # numerical function
                psi_x = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff)
                # exact function
                psi_x_exact = psi(x_plot) * self.FM.exp_V_half(x_plot)
                error_matrix[iq, ip] = abs(psi_x_exact - psi_x).max()
                #if with_plot:
                    ## plotting
                    #fig = plt.figure(1)
                    #ax1 = fig.add_subplot(121)
                    #ax1.set_xscale('linear')
                    #ax1.set_yscale('linear')
                    #ax1.plot(x_plot, psi_x, 'ko', label=None)
                    #ax1.plot(x_plot, psi_x_exact, 'r-', label=None)

                    #ax2 = fig.add_subplot(122)
                    #ax2.set_xscale('linear')
                    #ax2.set_yscale('linear')
                    #ax2.plot(x_plot, abs(psi_x - psi_x_exact), 'ko', label=None)
                    #plt.show()

        print('Error of half kick:', error_matrix.max())
        if with_plot:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(111)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.imshow(error_matrix[:,::-1].transpose(), interpolation='none')
            plt.show()

    def test_consistency_U_versus_UVTV(self, t=1, with_plot=True):
        """ test converted freely evolved wave packet vs analytic result
        """
        # first setting up all operators via time-evolution method
        print("setup via time evolution")
        t_ = time()
        self.FM.setup_U_te(degree=self.degree, reduced=True)
        U1 = self.FM.U
        print("Time evolution setup took sec:", time()-t_)

        print("setup via integration of the propagator")
        t_ = time()
        # next setup all operators via integration of the propagator
        print("setting up O")
        self.FM.setup_O()
        O = self.FM.O
        print("setting up UT")
        self.FM.setup_UT_int_fast(degree=self.degree2d, eps=10**(-15))
        UT = self.FM.UT
        print("setting up UV_half")
        self.FM.setup_UV_half_int(degree=self.degree)
        UV_half = self.FM.UV_half
        print("Setup by integration took sec:", time()-t_)
        print("setting up U")
        t_ = time()
        self.FM.setup_U_int(degree=self.degree2d, eps=10**(-15))
        U3 = self.FM.U
        print("Setup by full integration took sec:", time()-t_)

        error_matrix12 = np.zeros((self.Nq, self.Np), dtype=float)
        error_matrix13 = np.zeros((self.Nq, self.Np), dtype=float)
        error_matrix23 = np.zeros((self.Nq, self.Np), dtype=float)
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)

        # in the following we compare three types of time evulotion
        for iq in range(self.Nq):
            print(iq)
            for ip in range(self.Np):
                self.GWP.q0 = self.q_grid[iq]
                self.GWP.p0 = self.p_grid[ip]
                # get coefficient
                psi = lambda x: self.GWP.psi(x,t=0.0)
                coeff = self.FM.coeff_from_psi(psi, degree=self.degree)

                # propagate coefficients by pieces result from time-evolution
                coeff1 = coeff.copy()
                for i in range(t):
                    coeff1 = np.dot(U1, coeff1)
                # propagate coefficients by pieces
                coeff2 = coeff.copy()
                for i in range(t):
                    coeff2 =\
                    np.linalg.solve(self.FM.O,np.dot(UV_half, coeff2))
                    coeff2 =\
                    np.linalg.solve(self.FM.O,np.dot(self.FM.UT, coeff2))
                    coeff2 =\
                    np.linalg.solve(self.FM.O,np.dot(self.FM.UV_half, coeff2))
                # propagate coefficients by full U
                coeff3 = coeff.copy()
                for i in range(t):
                    coeff3=np.linalg.solve(self.FM.O,np.dot(self.FM.U, coeff3))
                # result by time-evolution setup
                psi_x1 = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff1)
                # result using integration of propagator by parts
                psi_x2 = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff2)
                # result using integration of the full propagator
                psi_x3 = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff3)

                error_matrix12[iq, ip] = abs(psi_x1 - psi_x2).max()
                error_matrix13[iq, ip] = abs(psi_x1 - psi_x3).max()
                error_matrix23[iq, ip] = abs(psi_x2 - psi_x3).max()

        print('Error time-evol setup versus parts:', error_matrix12.max())
        print('Error time-evol setup versus full:', error_matrix13.max())
        print('Error parts versus full:', error_matrix23.max())
        if with_plot:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(131)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.imshow(error_matrix12[:,::-1].transpose(),interpolation='none')

            ax2 = fig.add_subplot(132)
            ax2.set_xscale('linear')
            ax2.set_yscale('linear')
            ax2.imshow(error_matrix13[:,::-1].transpose(),interpolation='none')

            ax3 = fig.add_subplot(133)
            ax3.set_xscale('linear')
            ax3.set_yscale('linear')
            ax3.imshow(error_matrix23[:,::-1].transpose(),interpolation='none')
            plt.show()

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    # system parameters ---------------------------------------
    kappa=1.0
    states_per_cell = 5
    hbar = 1.0 / (2.0 * np.pi) / states_per_cell

    # computational parameters --------------------------------
    degree = 50
    degree2D = 50
    max_order = 10
    theta = 0.125 * np.pi
    x_max = 4.0
    N = recommended_N(hbar, theta, 2.0*x_max)

    degree = 30
    degree2d = 30

    # initialize the fishmen map ----------------------------
    FM = Fishman(kappa, hbar, max_order, N, x_max, theta=theta)

    # initialize the Gaussian wave packet dynamics --------------
    q0 = 0.3
    p0 = 0.1
    sigma = np.sqrt(hbar/2.0)
    GWP = gwp(q0, p0, sigma, hbar, theta)

    Test = test_fem_via_fishman(FM, GWP, degree=degree, dergee2d=degree2d)
    Test.test_operators_from_time_evolution(with_plot=True)
    # the above test worked ok. the half kick is not so well represented,
    # which is probably not a problem, since the errors will be removed
    # by projection
    Test.test_consistency_U_versus_UVTV()
    # did not work so great. we suspect this is due to time-evolution set up
    # not representing the half kick part well
    # it remains to investigate wether a true time-evolution method performs
    # better