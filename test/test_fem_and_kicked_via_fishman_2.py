#pylint: disable=C0103
"""
This Program tests the FEM Cell code Provided by

FEM_Cell_Legendre for setting up and solving the box potential

"""

from __future__ import (division, print_function)
import numpy as np
from numpy.polynomial import Hermite
from scipy.linalg import eig
from scipy.special import gamma
from matplotlib import pylab as plt

from FEM.systems.fishman import Fishman
from FEM.core.convenient_function import recommended_N
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
    def test_conversion(self, with_plot=True):
        """ test conversion of wave functions into coefficients and back
        """
        error_matrix = np.zeros((self.Nq, self.Np), dtype=float)
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)

        for iq in range(self.Nq):
            print(iq)
            for ip in range(self.Np):
                self.GWP.q0 = self.q_grid[iq]
                self.GWP.p0 = self.p_grid[ip]
                psi = lambda x: self.GWP.psi(x,t=0.0)
                coeff = self.FM.coeff_from_psi(psi, degree=self.degree)
                # numerical function
                psi_x = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff)
                # exact function
                psi_x_exact = psi(x_plot)
                error_matrix[iq, ip] = abs(psi_x_exact - psi_x).max()

        print('Error of conversion:', error_matrix.max())
        if with_plot:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(111)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.imshow(error_matrix[:,::-1].transpose(), interpolation='none')
            plt.show()

    def test_half_kick(self, with_plot=True):
        """ test converted half-kick evolved wave packet vs analytic result
        """
        error_matrix = np.zeros((self.Nq, self.Np), dtype=float)
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)

        self.FM.setup_O()
        self.FM.setup_UV_half_int(degree=self.degree)

        for iq in range(self.Nq):
            print(iq)
            for ip in range(self.Np):
                self.GWP.q0 = self.q_grid[iq]
                self.GWP.p0 = self.p_grid[ip]
                # get coefficient
                psi = lambda x: self.GWP.psi(x,t=0.0)
                coeff = self.FM.coeff_from_psi(psi, degree=self.degree)
                # propagate coefficients
                coeff =\
                    np.linalg.solve(self.FM.O, np.dot(self.FM.UV_half, coeff))
                # numerical function
                psi_x = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff)
                # exact function
                psi_x_exact = psi(x_plot) * self.FM.exp_V_half(x_plot)
                error_matrix[iq, ip] = abs(psi_x_exact - psi_x).max()

        print('Error of conversion:', error_matrix.max())
        if with_plot:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(111)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.imshow(error_matrix[:,::-1].transpose(), interpolation='none')
            plt.show()

    def test_free_evolution(self, t=1, with_plot=True):
        """ test converted freely evolved wave packet vs analytic result
        """
        error_matrix = np.zeros((self.Nq, self.Np), dtype=float)
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)

        self.FM.setup_O()
        self.FM.setup_UT_int_fast(degree=self.degree2d, eps=10**(-15))

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
                    coeff =\
                        np.linalg.solve(self.FM.O, np.dot(self.FM.UT, coeff))
                # numerical function
                psi_x = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff)
                # exact function
                psi_x_exact = self.GWP.psi(x_plot, t=t)
                error_matrix[iq, ip] = abs(psi_x_exact - psi_x).max()

        print('Error of conversion:', error_matrix.max())
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
        error_matrix = np.zeros((self.Nq, self.Np), dtype=float)
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)

        print("setting up O")
        self.FM.setup_O()
        print("setting up UT")
        self.FM.setup_UT_int_fast(degree=self.degree2d, eps=10**(-15))
        print("setting up UV_half")
        self.FM.setup_UV_half_int(degree=self.degree)
        print("setting up U")
        self.FM.setup_U_int(degree=self.degree2d, eps=10**(-15))

        for iq in range(self.Nq):
            print(iq)
            for ip in range(self.Np):
                self.GWP.q0 = self.q_grid[iq]
                self.GWP.p0 = self.p_grid[ip]
                # get coefficient
                psi = lambda x: self.GWP.psi(x,t=0.0)
                coeff = self.FM.coeff_from_psi(psi, degree=self.degree)
                # propagate coefficients by pieces
                coeff1 = coeff.copy()
                for i in range(t):
                    coeff1 =\
                    np.linalg.solve(self.FM.O,np.dot(self.FM.UV_half, coeff1))
                    coeff1 =\
                    np.linalg.solve(self.FM.O,np.dot(self.FM.UT, coeff1))
                    coeff1 =\
                    np.linalg.solve(self.FM.O,np.dot(self.FM.UV_half, coeff1))

                # propagate coefficients by U
                coeff2 = coeff.copy()
                for i in range(t):
                    coeff2=np.linalg.solve(self.FM.O,np.dot(self.FM.U,coeff2))

                # numerical function
                psi_x1 = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff1)
                # exact function
                psi_x2 = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff2)
                error_matrix[iq, ip] = abs(psi_x1 - psi_x2).max()

        print('Error of conversion:', error_matrix.max())
        if with_plot:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(111)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.imshow(error_matrix[:,::-1].transpose(), interpolation='none')
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

    Test.test_conversion(with_plot=True)     # worked well
    Test.test_half_kick(with_plot=True)      # worked well
    Test.test_free_evolution(with_plot=True) # worked well
    Test.test_consistency_U_versus_UVTV()    # worked well

