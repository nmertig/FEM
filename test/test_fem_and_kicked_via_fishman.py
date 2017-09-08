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
    def __init__(self, FM, GWP, degree=50, dergee2d=50):
        """
        """
        self.degree = degree
        self.degree2d = degree2d
        self.FM = FM
        self.GWP = GWP

    # ####################################################################

    def test_consistency_UT_versus_UT_fast(self):
        """ consistency check between setup methods
        """
        ## checking setup_UT versus setup_UT_fast
        print('get UT via fast')
        self.FM.setup_UT_int_fast(degree=self.degree2d, eps=10**(-15))
        UT = self.FM.UT.copy()
        print('get UT via slow')
        self.FM.setup_UT_int(degree=self.degree2d, eps=10**(-15))
        er = abs(UT - self.FM.UT).max()
        print("Error between UT and UT_fast:", er)

    def test_conversion(self, with_plot=True):
        """ test conversion of wave functions into coefficients and back
        """
        psi = lambda x: self.GWP.psi(x,t=0.0)
        coeff = self.FM.coeff_from_psi(psi, degree=self.degree)

        # prepare results on a grid
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)
        psi_x = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff)
        psi_x_exact = psi(x_plot)

        print('Error of conversion:', abs(psi_x_exact - psi_x).max())
        if with_plot:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.plot(x_plot, psi_x, 'ko', label=None)
            ax1.plot(x_plot, psi_x_exact, 'r-', label=None)

            ax2 = fig.add_subplot(122)
            ax2.set_xscale('linear')
            ax2.set_yscale('linear')
            ax2.plot(x_plot, abs(psi_x - psi_x_exact), 'ko', label=None)
            plt.show()

    def test_half_kick(self, with_plot=True):
        """ test converted half-kick evolved wave packet vs analytic result
        """
        self.FM.setup_O()
        self.FM.setup_UV_half_int(degree=self.degree)

        # get coefficients
        psi = lambda x: self.GWP.psi(x,t=0.0)
        coeff = self.FM.coeff_from_psi(psi, degree=self.degree)

        # propagate coefficients
        coeff_Vhalf = np.linalg.solve(self.FM.O, np.dot(self.FM.UV_half, coeff))

        # prepare results on a grid
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)
        psi_x = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff_Vhalf)
        psi_x_exact = psi(x_plot) * self.FM.exp_V_half(x_plot)

        print('Error of conversion:', abs(psi_x_exact - psi_x).max())
        if with_plot:
            # plotting
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.plot(x_plot, psi_x, 'ko', label=None)
            ax1.plot(x_plot, psi_x_exact, 'r-', label=None)

            ax2 = fig.add_subplot(122)
            ax2.set_xscale('linear')
            ax2.set_yscale('linear')
            ax2.plot(x_plot, abs(psi_x - psi_x_exact), 'ko', label=None)
            plt.show()

    def test_free_evolution(self, t=1, with_plot=True):
        """ test converted freely evolved wave packet vs analytic result
        """
        self.FM.setup_O()
        self.FM.setup_UT_int_fast(degree=self.degree2d, eps=10**(-15))

        # get coefficients
        psi = lambda x: self.GWP.psi(x,t=0.0)
        coeff = self.FM.coeff_from_psi(psi, degree=degree)

        # propagate coefficients
        for i in range(t):
            # propagates coefficients t steps
            coeff = np.linalg.solve(self.FM.O, np.dot(self.FM.UT, coeff))

        # prepare results on a grid
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)
        psi_x = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff)
        psi_x_exact = self.GWP.psi(x_plot, t=t)

        print('Error of conversion:', abs(psi_x_exact - psi_x).max())
        if with_plot:
            # plotting
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.plot(x_plot, abs(psi_x), 'ko', label=None)
            ax1.plot(x_plot, abs(psi_x_exact), 'r-', label=None)

            ax2 = fig.add_subplot(122)
            ax2.set_xscale('linear')
            ax2.set_yscale('linear')
            ax2.plot(x_plot, abs(psi_x - psi_x_exact), 'ko', label=None)
            plt.show()

    def test_consistency_U_versus_UVTV(self, t=1, with_plot=True):
        """ test converted freely evolved wave packet vs analytic result
        """
        self.FM.setup_O()
        print("setting up UT")
        self.FM.setup_UT_int_fast(degree=self.degree2d, eps=10**(-15))
        print("setting up UV_half")
        self.FM.setup_UV_half_int(degree=self.degree)
        print("setting up U")
        self.FM.setup_U_int(degree=self.degree2d, eps=10**(-15))

        # get coefficients
        psi = lambda x: self.GWP.psi(x,t=0.0)
        coeff_0 = self.FM.coeff_from_psi(psi, degree=degree)

        # propagate coefficients by pieces
        coeff1 = coeff_0.copy()
        for i in range(t):
            coeff1 = np.linalg.solve(self.FM.O,np.dot(self.FM.UV_half, coeff1))
            coeff1 = np.linalg.solve(self.FM.O,np.dot(self.FM.UT, coeff1))
            coeff1 = np.linalg.solve(self.FM.O,np.dot(self.FM.UV_half, coeff1))

        # propagate coefficients by U
        coeff2 = coeff_0.copy()
        for i in range(t):
            coeff2 = np.linalg.solve(self.FM.O, np.dot(self.FM.U, coeff2))

        # prepare results on a grid
        x_plot = np.linspace(self.FM.x_grid[0], self.FM.x_grid[-1], 4001)
        psi_x1 = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff1)
        psi_x2 = self.FM.psi_from_coeff(x_plot, m=0, coeff=coeff2)

        print('Error between methods:', abs(psi_x1 - psi_x2).max())
        if with_plot:
            # plotting
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121)
            ax1.set_xscale('linear')
            ax1.set_yscale('linear')
            ax1.plot(x_plot, abs(psi_x1), 'ko', label=None)
            ax1.plot(x_plot, abs(psi_x2), 'r-', label=None)

            ax2 = fig.add_subplot(122)
            ax2.set_xscale('linear')
            ax2.set_yscale('linear')
            ax2.plot(x_plot, abs(psi_x1 - psi_x2), 'ko', label=None)
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

    Test.test_consistency_UT_versus_UT_fast() # test was passed (took ages)
    Test.test_conversion(with_plot=False)
    Test.test_half_kick(with_plot=False)
    Test.test_free_evolution(with_plot=False)
    #Test.test_consistency_U_versus_UVTV()

