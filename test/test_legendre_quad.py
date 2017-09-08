#pylint: disable=C0103
"""
This class provides several tests for integration via Quadratures of
legendre Polynomials
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt
from scipy.special import erf
from FEM.core.legendre_quad import LegendreQuad

# --------------------------------------------------------------------------
#Tests

class HelpTesting(object):
    """ Visualizes the results of numerical integration.
    """
    def __init__(self, degree_min, degree_max):
        """
        """
        self.degree_min = degree_min
        self.degree_max = degree_max
        self.degree_array = degree_min + np.arange(degree_max - degree_min)
        self.F_num_array = None
        self.F_exact = None

    def compute_integrals(self, f, x_, F=None):
        """
        """
        # set up data array
        self.F_num_array = np.zeros(len(self.degree_array), dtype=complex)
        LQ = LegendreQuad()
        for i in range(len(self.degree_array)):
            self.F_num_array[i] = LQ.Integrate(f, x_, self.degree_array[i])

        if F is not None:
            self.F_exact = (F(x_[-1])-F(x_[0]))
            print('analytical result:', self.F_exact)

    def visualize(self):
        """
        """
        fig = plt.figure(1)

        # showing the result of the integration on a linear scale -------
        ax1 = fig.add_subplot(121)
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax1.set_xlim(self.degree_min-1, self.degree_max+1)

        ax1.plot(self.degree_array, self.F_num_array.real,
                 'k-', label='num real')
        ax1.plot(self.degree_array, self.F_num_array.imag,
                 'k.', lw=1, label='num imag')
        if self.F_exact is not None:
            ax1.plot(self.degree_array,
                     self.F_exact.real + 0.0*self.degree_array,
                     'r-', label='analytic real')
            ax1.plot(self.degree_array,
                     self.F_exact.imag + 0.0*self.degree_array,
                     'r.', label='analytic imag')

        # showing the relative and absolute error of the result on a log scale
        ax2 = fig.add_subplot(122)
        ax2.set_xscale('linear')
        ax2.set_yscale('log')
        ax2.set_xlim(self.degree_min-1, self.degree_max+1)
        ax2.plot(self.degree_array[:-1],
                abs(self.F_num_array[1:] - self.F_num_array[:-1]),
                'ko', label='abs error est')
        if self.F_exact is not None:
            ax2.plot(self.degree_array,
                    abs(self.F_num_array - self.F_exact),
                    color='k', lw=1, label='abs error')
            ax2.plot(self.degree_array,
                     abs(self.F_num_array - self.F_exact) / abs(self.F_exact),
                     color='r', lw=1, label='rel error')
        plt.show()


def test_on_Polynomial(order=30):
    """ This test integrates a Polynomial. The result should be exact as soon
    as: 2*degree > order
    """
    f = lambda x: x**order
    F = lambda x: x**(order + 1) / (1.0 * (order + 1))

    x_ = np.linspace(-1, 2, 2)
    T = HelpTesting(1, 20)
    T.compute_integrals(f, x_, F=F)
    T.visualize()

def test_Fishman_Potential(kappa=3.4):
    """ This shows that integration can be very exact, if the function is
    integrated by quadratures in the relevant region.
    """
    f = lambda x: -kappa * np.exp(- 8.0 * x*x) / 16.0
    F = lambda x: -kappa * erf(x * np.sqrt(8.0)) * np.sqrt(2.0 * np.pi) / 128.0

    x_ = np.linspace(-4, 4, 20)
    T = HelpTesting(1, 20)
    T.compute_integrals(f, x_, F=F)
    T.visualize()

def test_complex_exp():
    """ This is both a test for oscillatory as well as complex integration
    """
    f = lambda x: np.exp(1j*x)
    F = lambda x: -1j*np.exp(1j*x)

    x_ = np.linspace(-0.03, np.pi/2.0 - 0.01, 2)
    T = HelpTesting(1, 20)
    T.compute_integrals(f, x_, F=F)
    T.visualize()

def test_complex_Gauss():
    """ This test integrates
    """
    f = lambda x: np.exp(1j * x*x)
    sqrt_i = np.exp(1j*np.pi/4.0)
    F = lambda x: erf(x/sqrt_i) * np.sqrt(np.pi) * sqrt_i / 2.0

    n = 1000
    m = 10
    x_ = np.linspace(np.sqrt(m * np.pi), np.sqrt(n * np.pi), (n - m + 1))
    #x_ = np.sqrt(np.pi * np.linspace(m, n, (n - m + 1)))
    #x_ = np.linspace(np.sqrt(m * np.pi), np.sqrt(n * np.pi), int(n))
    T = HelpTesting(1, 20)
    T.compute_integrals(f, x_, F=F)
    T.visualize()

def test_complex_Gauss():
    """ This test integrates
    """
    f = lambda x: np.exp(1j * x*x)
    sqrt_i = np.exp(1j*np.pi/4.0)
    F = lambda x: erf(x/sqrt_i) * np.sqrt(np.pi) * sqrt_i / 2.0

    n = 1000
    m = 10
    x_ = np.linspace(np.sqrt(m * np.pi), np.sqrt(n * np.pi), (n - m + 1))
    #x_ = np.sqrt(np.pi * np.linspace(m, n, (n - m + 1)))
    #x_ = np.linspace(np.sqrt(m * np.pi), np.sqrt(n * np.pi), int(n))
    T = HelpTesting(1, 20)
    T.compute_integrals(f, x_, F=F)
    T.visualize()

def test_Fishman_Kick(kappa=3.4, N=10):
    """ This test integrates the Fishman propagator
    """
    heff = 1.0 / (1.0 * N) / (2.0 * np.pi)
    V = lambda x: -kappa * np.exp(- 8.0 * x*x) / 16.0
    f = lambda x: np.exp(-1j*V(x)/heff)

    x_min = -4.0
    x_max = 4.0
    N_int = int(N * 4.0 * np.exp(0.5) * (x_max - x_min) / kappa)

    x_ = np.linspace(x_min, x_max, N_int)

    T = HelpTesting(1, 20)
    T.compute_integrals(f, x_)
    T.visualize()

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    test_on_Polynomial(order=30)
    test_Fishman_Potential(kappa=3.4)
    test_complex_exp()
    test_complex_Gauss()
    test_Fishman_Kick(kappa=3.4, N=10)

