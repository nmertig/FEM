#pylint: disable=C0103
"""
This class provides several tests for integration via Quadratures of
legendre Polynomials
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt
from scipy.special import erf
from FEM.core.legendre_quad_2d import LegendreQuad2d

# --------------------------------------------------------------------------
# Tests

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

    def compute_integrals(self, f, x, y, F=None):
        """
        """
        # set up data array
        self.F_num_array = np.zeros(len(self.degree_array), dtype=complex)
        LQ = LegendreQuad2d()
        for i in range(len(self.degree_array)):
            self.F_num_array[i] = LQ.Integrate(f, x, y,
                                               self.degree_array[i],
                                               self.degree_array[i])

        if F is not None:
            self.F_exact = F(x[0], x[-1], y[0], y[-1])
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


def test_on_Polynomial(m=17, n=34):
    """ This test integrates a Polynomial. The result should be exact as soon
    as: 2*degree > order
    """
    def f(x,y, m=m, n=n):
        return (x**m) * (y**n)

    def F(x0, x1, y0, y1, m=m, n=n):
        Fx = (x1**(m+1) - x0**(m+1)) / (m+1.0)
        Fy = (y1**(n+1) - y0**(n+1)) / (n+1.0)
        return Fx * Fy

    x = np.linspace(0.0, 1.0, 2)
    y = np.linspace(0.0, 1.0, 2)

    T = HelpTesting(1, 30)
    T.compute_integrals(f, x, y, F=F)
    T.visualize()

def test_complex_exp(m=4, n=5):
    """ This is both a test for oscillatory as well as complex integration
    """
    def f(x, y, m=m, n=n):
        return np.exp(1j * (x*m + y*n))

    def F(x0, x1, y0, y1, m=m, n=n):
        Fx = (np.exp(1j*m*x1) - np.exp(1j*m*x0)) / (1.0 * m)
        Fy = (np.exp(1j*n*y1) - np.exp(1j*n*y0)) / (1.0 * n)
        return - Fx * Fy

    x = np.linspace(0.0, 2.0*np.pi*m, (2*m+1))
    y = np.linspace(0.0, 2.0*np.pi*n, (2*n+1))

    x = np.linspace(0.01, 0.5*np.pi, 2)
    y = np.linspace(-0.02, 0.5*np.pi, 2)

    T = HelpTesting(1, 30)
    T.compute_integrals(f, x, y, F=F)
    T.visualize()

def test_Complex_Scaled_Propagator(N=100, theta=np.pi/4.0):
    """ double integral arrising from the complex scaled propagator
    """
    heff = 1.0 / (1.0 * N) / (2.0 * np.pi)
    alpha = np.exp(1j*(theta - np.pi/4.0)) / np.sqrt(2.0 * heff)

    def f(x, y, alpha=alpha):
        return alpha * np.exp(- alpha**2 * (x-y)**2) / np.sqrt(np.pi)

    def F(x0, x1, y0, y1, alpha=alpha):
        z1 = alpha * (y1 - x0)
        z2 = alpha * (y0 - x1)
        z3 = alpha * (y1 - x1)
        z4 = alpha * (y0 - x0)
        F1 = lambda z: z * erf(z) / (2.0 * alpha)
        F2 = lambda z: np.exp(-z**2) / np.sqrt(np.pi) / (2.0 * alpha)
        return F1(z1) + F1(z2) - (F1(z3) + F1(z4)) +\
                    F2(z1) + F2(z2) - (F2(z3) + F2(z4))

    L = 0.5
    fact=1.0
    x_=0.0
    y_=0.0
    x = np.linspace(x_, x_ + L, int(L**2*N*fact) + 2)
    y = np.linspace(y_, y_ + L, int(L**2*N*fact) + 2)

    T = HelpTesting(1, 30)
    T.compute_integrals(f, x, y, F=F)
    T.visualize()


if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    test_on_Polynomial()
    test_complex_exp()
    test_Complex_Scaled_Propagator()
