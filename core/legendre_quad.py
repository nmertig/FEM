"""
This class integrates a function via the use of Quadratures of the Legendre
Polynomials
"""

from __future__ import (division, print_function)
import numpy as np
from numpy.polynomial.legendre import leggauss

class LegendreQuad(object):
    """ This is a numerical integrator based on Hermite Polynomials
    """
    def __init__(self):
        """ Nande mo nai
        """

    def Integrate(self, f, x_, degree):
        """
        """
        N = len(x_)
        nodes, weights = leggauss(degree)

        F_ab = 0.0
        for i in range(N-1):
            a = x_[i]
            b = x_[i+1]
            scaled_nodes = a + (b-a)/2.0 * (nodes + 1)
            F_ab += np.sum(((b-a)/2.0) * weights * f(scaled_nodes))
        return F_ab
