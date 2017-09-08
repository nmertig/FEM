#pylint: disable=C0103
"""
The pupose of this program is to get some intuition about integration based on
quadratures. Here, we specifically deal with
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt
from scipy.special import erf
from numpy.polynomial.legendre import leggauss
from numpy.random import rand

class LegendreQuad2d(object):
    """ This is a numerical integrator based on Hermite Polynomials for
    functions of two coordinates.
    """
    def __init__(self):
        """ Nande mo nai
        """

    def Integrate(self, f, x, y, degree_x, degree_y):
        """
        """
        Nx = len(x)
        Ny = len(y)
        x_nodes, x_weights = leggauss(degree_x)
        y_nodes, y_weights = leggauss(degree_y)

        F = 0.0
        na = np.newaxis
        for i in range(Nx-1):
            x0 = x[i]
            x1 = x[i+1]
            scaled_x_nodes = x0 + (x1 - x0)/2.0 * (x_nodes + 1)
            x_grid = scaled_x_nodes[:,na] * np.ones(degree_y)[na,:]
            # now iterate through the y simplixes
            for j in range(Ny-1):
                y0 = y[j]
                y1 = y[j+1]
                scaled_y_nodes = y0 + (y1 - y0)/2.0 * (y_nodes + 1)
                # set up the integration grid
                y_grid = scaled_y_nodes[na,:] * np.ones(degree_x)[:,na]
                weights_2d = x_weights[:,na] * y_weights[na,:]

                # add to value
                F += (x1-x0)*(y1-y0)*np.sum(weights_2d*f(x_grid,y_grid))/4.0
        return F
