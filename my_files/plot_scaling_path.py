#pylint: disable=C0103
"""
This program calculates the resonances of the fishman map
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt

from FEM.core.fem import FEM

# testing stuff ---------------------------------------------------------

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    hbar = 1.0
    max_order = 10
    x_grid = np.linspace(-5, 5, 301)

    fem = FEM(hbar, max_order, x_grid, theta=np.pi/4.0,
              dirichlet_bound_conds=True,
              exterior_scaling=True, exterior_scaling_width=0.6,
              exterior_x0=2.0)

    scaling_path = fem.scaling_path(fem.x_grid)

    fig = plt.figure(1)

    ax1 = fig.add_subplot(311)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.plot(scaling_path.real, scaling_path.imag, 'k-', label=None)

    ax2 = fig.add_subplot(312)
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.plot(fem.x_grid, scaling_path.real, 'k-', label=None)

    ax3 = fig.add_subplot(313)
    ax3.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.plot(fem.x_grid, scaling_path.imag, 'k-', label=None)

    plt.show()