#pylint: disable=C0103
"""
This program plots the absorbing potential of the fishman map
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt

from FEM.core.convenient_function import recommended_N
from FEM.systems.fishman_absorbing_potential import Fishman

# testing stuff ---------------------------------------------------------

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    # system parameters ---------------------------------------
    kappa=1.0
    states_per_cell = 5
    hbar = 1.0 / (2.0 * np.pi) /states_per_cell

    qabs = 0.0
    width = 0.001
    amp = 0.01
    poly_order = 2

    # computational parameters --------------------------------
    max_order = 4
    x_max = 5.0
    p_max = 5.0
    N = int(4 * p_max * x_max * states_per_cell)

    # initialize the fishmen map ----------------------------
    FM = Fishman(kappa, hbar, max_order, N, x_max,
                 qabs, amp, width, poly_order)

    # plotting the absorbing potentials --------------------------------
    fig = plt.figure()

    x_plot = np.linspace(-5.0, 5.0, 1001)

    ax1 = fig.add_subplot(111)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.plot(x_plot, FM.F_(x_plot), '-', color=(0.7,0.7,0.7), label=None)
    ax1.plot(x_plot, amp * x_plot**poly_order, 'k-', label=None)

    plt.show()