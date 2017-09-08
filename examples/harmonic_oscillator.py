#pylint: disable=C0103
"""
This program runs the fem class on the harmonic oscillator
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt

from FEM.systems.harmonic_oscillator import HarmonicOscillator

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    hbar = 1.0
    N_cell = 30
    max_order = 10
    L = 10.0
    HO = HarmonicOscillator(hbar, N_cell, max_order, L=L,
                            dirichlet_bound_conds=False)

    # set up
    HO.setup_O()
    HO.setup_T()
    HO.setup_V()
    HO.setup_H()

    # diagonalize
    HO.compute_evecs()
    HO.order_evecs()
    HO.normalize_R()

    # show the results -----------------------------------------------
    E_num = HO.evals
    E_an = HO.E(np.arange(HO.dim))
    x_plot = np.linspace(HO.x_grid[0], HO.x_grid[-1], 1000)

    fig = plt.figure(1)

    # showing the result of the integration on a linear scale -------
    ax1 = fig.add_subplot(121)
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_xlim(-1, int(N_cell/2.0))
    ax1.plot(np.arange(HO.dim), abs(E_an-E_num), 'ko', label=None)

    i_state=0
    ax2 = fig.add_subplot(122)
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.plot(x_plot, HO.psi(x_plot, n=i_state), 'k-')
    ax2.plot(x_plot, HO.psi_from_coeff(x_plot, m=i_state), 'r+')

    def onclick(event):
        i_state = int(event.xdata)
        print(i_state)
        ax2.lines = []
        ax2.plot(x_plot, HO.psi(x_plot, n=i_state), 'k-')
        ax2.plot(x_plot, HO.psi_from_coeff(x_plot, m=i_state), 'r+')
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
