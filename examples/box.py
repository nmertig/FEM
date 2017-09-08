#pylint: disable=C0103
"""
This program solves the box potential using a finite element method
"""

from __future__ import (division, print_function)
import numpy as np
from scipy.linalg import eig
from matplotlib import pylab as plt

from FEM.systems.box import Box_Potential

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    hbar = 1.0 / (np.pi)
    N_cell = 40
    max_order = 10
    BP = Box_Potential(hbar, N_cell, max_order,
                       dirichlet_bound_conds=False)

    # set up
    BP.setup_O()
    BP.setup_T()
    BP.setup_V(nonzero=False)
    BP.setup_H()

    # diagonalize
    BP.compute_evecs()
    BP.order_evecs()
    BP.normalize_R()

    # show the results -----------------------------------------------
    E_num = BP.evals
    E_an = BP.E(np.arange(len(E_num)))
    x_plot = np.linspace(0.0,1.0,1000)

    fig = plt.figure(1)

    # showing the result of the integration on a linear scale -------
    ax1 = fig.add_subplot(121)
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_xlim(-1, int(N_cell/2.0))
    ax1.plot(np.arange(len(E_num)), abs(E_an-E_num), 'ko', label=None)

    i_state=0
    ax2 = fig.add_subplot(122)
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.plot(x_plot, BP.psi(x_plot, n=i_state), 'k-')
    ax2.plot(x_plot, BP.psi_from_coeff(x_plot, m=i_state), 'r+')

    def onclick(event):
        i_state = int(event.xdata)
        print(i_state)
        ax2.lines = []
        ax2.plot(x_plot, BP.psi(x_plot, n=i_state), 'k-')
        ax2.plot(x_plot, BP.psi_from_coeff(x_plot, m=i_state), 'r+')
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
