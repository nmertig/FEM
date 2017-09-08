#pylint: disable=C0103
"""
This program visualizes eigenstates of the overlap matrix O
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt

# testing stuff ---------------------------------------------------------

class ModesO(object):
    """
    """
    def __init__(self, fem_kicked_instance, evals, evecs):
        """
        """
        self.fem = fem_kicked_instance
        self.evecs
        self.evals

    fig = plt.figure(1)
    x_plot = np.linspace(-1.0, 1.0, 50*len(x_grid))

    ax1 = fig.add_subplot(221)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    #ax1.set_xlim(-1, 1)
    #ax1.set_ylim(-1, 1)
    ax1.plot(np.arange(len(evals1)), evec_diff, 'y-', label=None)
    ax1.plot(np.arange(len(evals1)), evals1, 'ko', label=None)
    ax1.plot(np.arange(len(evals2)), evals2, 'r+', label=None)
    #ax1.plot(np.arange(len(evals3)), evals3, 'bs', label=None)

    i_state=0
    ax2 = fig.add_subplot(222)
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.plot(x_plot, abs(fem.psi_from_coeff(x_plot, m=i_state)), 'k-')
    ax2.plot(x_plot, abs(fem2.psi_from_coeff(x_plot, m=i_state)), 'r-')
    #ax2.plot(x_plot, abs(fem3.psi_from_coeff(x_plot, m=i_state)), 'bs')

    ax3 = fig.add_subplot(223)
    ax3.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.plot(np.arange(fem.dim), fem.R[:,i_state].real, 'ko')
    ax3.plot(np.arange(fem.dim), fem2.R[:,i_state].real, 'r+')

    ax4 = fig.add_subplot(224)
    ax4.set_xscale('linear')
    ax4.set_yscale('linear')
    ax4.plot(np.arange(fem.dim), fem.R[:,i_state].imag, 'ko')
    ax4.plot(np.arange(fem.dim), fem2.R[:,i_state].imag, 'r+')

    def onclick(event):
        x, y = event.xdata, event.ydata
        i_state = int(x)

        ax2.lines = []
        ax2.plot(x_plot, abs(fem.psi_from_coeff(x_plot, m=i_state)), 'k-')
        ax2.plot(x_plot, abs(fem2.psi_from_coeff(x_plot, m=i_state)), 'r-')

        ax3.lines = []
        ax3.plot(np.arange(fem.dim), fem.R[:,i_state].real, 'ko')
        ax3.plot(np.arange(fem.dim), fem2.R[:,i_state].real, 'r+')

        ax4.lines = []
        ax4.plot(np.arange(fem.dim), fem.R[:,i_state].imag, 'ko')
        ax4.plot(np.arange(fem.dim), fem2.R[:,i_state].imag, 'r+')

        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()