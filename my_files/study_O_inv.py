#pylint: disable=C0103
"""
This program investigates the inversion of the overlap Matrix O
"""

from __future__ import (division, print_function)
import numpy as np
from numpy.linalg import eig, eigh
from matplotlib import pylab as plt

# we proceed
from FEM.core.fem_kicked import FEMKicked as FEM

# testing stuff ---------------------------------------------------------

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    hbar = 1.0
    max_order = 10
    x_grid = np.linspace(-1.0, 1.0, 20)
    theta = 0.1

    fem1 = FEM(hbar, max_order, x_grid, theta=theta,
              dirichlet_bound_conds=False)
    fem2 = FEM(hbar, max_order, x_grid, theta=0.1,
               dirichlet_bound_conds=False)
    fem3 = FEM(hbar, max_order, x_grid, theta=0.1,
               dirichlet_bound_conds=True)

    # setting up O and its inverse
    fem1.setup_O()
    fem1.setup_Oinv(fast=True)
    Oinv1 = fem1.Oinv

    fem2.O = fem1.O.copy()
    fem2.setup_Oinv(fast=False, show_modes=False)
    Oinv2 = fem2.Oinv

    fem3.O = fem1.O.copy()
    fem3.setup_Oinv(fast=False, show_modes=False)
    Oinv3 = fem3.Oinv

    indexes = np.arange(fem1.dim)
    index_shift = np.concatenate((indexes[2:], indexes[:2]))

    # diagonalizing via hermitian Hamiltonian solver ----------------------
    evals1, evecs1 = eig(Oinv1)
    i_sort = np.argsort(evals1)
    evals1 = evals1[i_sort]
    fem1.R = evecs1[:,i_sort]
    # diagonalizing of Oinv ----------------------
    evals2, evecs2 = eig(Oinv2)
    i_sort = np.argsort(evals2)
    evals2 = evals2[i_sort][index_shift]
    fem2.R = evecs2[:,i_sort][:,index_shift]
    # diagonalizing of Oinv ----------------------
    evals3, evecs3 = eig(Oinv3)
    i_sort = np.argsort(evals3)
    evals3 = evals3[i_sort][index_shift]
    fem3.R = evecs3[:,i_sort][:,index_shift]

    fig = plt.figure(1)
    x_plot = np.linspace(-1.1, 1.1, 50*len(x_grid))

    ax1 = fig.add_subplot(221)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    #ax1.plot(np.arange(len(evals1)), evec_diff, 'y-', label=None)
    ax1.plot(np.arange(len(evals1)), evals1, 'ks', label=None)
    ax1.plot(np.arange(len(evals2)), evals2, 'ro', label=None)
    ax1.plot(np.arange(len(evals3)), evals3, 'b+', label=None)

    i_state=0
    ax2 = fig.add_subplot(222)
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.plot(x_plot, abs(fem1.psi_from_coeff(x_plot, m=i_state)), 'k-')
    ax2.plot(x_plot, abs(fem2.psi_from_coeff(x_plot, m=i_state)), 'r-')
    ax2.plot(x_plot, abs(fem3.psi_from_coeff(x_plot, m=i_state)), 'b-')

    ax3 = fig.add_subplot(223)
    ax3.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.plot(np.arange(fem1.dim), fem1.R[:,i_state].real, 'ks')
    ax3.plot(np.arange(fem2.dim), fem2.R[:,i_state].real, 'ro')
    ax3.plot(np.arange(fem3.dim), fem3.R[:,i_state].real, 'b+')

    ax4 = fig.add_subplot(224)
    ax4.set_xscale('linear')
    ax4.set_yscale('linear')
    ax4.plot(np.arange(fem1.dim), fem1.R[:,i_state].imag, 'ks')
    ax4.plot(np.arange(fem2.dim), fem2.R[:,i_state].imag, 'ro')
    ax4.plot(np.arange(fem3.dim), fem3.R[:,i_state].imag, 'b+')

    def onclick(event):
        x, y = event.xdata, event.ydata
        i_state = int(x)

        ax2.lines = []
        ax2.plot(x_plot, abs(fem1.psi_from_coeff(x_plot, m=i_state)), 'k-')
        ax2.plot(x_plot, abs(fem2.psi_from_coeff(x_plot, m=i_state)), 'r-')
        ax2.plot(x_plot, abs(fem3.psi_from_coeff(x_plot, m=i_state)), 'b-')

        ax3.lines = []
        ax3.plot(np.arange(fem1.dim), fem1.R[:,i_state].real, 'ks')
        ax3.plot(np.arange(fem2.dim), fem2.R[:,i_state].real, 'ro')
        ax3.plot(np.arange(fem3.dim), fem3.R[:,i_state].real, 'b+')

        ax4.lines = []
        ax4.plot(np.arange(fem1.dim), fem1.R[:,i_state].imag, 'ks')
        ax4.plot(np.arange(fem2.dim), fem2.R[:,i_state].imag, 'ro')
        ax4.plot(np.arange(fem3.dim), fem3.R[:,i_state].imag, 'b+')

        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()