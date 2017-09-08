#pylint: disable=C0103
"""
The goal of this program is to plot the time evolution matrix of U and check its
banded-ness.
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt

from FEM.core.convenient_function import recommended_N
from FEM.systems.fishman_absorbing_potential import Fishman
from qmaps.maps.fishman_half_kick_projector import \
                                      FishmanHalfKickProj as FishmanHalfKick

# testing stuff ---------------------------------------------------------

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    # system parameters ---------------------------------------
    kappa=1.0
    states_per_cell = 5
    hbar = 1.0 / (2.0 * np.pi) / states_per_cell

    qabs = 2.5
    width = 0.1
    amp = 0.0       # the absorption matrix is practically zero
    poly_order = 3

    # computational parameters --------------------------------
    degree = 50
    degree2D = 50
    max_order = 10
    x_max = 10.0
    p_max = 2.0
    N = int(4 * p_max * x_max * states_per_cell)
    print("Using grid points:", N)

    # initialize the fishmen map ----------------------------
    FM = Fishman(kappa, hbar, max_order, N, x_max,
                 qabs, amp, width, poly_order, dirichlet_bound_conds=False)

    # set up ---------------------------------------
    FM.generateU(method=0, degree=degree, degree2D=degree2D, eps=10**(-15),
                 Oinv_fast=True, Oinv_max_eval=None, Oinv_show_modes=False,
                 show_matrix=False)

    # plotting the resulting unitary matrix ---------------
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.imshow(abs(FM.U), interpolation='none')
    ax1 = fig.add_subplot(122)
    ax1.imshow(np.log(abs(FM.U)), interpolation='none')
    plt.show()

    # compare with quantum map ----------------------------
    qmap = FishmanHalfKick(N, kappa, qabs, amp, width, poly_order,
                           2, delta=0.5,
                           Mqp=(int(2*x_max), int(2.0*p_max)),
                           qmin=-x_max, pmin=-p_max)
    qmap.generateU()

    # plotting the resulting unitary matrix ---------------
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.imshow(abs(qmap.U), interpolation='none')
    ax1 = fig.add_subplot(122)
    ax1.imshow(np.log(abs(qmap.U)), interpolation='none')
    plt.show()

