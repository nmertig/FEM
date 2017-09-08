#pylint: disable=C0103
"""
This program calculates the resonances of the fishman map
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt

from FEM.core.filename import fname
from FEM.systems.fishman import Fishman

# testing stuff ---------------------------------------------------------

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    kappa=1.0
    degree = 50
    degree2D = 50

    # change several parameters
    max_order_array = np.array([5, 10, 15])
    theta_array = np.array([0.125, 0.1, 0.01]) * np.pi
    L_array = np.array([6.0, 10.0, 20.0])
    hbar_array = 1.0 / (2.0 * np.pi) / np.array([5, 10, 15])

    for i_max_order in range(len(max_order_array)):
        max_order = max_order_array[i_max_order]
        for i_theta in range(len(theta_array)):
            theta = theta_array[i_theta]
            for i_L in range(len(L_array)):
                L = L_array[i_L]
                for i_hbar in range(len(hbar_array)):
                    hbar = hbar_array[i_hbar]

                    # set the file name
                    filename = fname(kappa, hbar, theta, max_order, L)
                    print(filename)

                    # initialize the fishmen map ----------------------------
                    Ncell = 6
                    FM = Fishman(hbar, max_order, Ncell, L, kappa, theta=theta)
                    Ncell = FM.recommend_N()
                    print("New grid size:", Ncell)
                    # reinitialize the map
                    FM = Fishman(hbar, max_order, Ncell, L, kappa, theta=theta)

                    # set up ---------------------------------------
                    FM.setup_O()
                    FM.setup_Oinv()
                    FM.setup_UV_half(degree=degree)
                    FM.setup_UT_fast(degree=degree2D, eps=10**(-15))
                    # prepare the full propagation matrix ----------
                    UV_half = np.dot(FM.Oinv, FM.UV_half)
                    UT = np.dot(FM.Oinv, FM.UT)
                    FM.U = np.dot(FM.UV_half, np.dot(UT, UV_half))

                    FM.compute_evecs()
                    FM.order_evecs()

                    #
                    data_array = [FM.O, FM.UT, FM.UV_half, FM.U,
                                  FM.evals, FM.L, FM.R, FM.x_grid]

                    np.save(filename, data_array)