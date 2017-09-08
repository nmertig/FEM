#pylint: disable=C0103
"""
This program calculates the resonances of the fishman map
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt

from FEM.core.convenient_function import recommended_N
from FEM.systems.fishman_exterior_scaling import Fishman

# testing stuff ---------------------------------------------------------

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    # system parameters ---------------------------------------
    kappa=1.0
    states_per_cell = 5
    hbar = 1.0 / (2.0 * np.pi) / states_per_cell

    # computational parameters --------------------------------
    degree = 50
    degree2D = 50
    max_order = 10
    theta = 0.125 * np.pi

    scaling_radius = 0.5
    x_max = 4.0
    N_int = recommended_N(hbar, theta, 2.0*scaling_radius, check_box_size=False)
    N_ext = int(recommended_N(hbar, theta, 2.0*x_max)/2.0) + 1
    print("Number of interior cells:", N_int)
    print("Number of exterior cells:", N_ext)

    # initialize the fishmen map ----------------------------
    FM = Fishman(kappa, hbar, max_order, scaling_radius, N_int, x_max, N_ext,
                 theta=theta)
    # set up ---------------------------------------
    FM.setup_O()
    FM.setup_Oinv()
    FM.setup_UT_te()
    FM.setup_UV_half_int(degree=degree)

    # prepare the full propagation matrix ----------
    UV_half = np.dot(FM.Oinv, FM.UV_half)
    UT = FM.UT
    FM.U = np.dot(FM.UV_half, np.dot(UT, UV_half))

    FM.compute_evecs()
    FM.order_evecs()

    # ####################################################################
    # plotting the results -----------------------------------------------
    x_plot = np.linspace(-4.0, 4.0, 1001)

    ang = np.linspace(0.0, 2.0 * np.pi, 100)
    x_circ = np.cos(ang)
    y_circ = np.sin(ang)

    fig = plt.figure(1)

    # showing the result of the integration on a linear scale -------
    ax1 = fig.add_subplot(121)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    #ax1.set_xlim(-1, 1)
    #ax1.set_ylim(-1, 1)
    ax1.plot(FM.evals.real, FM.evals.imag, 'ko', label=None)
    ax1.plot(x_circ, y_circ, 'k-', label=None)

    i_state=0
    ax2 = fig.add_subplot(122)
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.plot(x_plot, abs(FM.psi_from_coeff(x_plot, m=i_state)), 'k-')

    def onclick(event):
        x, y = event.xdata, event.ydata
        print(x,y)
        dE = np.sqrt(abs(FM.evals.real - x)**2 + abs(FM.evals.imag - y)**2)
        i = np.argsort(dE)[0]
        print("Eigenvalue:", FM.evals[i])
        print("Decay rate:", -2.0 * np.log(abs(FM.evals[i])))
        ax2.lines = []
        ax2.plot(x_plot, abs(FM.psi_from_coeff(x_plot, m=i)), 'k-')
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


