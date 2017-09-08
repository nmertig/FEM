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

                    # reconstruct the data
                    data_array = np.load(filename)
                    FM.evals = data_array[4]
                    FM.R = data_array[5]

                    # plotting the results ------------------
                    x_plot = np.linspace(-4.0, 4.0, 1001)

                    ang = np.linspace(0.0, 2.0 * np.pi, 100)
                    x_circ = np.cos(ang)
                    y_circ = np.sin(ang)

                    fig = plt.figure(1)


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
                    ax2.set_yscale('log')
                    ax2.plot(x_plot,
                             abs(FM.psi_from_coeff(x_plot, m=i_state)), 'k-')

                    def onclick(event):
                        x, y = event.xdata, event.ydata
                        print(x,y)
                        dE = np.sqrt(abs(FM.evals.real - x)**2\
                           + abs(FM.evals.imag - y)**2)
                        i = np.argsort(dE)[0]
                        print("Eigenvalue:", FM.evals[i])
                        print("Decay rate:", FM.gamma[i])
                        ax2.lines = []
                        ax2.plot(x_plot, abs(FM.psi_from_coeff(x_plot, m=i)),
                                 'k-')
                        plt.draw()

                    cid = fig.canvas.mpl_connect('button_press_event', onclick)

                    plt.show()