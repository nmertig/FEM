#pylint: disable=C0103
"""
This Program computes the resonances of the bain system and plots them
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt

from FEM.systems.bain import Bain


if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    hbar = 1.0
    max_order = 9
    N1 = 50
    L1 = 10.0
    N2 = 10
    L2 = 30.0
    theta = 0.25 * np.pi
    BP = Bain(hbar, max_order, N1, L1, N2, L2, theta=theta,
              dirichlet_bound_conds=True)

    # set up
    #BP.plot_V()
    BP.setup_O()
    BP.setup_T()
    BP.setup_V(degree=50)
    BP.setup_H()
    #BP.impose_boundary_condition()

    # diagonalize
    BP.compute_evecs()
    BP.order_evecs()
    BP.normalize_R()
    #BP.embedd_evecs()

    # show the results ----------------------------------------------
    E_num = BP.evals
    x_plot = np.linspace(0.0,L2,1000)
    fig = plt.figure(1)

    # showing the result of the integration on a linear scale -------
    ax1 = fig.add_subplot(121)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.set_xlim(-3, 10)
    ax1.set_ylim(-20, 2)
    ax1.plot(E_num.real, E_num.imag, 'ko', label=None)

    i_state=0
    ax2 = fig.add_subplot(122)
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.set_xlim(0, L2)
    #ax2.set_ylim(10**(-30), 10)
    ax2.plot(x_plot, abs(BP.psi_from_coeff(x_plot, m=i_state))**2, 'k-')

    def onclick(event):
        x, y = event.xdata, event.ydata
        print(x,y)
        dE = np.sqrt(abs(BP.evals.real - x)**2 + abs(BP.evals.imag - y)**2)
        i = np.argsort(dE)[0]
        print(BP.evals[i])
        ax2.lines = []
        ax2.plot(x_plot, abs(BP.psi_from_coeff(x_plot, m=i))**2, 'k-')
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()






