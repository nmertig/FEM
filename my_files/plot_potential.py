#pylint: disable=C0103
"""
This program calculates the resonances of the fishman map
"""

from __future__ import (division, print_function)
import numpy as np
from matplotlib import pylab as plt

from FEM.core.fem import FEM

# testing stuff ---------------------------------------------------------

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    FIXME: needs to be implemented
    raise NotImplementedError

    def plot_V(self):
        """
        """
        x = np.linspace(self.x_grid.min(), self.x_grid.max(), 1000)
        fig = plt.figure(1)

        # showing the result of the integration on a linear scale -------
        ax1 = fig.add_subplot(111)
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax1.plot(x, self.V_fct_theta(x).real, 'b-', label=None)
        ax1.plot(x, self.V_fct_theta(x).imag, 'r-', label=None)
        plt.show()


