#pylint: disable=C0103
"""
This class describes the Gaussian Wave Packet Dynamics of a free particle
in the complex rotated space
"""

from __future__ import (division, print_function)
import numpy as np
from scipy.integrate import complex_ode
from scipy.linalg import solve

class RungeKutta(object):
    """
    """
    def __init__(self, H, Oinv, hbar):
        """
        """
        self.N = len(H[0,:])
        self.N2 = self.N**2
        self.H = H
        self.Oinv = Oinv
        self.H_ = -1j * np.dot(self.Oinv, self.H) / hbar
        self.hbar = hbar
        self.atol = 10**(-10)
        self.rtol = 10**(-10)

    # this defines the elements of the kicked system -------------------------
    def F(self, t, x):
        """
        """
        return np.dot(self.H_, x)

    def set_integrator(self):
        """
        """
        self.ode = complex_ode(self.F).set_integrator('dop853',
                                                      max_step=1000,
                                                      atol=self.atol,
                                                      rtol=self.rtol)

    def set_initial_value(self, x0):
        # set the initial condition for the propagator
        self.ode.set_initial_value(x0, 0.0)

    def integrate(self, dt, nsteps=1):
        """ integrate
        """
        dt *= 1.0 / np.float(nsteps)
        for i in range(nsteps):
            print("time step:", i)
            self.ode.integrate(self.ode.t + dt)
        return self.ode.y

if __name__ == '__main__':
    """ short example of usage
    """
    # Hamiltonian
    hbar = 0.001
    Oinv = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    H = np.array([[1.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 2.0]])
    x0 = np.array([0.0, 0.0, 1.0])
    dt = 2.0 * np.pi * hbar

    RK = RungeKutta(H, Oinv, hbar)
    RK.set_integrator()
    RK.set_initial_value(x0)
    x_t = RK.integrate(dt)
    print("initial state:", x0)
    print("final state:", x_t)