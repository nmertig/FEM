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
        self.H_ = -1j*np.dot(self.Oinv, self.H)/hbar
        self.hbar = hbar
        self.atol = 10**(-10)
        self.rtol = 10**(-10)

    def to_vector(self, M):
        """
        """
        return M.ravel()

    def to_matrix(self, vec):
        """
        """
        return vec.reshape(self.N, self.N)


    # this defines the elements of the kicked system -------------------------
    def F(self, t, x):
        """
        """
        print("hello")
        return self.to_vector(np.dot(self.H_, self.to_matrix(x)))

    def set_integrator(self):
        """
        """
        self.ode = complex_ode(self.F).set_integrator('dop853')
        #,
                                                      #atol=self.atol,
                                                      #rtol=self.rtol)
        # set the initial condition for the propagator
        x0 = self.to_vector(np.diag(np.ones(self.N, dtype=complex)))
        self.ode.set_initial_value(x0, 0.0)

    def integrate(self, dt, nsteps=1):
        """ integrate
        """
        dt *= 1.0 / np.float(nsteps)
        for i in range(nsteps):
            print("time step:", i)
            self.ode.integrate(self.ode.t + dt)
        return self.to_matrix(self.ode.y)

if __name__ == '__main__':
    """ short example of usage
    """
    # Hamiltonian
    hbar = 0.001
    Oinv = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    H = np.array([[1.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 2.0]])

    t1 = 2.0 * np.pi * hbar * 0.781
    RK = RungeKutta(H, Oinv, hbar)
    RK.set_integrator()
    U = RK.integrate(t1)
    print("Unitarity:", np.dot(U.transpose().conjugate(), U))
    print("final point:", RK.integrate(t1))