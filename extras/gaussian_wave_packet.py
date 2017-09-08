#pylint: disable=C0103
"""
This class describes the Gaussian Wave Packet Dynamics of a free particle
in the complex rotated space
"""

from __future__ import (division, print_function)
import numpy as np

class GWP(object):
    """
    """
    def __init__(self, q0, p0, sigma, hbar, theta):
        """
        """
        self.hbar = hbar
        self.q0 = q0
        self.p0 = p0
        self.sigma = sigma
        self.hbar = hbar
        self.theta = theta

    # this defines the elements of the kicked system -------------------------
    def abs_sigma_t(self, t):
        """
        """
        x = self.sigma**2 + self.hbar * t * np.sin(2*self.theta)/2.0
        y = self.hbar * t * np.cos(2*self.theta)/2.0
        return (x**2 + y**2)**(0.25)

    def phi_sigma_t(self, t):
        """
        """
        x = self.sigma**2 + self.hbar * t * np.sin(2*self.theta)/2.0
        y = self.hbar * t * np.cos(2*self.theta)/2.0
        return 0.5 * np.arctan2(y, x)

    def sigma_t(self, t):
        """ time dependant complex width parameter
        """
        return self.abs_sigma_t(t) * np.exp(1j*self.phi_sigma_t(t))

    def q_t(self, t):
        """
        """
        return self.q0 + self.p0 * t * np.exp(-1j*2.0*self.theta)

    def psi(self, q, t=0):
        """
        """
        pre_fac = np.sqrt(self.sigma / np.sqrt(2*np.pi)) / self.sigma_t(t)
        x = q-self.q_t(t)
        phase1 = -(x / (2.0 * self.sigma_t(t)))**2
        phase2 = 1j * self.p0 * x / self.hbar
        phase3 = 1j * self.p0 * self.q_t(t) / 2.0 /self.hbar
        return pre_fac * np.exp(phase1 + phase2 + phase3)