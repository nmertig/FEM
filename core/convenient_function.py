#pylint: disable=C0103
"""
This is a collection of convenient functions for fem routines
"""

from __future__ import (division, print_function)
import numpy as np

def recommended_N(hbar, theta, box_size, check_box_size=True):
    """
    Based on: hbar, scaling_angle, and box size, we compute
    -- the maximal momentum projected to 10**(-15)
    -- the shortest oscillation length
    -- the number of recommended grid points
    Finally, pmax can be interpreted as a damping_length. If it exeeds
    the box_size, we recommend an increase of it
    """
    pmax = np.sqrt(hbar * 60 * np.log(10) / np.sin(2.0*theta))
    print("The maximal momentum for your system is:", pmax)
    min_osc_length = (2.0 * np.pi * hbar) / pmax
    print("The minimal oscillation for your system is:", min_osc_length)
    N = int(box_size / min_osc_length) + 1
    print("recommended number of grid points:", N)
    # finally we check the ratio of pmax and
    if pmax > box_size:
        print("We recommend increasing 'box_size' beyond:", pmax)
    return N

def recommended_N_FM(hbar, theta, box_size, kappa):
    """ same as above, including the oscillation length of the fishmen map
    """
    h = 2.0 * np.pi * hbar
    min_osc_length = h * 8.0 * np.exp(0.5)/kappa
    print("The minimal oscillation length of Fishman Potential:",
          min_osc_length)
    N = int(box_size / min_osc_length) + 1
    print("recommended number of grid points for pot oscillation:", N)
    return N
