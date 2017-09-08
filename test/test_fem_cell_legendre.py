#pylint: disable=C0103
"""
This is a collection of tests for the code of FEM_Cell_Legendre
"""

from __future__ import (division, print_function)
import numpy as np
from numpy.polynomial import Legendre as LegPol
from matplotlib import pylab as plt

from FEM.core.fem_cell_legendre import FEM_Cell_Legendre

def plot(max_order=8, smin=-1.0, smax=1.0):
    """ Plots the first functions on the finite element cell
    """
    FEM = FEM_Cell_Legendre(max_order=max_order)
    polynomials = []
    derivatives = []
    domain = [smin, smax]

    for m in range(max_order+1):
        polynomials += [FEM.Xsi(m, domain=domain)]
        derivatives += [FEM.dXsi(m, domain=domain)]
        x = np.linspace(smin, smax, 101)

    fig = plt.figure(1)

    # showing the result of the integration on a linear scale -------
    ax1 = fig.add_subplot(121)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    for i in range(max_order+1):
        ax1.plot(x, polynomials[i](x), '-', label=None)

    ax1 = fig.add_subplot(122)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    for i in range(max_order+1):
        ax1.plot(x, derivatives[i](x), '-', label=None)

    plt.show()

def test_analytic_Matrix_Elements(max_order=8, smin=-1.0, smax=1.0):
    """ Compares analytically and numerically determined Matrix elements for
    the Overlap and the Laplace Matrix
    """
    FEM = FEM_Cell_Legendre(max_order=max_order)
    dO = abs(FEM.Overlap_matrix_numeric(smin=smin, smax=smax)
             - FEM.Overlap_matrix(smin=smin, smax=smax)).max()
    print('Overlap Matrix Error:', dO)
    dL = abs(FEM.Laplace_matrix_numeric(smin=smin, smax=smax)
             - FEM.Laplace_matrix(smin=smin, smax=smax)).max()
    print('Laplace Matrix Error:', dL)

# testing the code of the potential matrix

def test_potential_Matrix_Elements1(max_order=8, smin=-1.0, smax=1.0):
    """ Test on constant potentials
    """
    FEM = FEM_Cell_Legendre(max_order=max_order)
    alpha_array = np.linspace(-1.0, 1.0, 11)
    Error = 0.0
    for alpha in alpha_array:
        V = lambda x : alpha
        V_nm = FEM.Potential_matrix(V, smin=smin, smax=smax,
                                    degree=(max_order+1))
        O_nm = FEM.Overlap_matrix(smin=smin, smax=smax)
        Error = max(Error, abs(alpha * O_nm - V_nm).min())
    print('Error for constant potentials:', Error)

def test_potential_Matrix_Elements2():
    """ Test zero element versus integrator results (should be same)
    """
    FEM = FEM_Cell_Legendre(max_order=0)
    # take the potential to be an oscillatory function
    V = lambda x : np.exp(1j*x)
    alpha = -1.0 - 1j*2.0
    beta = -2.0 + 1j*2.0
    gamma = 1.0
    V_int = lambda x : -1j*np.exp(1j*x)*(alpha + beta*x + gamma*x**2)/4.0
    degree = 10
    V_nm = FEM.Potential_matrix(V, smin=-1.0, smax=1.0, degree=degree)
    V_00 = V_int(1.0) - V_int(-1.0)
    print('Error for V_00', abs(V_nm[0,0] - V_00))

def test_K_matrix1(N=10, theta=0.1*np.pi, degree=50,
                  qmin=-3.0, qmax=3.0, Ncell=None):
    """ here, the K matrix is tested based on the fact that
    \int dq' int dq f(q', q) = K[0,0] + K[0,1] + K[1,0] + K[1,1]
    """
    from scipy.special import erf
    # init the legendre cell
    FEM = FEM_Cell_Legendre(max_order=1)
    # set parameters parameters
    hbar = 1.0 / (1.0 * N) / (2.0 * np.pi)
    alpha = np.exp(1j*(theta - np.pi/4.0)) / np.sqrt(2.0 * hbar)
    # define a propagator like expression (to be integrated by K_matrix method)
    def f(x, y, alpha=alpha):
        return alpha * np.exp(- alpha**2 * (x-y)**2) / np.sqrt(np.pi)

    # and its analytical integral
    def F(x0, x1, y0, y1, alpha=alpha):
        z1 = alpha * (y1 - x0)
        z2 = alpha * (y0 - x1)
        z3 = alpha * (y1 - x1)
        z4 = alpha * (y0 - x0)
        F1 = lambda z: z * erf(z) / (2.0 * alpha)
        F2 = lambda z: np.exp(-z**2) / np.sqrt(np.pi) / (2.0 * alpha)
        return F1(z1) + F1(z2) - (F1(z3) + F1(z4)) +\
                    F2(z1) + F2(z2) - (F2(z3) + F2(z4))

    if Ncell is None: # recommended cell size
        # recommended when sclaing angle is zero
        L = qmax - qmin
        Ncell = (L)**2/ (2.0 * np.pi * hbar)
        # for finite scaling angle
        if 30 * np.log(10)/ (2.0 * np.pi * Ncell) < np.sin(2.0*theta):
            print('resetting')
            # resetting Ncell
            Ncell = L/np.sqrt(2.0 * np.pi * hbar) \
                    * np.sqrt(30*np.log(10)/(2.0 * np.pi * np.sin(2.0*theta)))
        Ncell = int(Ncell)
        print('Number of integration cells:', Ncell)

    # set array of errors
    dF = np.zeros(Ncell)
    dq = (qmax - qmin) / float(Ncell)
    smin = qmin
    smax = qmin + dq
    for i in range(Ncell):
        etamin = qmin + i * dq
        etamax = qmin + (i + 1) * dq
        K_nm = FEM.K_matrix(f, smin=smin, smax=smax,
                            etamin=etamin, etamax=etamax,
                            degree=degree)
        F_num = np.sum(K_nm)
        F_analytic = F(etamin, etamax, smin, smax, alpha=alpha)
        dF[i] = abs(F_num - F_analytic)

    print('Error of K matrix:', dF.max())

def test_K_matrix2(hbar=0.1, max_order=3, degree=50,
                   smin=-0.758, smax=0.1, etamin=-0.2, etamax=-0.04):
    """ here, the K matrix is tested on the function f(x,y)=1"""
    # init the legendre cell
    FEM = FEM_Cell_Legendre(max_order=max_order)
    # function
    #def f(x,y, m=0, n=0):
        #return LegPol(m)(y) * LegPol(n)(x)

    f = lambda x, y: (1.0 + 0.0*x + 0.0*y)
    # compute K-Matrix
    K_nm = FEM.K_matrix(f, smin=smin, smax=smax, etamin=etamin, etamax=etamax,
                        degree=degree)
    # for f=1 we have K_nm = int Xsi_n * int Xsi_m
    K_n = np.zeros(max_order+1)
    K_n[1:][::2] = -2.0
    K_n[0] = 1.0
    K_n[-1] = 1.0
    K_n *= (etamax - etamin)/2.0

    K_m = np.zeros(max_order+1)
    K_m[1:][::2] = -2.0
    K_m[0] = 1.0
    K_m[-1] = 1.0
    K_m *= (smax - smin)/2.0

    error = abs(K_nm - K_n[:,np.newaxis] * K_m[np.newaxis,:]).max()
    print('Error of K matrix:', error)

def test_K_matrix3(hbar=0.1, max_order=10, degree=50, smin=-0.758, smax=0.1,
                   etamin=-0.2, etamax=-0.04):
    """ here, the K matrix is tested on the function f(x,y)=1"""
    # init the legendre cell
    FEM = FEM_Cell_Legendre(max_order=max_order)
    # function

    Error_matrix = np.zeros((max_order+1, max_order+1), )

    for m in range(1, max_order):
        print(m)
        coeff_m = np.zeros(m+2)
        coeff_m[-1] = 1
        P_m = LegPol(coeff_m, domain=[etamin, etamax])
        for n in range(1, max_order):
            print(n)
            # integrand
            coeff_n = np.zeros(n+2)
            coeff_n[-1] = 1
            P_n = LegPol(coeff_n, domain=[smin, smax])

            f = lambda y, x: P_m(y) * P_n(x)
            # compute K-Matrix
            K_mn = FEM.K_matrix(f, smin=smin, smax=smax,
                                etamin=etamin, etamax=etamax,
                                degree=degree)
            K = (etamax - etamin)/(2*m+3) * (smax - smin)/(2*n+3)
            K_mn[m,n] = K_mn[m,n] - K
            Error_matrix[m,n] = abs(K_mn).max()
            print(Error_matrix[m,n])
    error = abs(Error_matrix).max()
    print('Error of K matrix:', error)

if __name__ == '__main__':
    """ Give some example which also serves as a test here.
    """
    smin =-0.035
    smax = 1.045
    plot(max_order=10, smin=smin, smax=smax)
    test_analytic_Matrix_Elements(max_order=20, smin=smin, smax=smax)
    test_potential_Matrix_Elements1(max_order=20, smin=smin, smax=smax)
    test_potential_Matrix_Elements2()
    test_K_matrix1()
    test_K_matrix2()
    test_K_matrix3()
