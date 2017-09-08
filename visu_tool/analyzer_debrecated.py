"""test for Hreg/hreg_systems/hreg_harmonic_oscillator.py
"""
# pylint: disable=C0103
from __future__ import (division, print_function)
import numpy as np
import matplotlib.pyplot as plt

import os

def gamma(evals):
    """
    """
    return -2.0 * np.log(abs(evals))

def phases(evals):
        """
        """
        phases = np.arctan2(evals.imag, evals.real)
        return phases

class Analyzer(object):
    """
    """
    def __init__(self, data_path, system=None, x_range=[-0.7, 0.7],
                 ext=False, N_evals=20):
        """
        """
        self.data_path = data_path
        self.system=system
        self.ext = ext
        self.x_plot = np.linspace(x_range[0], x_range[1], 1001)
        self.N_evals = N_evals

        # to be filled ---------------------
        self.fname_array = None
        self.parameter_array = None
        self.N_params = None

        self.im = None

        self.i = None
        self.m = None
        self.active_window=None

        # available collors
        self.col = ['k', 'b', 'r', 'g', 'm', 'y']

    # ######################################################################
    # loading the available data values
    # ######################################################################
    def get_parameter_array(self, parameter_index=1):
        """
        """
        # first we check the data of all available computations
        fname_array = next(os.walk(self.data_path + "/params/"))[2]
        # and initialize the parameter array
        self.N_params = len(fname_array)
        parameter_array = np.zeros(self.N_params, dtype=float)

        # now we fill the parameter array
        for i in range(self.N_params):
            parameter_array[i] =\
                np.load(self.data_path + "/params/" +
                        fname_array[i])[parameter_index]

        # save the relevant information
        i_sort = np.argsort(1.0/parameter_array)
        self.parameter_array = parameter_array[i_sort]
        self.fname_array = [fname_array[i] for i in i_sort]

    # ######################################################################
    # book keeping of im
    # ######################################################################
    def init_index_array(self):
        """
        """
        self.im = np.ones(self.N_params,dtype=int)[:,np.newaxis]*\
                  np.arange(self.N_evals,dtype=int)[np.newaxis,:]

    def load_index_array(self):
        """
        """
        self.im = np.load(self.data_path + "index_array.npy")

    def save_index_array(self):
        """
        """
        np.save(self.data_path + "index_array.npy", self.im)

    def m_data(self, i, m_state):
        """
        """
        return self.im[i, m_state]

    def m_state(self, i, m_data):
        """
        """
        return np.where(self.im[i,:] == m_data)[0][0]

    def load(self):
        """
        """
        fname = self.fname_array[self.i]
        params = np.load(self.data_path + "/params/" + fname)
        if not self.ext:
            kappa = params[0]
            hbar = params[1]
            max_order = int(params[2])
            N = int(params[3])
            x_max = params[0]
            theta = params[0]
            self.FM = self.system(kappa, hbar, max_order, N, x_max,
                                  theta=theta)
        else:
            pass

        self.FM.evals = np.load(self.data_path + "/evals/" + fname)
        fname = fname.replace(".npy", "_R.npy")
        self.FM.R = np.load(self.data_path + "/evecs/" + fname)
        #fname = fname.replace("_R.npy", "_L.npy")
        #self.FM.L = np.load(self.data_path + "/evecs/" + fname)

    def load_evals(self):
        """
        """
        self.evals = np.ones((self.N_params, self.N_evals), dtype=complex)
        for i in range(self.N_params):
            fname = self.fname_array[i]
            evals = np.load(self.data_path + "/evals/" + fname)
            if len(evals) > self.N_evals:
                self.evals[i, :] = evals[:self.N_evals]
            else:
                self.evals[i,:len(evals)] = evals

    def plot_data(self):
        """
        """
        self.i = 0 # index of possible hbar
        self.m = 0 # index of possible states
        self.active_window=1
        self.parameter_info()

        self.load_evals()
        self.load()

        # prepare the level figures ---------------------------------------
        self.fig = plt.figure(1)

        # plotting the spectrum
        self.ax1 = self.fig.add_subplot(221)
        self._plot_evals()

        # wave function
        self.ax2 = self.fig.add_subplot(223)
        self._plot_evecs()

        # decay rates over hinv
        self.ax3 = self.fig.add_subplot(222)
        self.ax3.set_yscale('log')
        self._plot_gammas()

        # phases over hinv
        self.ax4 = self.fig.add_subplot(224)
        self._plot_phases()

        self.cid_key = \
        self.fig.canvas.mpl_connect('key_press_event', self._key_press)
        self.cid_mouse = \
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        plt.show()

    def _on_click(self, event):
        """
        """
        if event.button == 1:
            print("resetting to data in window:", self.active_window)
            if self.active_window == 1:
                print('evals', event.xdata, event.ydata)
                evals = self.FM.evals[0:self.N_evals]
                dx = (event.xdata - evals.real)**2
                dy = (event.ydata - evals.imag)**2
                dE = np.sqrt(dx + dy)
                self.m = self.m_state(self.i, np.argmin(dE))

            elif self.active_window == 3:
                print('gammas', event.xdata, event.ydata)
                x = 1.0/2.0/np.pi/self.parameter_array
                self.i = np.argmin(abs(x-event.xdata))
                y = gamma(self.evals[self.i,:])
                m_data = np.argmin(abs(np.log(y/event.ydata)))
                self.m = self.m_state(self.i, m_data)

            elif self.active_window == 4:
                print('phases', event.xdata, event.ydata)

            else:
                pass

            self.re_plot()

        else:
            return

    def _key_press(self, event):
        """
        """
        if event.key == 'n':
            self.i = (self.i + 1)%(self.N_params)

        elif event.key == 'p':
            self.i = (self.i - 1)%(self.N_params)

        elif event.key == 'r':
            self.m = (self.m + 1)%(self.N_evals)

        elif event.key == 'l':
            self.m = (self.m - 1)%(self.N_evals)

        elif event.key == 'z':
            self._set_new_index()

        elif event.key == 'x':
            self._connect_phases_up()

        elif event.key == 'c':
            self._connect_phases_down()

        # active windows
        elif event.key == '1':
            self.active_window=1
        elif event.key == '3':
            self.active_window=3
        elif event.key == '4':
            self.active_window=4

        elif event.key == 'q':
            plt.close()
            return

        else:
            pass

        self.re_plot()

    def re_plot(self):
        """
        """
        self.parameter_info()
        self.load()
        self._update_evals()
        self._update_evecs()
        self._update_gammas()
        self._update_phases()
        plt.draw()

    def parameter_info(self):
        """
        """
        print('-----------------------------------------------')
        print('active window:', self.active_window)
        print("current data index i", self.i, "/", self.N_evals)
        print("current state index", self.m, "/", self.N_params)
        print("current data index m", self.m_data(self.i, self.m))
        print("current state index from data index",
              self.m_state(self.i, self.m_data(self.i, self.m)))

    # #####################################################################
    #
    # #####################################################################
    def _plot_evals(self):
        """
        """
        self.ax1.plot(self.FM.evals.real,
                      self.FM.evals.imag, 'ko', label=None)
        self.ax1.plot([self.FM.evals.real[self.m_data(self.i, self.m)]],
                      [self.FM.evals.imag[self.m_data(self.i, self.m)]],
                      'ro', label=None)
        # add a unit circle for orientation
        ang = np.linspace(0.0, 2.0 * np.pi, 100)
        x_circ = np.cos(ang)
        y_circ = np.sin(ang)
        self.ax1.plot(x_circ, y_circ, 'k-', label=None)

    def _update_evals(self):
        """
        """
        self.ax1.lines = []
        self._plot_evals()

    # #####################################################################
    #
    # #####################################################################
    def _plot_evecs(self):
        """
        """
        self.ax2.plot(self.x_plot, abs(self.FM.psi_from_coeff(self.x_plot,
                      m=self.m_state(self.i, self.m))), 'k-')

    def _update_evecs(self):
        """
        """
        self.ax2.lines = []
        self._plot_evecs()

    # #####################################################################
    #
    # #####################################################################
    def _plot_gammas(self):
        """
        """
        i_array = np.arange(self.N_params)
        for m in range(self.N_evals):
            self.ax3.plot(1.0/2.0/np.pi/self.parameter_array,
                          gamma(self.evals[i_array, self.im[i_array,m]]),
                          '+', color = '0.75')
        # current line
        self.ax3.plot(1.0/2.0/np.pi/self.parameter_array,
                      gamma(self.evals[i_array, self.im[i_array,self.m]]),
                      'ko')
        # current data point
        self.ax3.plot([1.0/2.0/np.pi/self.parameter_array[self.i]],
                      [gamma(self.evals[self.i, self.im[self.i, self.m]])],
                      'ro')

    def _update_gammas(self):
        """
        """
        self.ax3.lines = []
        self._plot_gammas()

    # #####################################################################
    #
    # #####################################################################
    def _plot_phases(self):
        """
        """
        i_array = np.arange(self.N_params)
        for m in range(self.N_evals):
            self.ax4.plot(1.0/2.0/np.pi/self.parameter_array,
                          phases(self.evals[i_array, self.im[i_array,m]]),
                          '+', color = '0.75')
        # current line
        self.ax4.plot(1.0/2.0/np.pi/self.parameter_array,
                      phases(self.evals[i_array, self.im[i_array,self.m]]),
                      'ko')
        # current data point
        self.ax4.plot([1.0/2.0/np.pi/self.parameter_array[self.i]],
                      [phases(self.evals[self.i, self.im[self.i, self.m]])],
                      'ro')

    def _update_phases(self):
        """
        """
        self.ax4.lines = []
        self._plot_phases()

    # #####################################################################
    #
    # #####################################################################
    def _swap_index(self, m_is_currently, m_should_be, i=None):
        """ swaps state indices in the state index to data index matrix
        """
        if i is None:
            i = self.i
        self.im[i, np.array([m_should_be, m_is_currently])] =\
            self.im[i, np.array([m_is_currently, m_should_be])]

    def _set_new_index(self):
        """
        """
        print('assign a new index')
        m_should_be = int(raw_input())
        print("index should be:", m_should_be)
        print("will be exchanged with:", self.m)
        self._swap_index(self.m, m_should_be)
        self.m = m_should_be

    def _connect_phases_up(self):
        """
        """
        for i in range(self.i, self.N_params-1):
            phi0 = phases(self.evals[i, self.m_data(i, self.m)])
            phi_ = phases(self.evals[i+1,:])
            dphi = (phi_ - phi0 + np.pi)%(2.0 * np.pi) - np.pi
            m_data_closest_phi = np.argmin(abs(dphi))
            m_state_closest_phi = self.m_state(i+1, m_data_closest_phi)
            self._swap_index(m_state_closest_phi, self.m, i=i+1)

    def _connect_phases_down(self):
        """
        """
        for di in range(0, self.i):
            i = self.i - di
            phi0 = phases(self.evals[i, self.m_data(i, self.m)])
            phi_ = phases(self.evals[i-1,:])
            dphi = (phi_ - phi0 + np.pi)%(2.0 * np.pi) - np.pi
            m_data_closest_phi = np.argmin(abs(dphi))
            m_state_closest_phi = self.m_state(i-1, m_data_closest_phi)
            self._swap_index(m_state_closest_phi, self.m, i=i-1)

    def _connect_evals_up(self):
        """
        """
        for i in range(self.i, self.N_params-1):
            u0 = self.evals[i, self.m_data(i, self.m)]
            u_ = self.evals[i+1,:]
            du = u_ - u0
            m_data_closest_phi = np.argmin(abs(du))
            m_state_closest_phi = self.m_state(i+1, m_data_closest_phi)
            self._swap_index(m_state_closest_phi, self.m, i=i+1)

    def _connect_evals_down(self):
        """
        """
        for di in range(0, self.i):
            i = self.i - di
            u0 = self.evals[i, self.m_data(i, self.m)]
            u_ = self.evals[i-1,:]
            du = u_ - u0
            m_data_closest_phi = np.argmin(abs(du))
            m_state_closest_phi = self.m_state(i-1, m_data_closest_phi)
            self._swap_index(m_state_closest_phi, self.m, i=i-1)