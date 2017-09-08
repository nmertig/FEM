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
    def __init__(self, data_path, N_evals, x_plot):
        """
        """
        self.data_path = data_path
        self.N_evals = N_evals
        self.Nx_plot = len(x_plot)
        self.x_plot = x_plot

        # to be filled ---------------------
        self.fname_array = None
        self.parameter_array = None
        self.N_params = None
        self.evals = None
        self.evecs = None

        # indexes
        self.i = 0
        self.k = 0
        self.m = 0
        self.N_max = 20
        assert(self.N_max < self.N_evals)

        # for plotting
        self.active_window=None

        # available collors
        self.col = ['b', 'g', 'y', 'm', 'c']

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
    # loading the data
    # ######################################################################
    def load(self):
        """ loads the fully available data for the current parameter value
        """
        fname = self.fname_array[self.i]
        self.evals = np.load(self.data_path + "/evals/" + fname)
        fname = fname.replace(".npy", "_R.npy")
        self.evecs = np.load(self.data_path + "/evecs/" + fname)

    def load_eval_array(self):
        """ (level spaghetti)
        loads an eigenvalue array for all parameter values
        """
        self.eval_array = np.ones((self.N_params, self.N_evals),
                                  dtype=complex)
        for i in range(self.N_params):
            fname = self.fname_array[i]
            evals = np.load(self.data_path + "/evals/" + fname)
            self.eval_array[i, :] = evals[:self.N_evals]

    # ######################################################################
    # book keeping of indexes
    # ######################################################################
    def init_k_im(self):
        """
        """
        self.k_im = np.ones(self.N_params, dtype=int)[:,np.newaxis]*\
                            np.arange(self.N_evals, dtype=int)[np.newaxis,:]

    def load_k_im(self):
        """
        """
        self.k_im = np.load(self.data_path + "diabatic_line_index_array.npy")

    def save_k_im(self):
        """
        """
        np.save(self.data_path + "diabatic_line_index_array.npy", self.k_im)

    def clean_up_k_im(self):
        """
        """
        for i in range(self.N_params):
            m_ = self.m+1
            for k in range(self.N_evals):
                if k in self.k_im[i, :(self.m+1)]:
                    pass
                else:
                    self.k_im[i, m_] = k
                    m_ += 1

    def update_k_im(self, i=None, k=None, force=False):
        """
        """
        if i is None:
            i = self.i
            print(i)
        if k is None:
            k = self.k
        old_entry = self.k_im[i, self.m]
        m_of_new_entry = np.argmin(abs(self.k_im[i, :] - k))

        if force:
            self.k_im[i, self.m] = k
            self.k_im[i, m_of_new_entry] = old_entry
        else:
            if m_of_new_entry < self.m:
                print("not updating to not destroy agree data")
            else:
                self.k_im[i, self.m] = k
                self.k_im[i, m_of_new_entry] = old_entry

    # -------------------------------------------------------------
    def init_i_min_m(self):
        """
        """
        self.i_min_m = np.zeros(self.N_evals, dtype=int)

    def load_i_min_m(self):
        """
        """
        self.i_min_m = np.load(self.data_path + "/i_min_m.npy")

    def save_i_min_m(self):
        """
        """
        np.save(self.data_path + "/i_min_m.npy", self.i_min_m)

    def update_i_min_m(self):
        """
        """
        self.i_min_m[self.m:] = self.i

    def revert_i_min_m(self):
        """
        """
        self.i_min_m[self.m:] = self.i_min_m[self.m-1]

    # ######################################################################
    # plotting
    # ######################################################################

    def plot_data(self):
        """
        """
        self.active_window=1
        self.parameter_info()

        self.load()
        self.load_eval_array()

        # prepare the level figures ---------------------------------------
        self.fig = plt.figure(1)

        # plotting the spectrum
        self.ax1 = self.fig.add_subplot(221, xlim=(-1.1, 1.1),
                                        ylim=(-1.1,1.1))
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
                dx = (event.xdata - self.evals.real)**2
                dy = (event.ydata - self.evals.imag)**2
                self.k = np.argmin(np.sqrt(dx + dy))

            elif self.active_window == 3:
                print('gammas', event.xdata, event.ydata)
                x = 1.0/2.0/np.pi/self.parameter_array
                self.i = np.argmin(abs(x-event.xdata))
                y = gamma(self.eval_array[self.i,:self.N_max])
                self.k = np.argmin(abs(np.log(y/event.ydata)))

            elif self.active_window == 4:
                print('phases', event.xdata, event.ydata)
                x = 1.0/2.0/np.pi/self.parameter_array
                self.i = np.argmin(abs(x-event.xdata))
                y = phases(self.eval_array[self.i,:self.N_max])
                self.k = np.argmin(abs(y-event.ydata))

            else:
                pass

        elif event.button == 2:
            self.update_k_im()

        else:
            pass

        self.re_plot()

    def _key_press(self, event):
        """
        """
        # changing the parameter index
        if event.key == 'r':
            self.i = (self.i + 1)%(self.N_params)
        elif event.key == 'e':
            self.i = (self.i - 1)%(self.N_params)

        # changing the level index (ordered by gamma)
        elif event.key == 'n':
            self.k = (self.k + 1)%(self.N_evals)
        elif event.key == 'p':
            self.k = (self.k - 1)%(self.N_evals)

        # changing the diabatic line
        elif event.key == 'x':
            self.m = (self.m - 1)%(self.N_evals)
        elif event.key == 'v':
            self.m = (self.m + 1)%(self.N_evals)
        elif event.key == 'm':
            print('enter index of diabatic line')
            tmp = raw_input()
            self.m = int(tmp)

        elif event.key == '.':
            self.N_max += 10
        elif event.key == ',':
            self.N_max -= 10

        # connecting phase upwards
        elif event.key == 'c':
            self._connect_evals_up()
        elif event.key == 'z':
            self.update_i_min_m()
        elif event.key == 'b':
            self.revert_i_min_m()

        # active windows
        elif event.key == '1':
            self.active_window=1
        elif event.key == '2':
            self.active_window=2
        elif event.key == '3':
            self.active_window=3
        elif event.key == '4':
            self.active_window=4

        elif event.key == 'k':
            self._save_diabatic_line()

        elif event.key == 'y':
            self.update_k_im(force=True)

        elif event.key == '[':
            self.clean_up_k_im()

        elif event.key == 'u':
            self.save_k_im()
            self.save_i_min_m()
        elif event.key == 'o':
            self.load_k_im()
            self.load_i_min_m()
        elif event.key == 'i':
            self.init_k_im()
            self.init_i_min_m()

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
        print("current parameter index i = ", self.i, "/", self.N_evals)
        print("current level index k = ", self.k)
        print("current diabatic line index m = ", self.m)

    # #####################################################################
    #
    # #####################################################################
    def _plot_evals(self):
        """
        """
        # add a unit circle for orientation
        ang = np.linspace(0.0, 2.0 * np.pi, 100)
        x_circ = np.cos(ang)
        y_circ = np.sin(ang)
        self.ax1.plot(x_circ, y_circ, 'k-', label=None)

        # plot all stored evals
        for m in np.arange(self.N_evals):
            k = self.k_im[self.i, m]
            self.ax1.plot([self.evals.real[k]], [self.evals.imag[k]], '*',
                          color = self.col[m%len(self.col)], label=None)
        # selected state
        z = self.evals[self.k]
        self.ax1.plot([z.real], [z.imag], 'rd', markersize=10,
                      label=None)
        # diabatic state
        z = self.evals[self.k_im[self.i, self.m]]
        self.ax1.plot([z.real], [z.imag], 'ko',
                      label=None)

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
        self.ax2.plot(self.x_plot, abs(self.evecs[:, self.k]), 'k-')

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
        # current data point ----------------------------
        # k point
        if self.k < self.N_max:
            self.ax3.plot([1.0/2.0/np.pi/self.parameter_array[self.i]],
                          [gamma(self.eval_array[self.i, self.k])],
                          'rd', markersize=10)

        i_array = np.arange(self.N_params)
        for m in np.arange(self.N_max)[::-1]:
            i_ = i_array[self.i_min_m[m]:]
            self.ax3.plot(1.0/2.0/np.pi/self.parameter_array[i_],
                          gamma(self.eval_array[i_, self.k_im[i_, m]]),
                          '+', color = self.col[m%len(self.col)])
        for m in np.arange(self.m):
            i_ = i_array[self.i_min_m[m]:]
            self.ax3.plot(1.0/2.0/np.pi/self.parameter_array[i_],
                          gamma(self.eval_array[i_, self.k_im[i_, m]]),
                          '+', color = (0.7, 0.7, 0.7))

        # highlight diabatic line -----------------------
        if (self.k_im[:,self.m]).all() < self.N_max:
            i_ = i_array[self.i_min_m[self.m]:]
            # current spaghetti
            self.ax3.plot(1.0/2.0/np.pi/self.parameter_array[i_],
                          gamma(self.eval_array[i_, self.k_im[i_, self.m]]),
                          'ko')

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
        # current data point ------------------------
        if self.k < self.N_max:
            self.ax4.plot([1.0/2.0/np.pi/self.parameter_array[self.i]],
                          [phases(self.eval_array[self.i, self.k])],
                          'rd', markersize=10)

        i_array = np.arange(self.N_params)
        for m in np.arange(self.N_max)[::-1]:
            i_ = i_array[self.i_min_m[m]:]
            self.ax4.plot(1.0/2.0/np.pi/self.parameter_array[i_],
                          phases(self.eval_array[i_, self.k_im[i_, m]]),
                          '+', color = self.col[m%len(self.col)])

        # gray out the lower diabatic lines -------------
        i_array = np.arange(self.N_params)
        for m in range(self.m):
            i_ = i_array[self.i_min_m[m]:]
            self.ax4.plot(1.0/2.0/np.pi/self.parameter_array[i_],
                          phases(self.eval_array[i_, self.k_im[i_, m]]),
                          '+', color = (0.7, 0.7, 0.7))


        # highlight diabatic line -----------------------
        if (self.k_im[:,self.m]).all() < self.N_max:
            i_ = i_array[self.i_min_m[self.m]:]
            # current spaghetti
            self.ax4.plot(1.0/2.0/np.pi/self.parameter_array[i_],
                          phases(self.eval_array[i_, self.k_im[i_, self.m]]),
                          'ko')

    def _update_phases(self):
        """
        """
        self.ax4.lines = []
        self._plot_phases()

    # #####################################################################
    # ordering commands
    # #####################################################################
    def _connect_evals_up(self):
        """
        """
        self.k_im[self.i,self.m] = self.k
        for i in range(self.i, self.N_params-1):
            if i == 0:
                u0 = self.eval_array[i, self.k_im[i,self.m]]
            else:
                _1 = self.eval_array[i, self.k_im[i,self.m]]
                _2 = self.eval_array[i-1, self.k_im[i-1,self.m]]
                u0 = _1 + (_1 - _2)
            u_ = self.eval_array[i+1, :]
            du = abs(u_ - u0)
            self.update_k_im(i=i+1, k=np.argmin(du))

    # #####################################################################
    # saving the post processed data
    # #####################################################################
    def _save_diabatic_line(self):
        """
        """
        print('-------------------------------------------')
        print('saving the currently selected diabatic line')
        i_array = np.arange(self.N_params)
        i_ = i_array[self.i_min_m[self.m]:]
        hinv = 1.0/2.0/np.pi/self.parameter_array[i_]
        evals = self.eval_array[i_, self.k_im[i_, self.m]]
        np.save(self.data_path + "/pp/" + "evals_m_" + str(self.m),
                [hinv, evals])