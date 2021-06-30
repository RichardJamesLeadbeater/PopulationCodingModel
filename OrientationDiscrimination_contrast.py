#
# PopulationResponse Coding Model
#
# Important:
# Feature space extended from (-90:+90) to (-180:+180) for a moving prefs window which provides circular tuning;
# ...MyStimuli are only presented within (-90:+90) range
# Preferences for each trial only contain values within a 180 degree range;
# ...these values change depending upon the stimulus value;
# ...e.g. if stimulus value == -45, the preferences used will be (-135:135)
# ...     if stimulus value == 85, the preferences used will be (-5:175)
import datetime
import random

import numpy as np
from scipy.linalg import hankel
import pandas as pd
import time
import dill as pickle
import multiprocessing as mp


"""Population Coding Model that performs an orientation discrimination two-alternative forced-choice task"""
# notes:
#       - circular tuning enabled by having a moving window of prefs centred on stim_val
#       - currently one staircase doing one run at a time (WTA: 3m47s, PV: 3m57s, ML: 5m54s)
#               - limited by moving prefs window
#       - with multiprocessing ((WTA + PV + ML) * 2runs: 2m12s   (data is easiest to work with)
#                                               * 6runs: 6m01s
#       - with multiprocessing_1cond: allconds * 2runs: 1m57s
#                                              * 6runs: 5m54s


def get_boundaries(mid_values, bound_range):
    boundaries = [[]] * len(mid_values)
    for idx, val in enumerate(mid_values):
        boundaries[idx] = [val - (bound_range / 2), val + (bound_range / 2)]
    return boundaries


def find_nearest(array, value):  # nearest value and index in array for each value
    if isinstance(value, int) or isinstance(value, float):
        nearest_idx = (np.abs(array - value)).argmin()
        nearest_value = array[nearest_idx]
    else:
        array = np.asarray(array)
        value = np.asarray(value)
        nearest_idx = []
        nearest_value = []
        for val in value:
            nearest_idx.append((np.abs(array - val)).argmin())
            nearest_value.append(array[nearest_idx[-1]])
    return nearest_value, nearest_idx


def sort_by_standard(standard, *args):
    # sorts arrays with inds matching ascending order of standard
    standard = np.asarray(standard)
    inds = standard.argsort()  # gets indices of sorted standard
    sorted_arrays = [standard[inds[::1]]]  # create list with first element as sorted standard
    for i_arg in args:
        i_arg = np.asarray(i_arg)
        sorted_arrays.append(i_arg[inds[::1]])  # append arrays sorted to inds of standard
    # sorted in ascending numerical order [::1]
    return sorted_arrays


def rolling_window(vector, window):
    # create rolling window across vector
    m = len(vector) - len(window)
    windows_idxs = hankel(np.arange(0, m + 1), np.arange(m, len(vector)))
    windows_vals = vector[windows_idxs]
    return windows_vals, windows_idxs


def generate_gaussian(x, mu, sigma):
    # gaussian tunings for multiple prefs (mu) & single halfwidth (sigma)
    mu = np.asarray([mu]).transpose()  # formatting so we can broadcast together with x
    p1 = -.5 * (((x - mu) / sigma) ** 2)
    p2 = sigma * np.sqrt(2 * np.pi)
    tunings = np.exp(p1) / p2
    return tunings


class Tiling:
    """tiling of feature space"""

    def __init__(self, min_tile, max_tile, stepsize):
        self.min_val = min_tile
        self.max_val = max_tile
        self.range = abs(min_tile - max_tile)
        self.stepsize = stepsize
        self.tiling = np.arange(min_tile, max_tile, stepsize)


class Stimuli:
    """values of MyStimuli across feature space"""

    def __init__(self, ori=None, con=None, tiling=None):
        self.tiling = tiling
        self.ori = ori
        self.con = con
        _, self.ori_idx = find_nearest(tiling, ori)

    def get_stim_ori_idx(self, stim_ori=None):
        if stim_ori is None:
            stim_ori = self.ori
        self.ori, self.ori_idx = find_nearest(self.tiling, stim_ori)
        return self.ori_idx


class NeuralPopulation:
    """params and sampling for neural population with independent properties"""

    def __init__(self, name=None, boundaries=None, sampling_freq=None, r_max=60, semi_sat=10, exponent=4, sigma=10,
                 spont=0.05, info=None):
        if info is None:
            self.info = {}
        elif isinstance(info, dict):  # if dict input to give info
            self.name = info['name']
            self.boundaries = info['boundaries']
            self.sampling_freq = info['sampling_freq']
            self.sigma = info['sigma']
            self.r_max = info['r_max']
            self.spont = info['spont']
            self.semi_sat = info['semi_sat']
            self.exponent = info['exponent']
        else:  # if values input on call
            self.name = name
            self.boundaries = boundaries  # [min, max]
            self.sampling_freq = sampling_freq  # degrees per neuron pref
            self.sigma = sigma  # sigma of gaussian tuning
            self.r_max = r_max
            self.spont = spont
            self.semi_sat = semi_sat
            self.exponent = exponent
        # initialise
        self.tunings = None
        self.prefs = None
        self.prefs_idx = None

    def get_info(self):
        self.info = dict(name=self.name, tunings=self.tunings, prefs=self.prefs, prefs_idx=self.prefs_idx,
                         boundaries=self.boundaries, sampling_freq=self.sampling_freq, sigma=self.sigma,
                         r_max=self.r_max, spont=self.spont, semi_sat=self.semi_sat, exponent=self.exponent)
        return self.info

    def generate_prefs(self, tiling):
        #  in: self.boundaries, self.sampling_freq, min_val, max_val, tiling
        # fun: adjust sampling rate to nearest possible
        #      generate prefs and round to nearest value in tiling
        #      get idx of each pref in tiling
        # out: prefs, prefs_idx, sampling_freq for each neuron from bounds and sampling freq
        # if 1D array then put boundaries into list
        if np.asarray(self.boundaries).ndim == 1:  # if one set of boundaries put in list for compatibility
            self.boundaries = [self.boundaries]
        n_bounds = len(self.boundaries)
        prefs = [[]] * n_bounds
        sampling_freq = [[]] * n_bounds

        for idx, ii in enumerate(self.boundaries):
            n_neurons = abs(ii[0] - ii[1]) / self.sampling_freq
            n_neurons = int(np.round(n_neurons, 0))  # to nearest whole n_neurons for nearest possible sampling freq
            prefs[idx] = np.linspace(ii[0], ii[1], n_neurons)
            sampling_freq[idx] = abs(ii[0] - ii[1]) / (len(prefs[idx]) - 1)
        self.sampling_freq = sampling_freq  # assign to attribute
        # prefs into 1D numpy array
        if n_bounds == 1:
            prefs = np.asarray(prefs[0])
        if n_bounds > 1:
            prefs = np.concatenate([i for i in prefs])
        # only prefs within range of tiling (e.g. -180 to 180)
        prefs = prefs[(prefs >= tiling.min()) & (prefs <= tiling.max())]
        # round each pref to nearest value in tiling and get its index
        self.prefs, self.prefs_idx = find_nearest(tiling, prefs)

        return self.prefs, self.prefs_idx, self.sampling_freq

    def generate_tunings(self, tiling):
        """pass in prefs and generate tuning"""
        repeated_tiling = tiling * np.ones([len(self.prefs), 1])
        tunings = generate_gaussian(repeated_tiling, self.prefs, self.sigma)
        tunings = tunings / tunings.max()  # normalise to 1
        tunings = tunings * self.r_max + (self.spont * self.r_max)  # normalise to rmax then + spont firing
        self.tunings = tunings
        return self.tunings


class PopulationCode:
    def __init__(self, tiling, neural_populations=None):
        self.tiling = tiling
        if neural_populations is None:
            neural_populations = []
        self.neural_populations = neural_populations
        # the following will be appended in methods of class
        self.all_tunings = []
        self.all_prefs = []
        self.all_prefs_idx = []
        # the following will be calculated in methods of class
        self.all_prefs_windows = None
        self.all_prefs_windows_idx = None
        self.trial_stim_ori = None
        self.trial_stim_ori_idx = None
        self.trial_stim_con = None
        self.trial_prefs = None
        self.trial_prefs_idx = None
        self.trial_tunings = None
        self.resp_tuned = None
        self.resp_contrast = None
        self.resp_noisy = None
        self.decoded = None

    def get_bounds_adjusters(self, value_list=None, operation='mean'):
        # used to get average sampling frequency to allow boundary adjustment
        """for each value, get average with each neighbouring value"""
        if value_list is None:
            value_list = [i_pop.sampling_freq for i_pop in self.neural_populations]
        avgs = [[]] * len(value_list)
        for idx, val in enumerate(value_list):
            if idx == 0:  # if first val then circle to end as cannot index 1-1
                x = len(value_list) - 1
                y = idx + 1
            elif idx == len(value_list) - 1:  # if last val then circle to start as cannot index end+1
                x = idx - 1
                y = 0
            else:  # middle indices pull vals to left and right of val
                x = idx - 1
                y = idx + 1
            if operation == 'mean':
                avgs[idx] = [np.mean([value_list[idx], value_list[x]]), np.mean([value_list[idx], value_list[y]])]
        return avgs

    def adjust_boundaries(self, update=False):
        # todo depends on order of list, need to ensure this is correct (use i_pop.boundaries[0] to order?)
        # adjust boundaries to smoothly switch between different sampling rates
        boundaries = [np.asarray(i_pop.boundaries) for i_pop in self.neural_populations]
        # get average sampling frequency of adjacent orientations
        adjuster_values = np.asarray(self.get_bounds_adjusters())
        for idx, i_bounds in enumerate(boundaries):  # loop through each ori bounds
            i_adjusted_bounds = i_bounds + ((adjuster_values[idx] * [1, -1]) / 2)
            self.neural_populations[idx].boundaries = i_adjusted_bounds
            if update:  # update tunings for new boundaries
                self.neural_populations[idx].generate_prefs(self.tiling)
                self.neural_populations[idx].generate_tunings(self.tiling)
            else:
                pass

    def stack_populations(self, sort_by_prefs=True):
        all_prefs = []
        all_prefs_idx = []
        all_tunings = []
        for idx, i_pop in enumerate(self.neural_populations):
            all_prefs = np.hstack([all_prefs, i_pop.prefs])
            all_prefs_idx = np.hstack([all_prefs_idx, i_pop.prefs_idx])
            all_tunings.append(i_pop.tunings)
        all_tunings = np.vstack([i for i in all_tunings])  # concat all appended tunings
        # sort prefs_idx and tunings by indices of prefs in ascending order
        if sort_by_prefs:
            (all_prefs, all_prefs_idx, all_tunings) = sort_by_standard(all_prefs, all_prefs_idx, all_tunings)
        # assign
        self.all_prefs = all_prefs
        self.all_prefs_idx = all_prefs_idx
        self.all_tunings = all_tunings

    def rolling_window(self, vector=None, window=None):
        # create rolling prefs window across any range of space (circular tuning)
        if vector is None:
            vector = self.all_prefs
        if window is None:
            # >= -90 to < 90 consists of all possible orientations in standard size feature space
            window = self.all_prefs[(self.all_prefs >= -90) & (self.all_prefs < 90)]
        m = len(vector) - len(window)
        self.all_prefs_windows_idx = hankel(np.arange(0, m + 1), np.arange(m, len(vector)))
        self.all_prefs_windows = vector[self.all_prefs_windows_idx]
        return self.all_prefs_windows, self.all_prefs_windows_idx

    def gen_tunings(self, stack=True, window=True):
        # generate prefs and tunings for each ori_pop
        for idx, i_pop in enumerate(self.neural_populations):
            self.neural_populations[idx].generate_prefs(self.tiling)
            self.neural_populations[idx].generate_tunings(self.tiling)
        if stack:
            self.stack_populations()
        if window:
            self.rolling_window()

    def contrast_response(self, resp_tuned=None, stim_contrast=100, n=3.4, c50=24):
        # input all vars externally except resp_tuned
        # Albrecht & Hamilton (1992): n=3.4 c50=24
        # n is exponent for steepness of curve
        # c50 semi_saturation constant is contrast for half-max response
        update = False
        if resp_tuned is None:
            resp_tuned = self.resp_tuned
            update = True
        resp_contrast = resp_tuned * (stim_contrast ** n / (stim_contrast ** n + c50 ** n))  # hyperbolic ratio
        if update:
            self.resp_contrast = resp_contrast
        return resp_contrast

    def get_contrast_adjusted_tunings(self, stim_con=None):
        if stim_con is None:
            stim_con = self.trial_stim_con
        adjusted_tunings = []
        all_prefs = []
        for i_pop in self.neural_populations:
            all_prefs = np.hstack([all_prefs, i_pop.prefs])  # use this to sort by
            # contrast mod on each pops tunings, utilising each pops exponent and semi_sat
            adjusted_tunings.append(self.contrast_response(i_pop.tunings, stim_contrast=stim_con,
                                                           n=i_pop.exponent, c50=i_pop.semi_sat))
        adjusted_tunings = np.vstack([i for i in adjusted_tunings])
        (all_prefs, adjusted_tunings) = sort_by_standard(all_prefs, adjusted_tunings)

        return adjusted_tunings

    def shift_prefs(self, stim_ori):
        all_windows = self.all_prefs_windows
        all_prefs = np.asarray(self.all_prefs)
        # get central value of every possible prefs window
        centre_vals = np.mean([all_windows.min(1), all_windows.max(1)], 0)
        # get and use row_idx (window) where centre is closest to stim_ori
        window_idx = abs(centre_vals - stim_ori).argmin()
        window_vals = all_windows[window_idx, :]
        # get range of idxs so we can index tunings array
        min_idx = abs(all_prefs - window_vals.min()).argmin()  # start of window
        max_idx = abs(all_prefs - window_vals.max()).argmin()  # end of window
        window_idxs = np.arange(min_idx, max_idx + 1)  # all idxs within window
        # assign / return
        self.trial_prefs = window_vals
        self.trial_prefs_idx = window_idxs
        return window_vals, window_idxs

    def gen_stim_response(self, stim_ori, stim_ori_idx=None, stim_con=100, decoder=None):
        # assign mutable stim vals
        self.trial_stim_ori = stim_ori
        if stim_ori_idx is None:
            _, stim_ori_idx = find_nearest(self.tiling, stim_ori)
        self.trial_stim_ori_idx = stim_ori_idx
        self.trial_stim_con = stim_con
        # adjust tunings dependent on CRF of each pop
        trial_tunings = self.get_contrast_adjusted_tunings()
        # save used window of prefs and tunings for later decoding
        trial_prefs, trial_prefs_idx = self.shift_prefs(stim_ori)
        # tunings within prefs window for this stim_ori
        self.trial_tunings = trial_tunings[trial_prefs_idx, :]  # save trial tunings for max likelihood
        # generate noiseless response from tunings
        self.resp_tuned = self.trial_tunings[:, stim_ori_idx]

        # error checking that stim val is in centre of prefs window - circular tuning maintained
        maxidx_ = (len(self.resp_tuned) - 1) / 2
        if self.resp_tuned.argmax() != np.ceil(maxidx_):
            if self.resp_tuned.argmax() != np.floor(maxidx_):
                print('circular tuning failed')
                print(f'stim_ori {stim_ori} not in centre of prefs window, in position {self.resp_tuned.argmax()}')
                if stim_ori < 90:
                    print('hi')
                debug = 1
        # generate noisy response using poisson noise
        self.resp_noisy = np.random.poisson(self.resp_tuned)

        if decoder is None:
            pass
        else:
            while all(decoder != i for i in ['WTA', 'PV', 'ML']):
                decoder = [input('Input any of the following:\n\tWTA\tPV\tML')]
                if not decoder:  # if no keys entered
                    decoder = ['WTA']
            self.decode_response(decoder)
        # assign
        self.trial_prefs = trial_prefs
        self.trial_prefs_idx = trial_prefs_idx
        # return
        return self.resp_noisy

    def wta(self, pop_resp=None, trial_prefs=None):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # trial_prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if trial_prefs is None:
            trial_prefs = self.trial_prefs
        trial_max = pop_resp.max()  # max response for each trial
        ismax = (trial_max == pop_resp).astype('int')  # bool 01 matrix of max for each trial
        trial_pref_idx = (ismax * np.random.random(size=ismax.shape)).argmax()  # idx of pref with max response
        wta_est = trial_prefs[trial_pref_idx]
        # which pref produced the strongest response each trial
        return wta_est

    def popvector(self, pop_resp=None, prefs=None):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # trial_prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if prefs is None:
            prefs = self.trial_prefs
        trial_resp = pop_resp
        cond_prefs = prefs * (np.pi / 180)  # convert to radians
        hori = np.sum((trial_resp * np.cos(cond_prefs)))
        vert = np.sum((trial_resp * np.sin(cond_prefs)))
        popvector_est = (np.arctan2(vert, hori) * (180 / np.pi))  # inverse tangent in degrees
        return popvector_est

    def maxlikelihood(self, pop_resp=None, tunings=None, tiling=None):
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if tunings is None:
            tunings = self.trial_tunings
        if tiling is None:
            tiling = self.tiling
        log_tunings = np.log10(tunings)
        log_likelihood = pop_resp.T @ log_tunings  # matrix multiplication; outputs: size=[trials, tiling]
        max_likelihood_idx = log_likelihood.argmax()  # gives idx in feature space of ML est
        maxlikelihood_est = (tiling[max_likelihood_idx])
        return maxlikelihood_est

    def decode_response(self, decoder='WTA'):
        decoded_val = None
        if decoder == 'WTA':
            decoded_val = self.wta()
        elif decoder == 'PV':
            decoded_val = self.popvector()
        elif decoder == 'ML':
            decoded_val = self.maxlikelihood()
        self.decoded = decoded_val
        return decoded_val


class StaircaseHandler:
    def __init__(self, start_level, step_sizes, n_up, n_down, n_reversals, revs_per_thresh, decoder_info=None,
                 max_level=None, current_level=None, extra_info=None):
        self.revs_per_thresh = revs_per_thresh
        self.n_down = n_down
        self.n_up = n_up
        self.n_reversals = n_reversals
        self.is_corr = []  # tracks n of corr answers in a row
        self.start_level = start_level
        if current_level is None:
            self.current_level = start_level
        if max_level is None:
            self.max_level = start_level
        self.step_sizes = step_sizes
        self.stepsizes_idx = 0
        self.current_stepsize = step_sizes[0]
        self.level_list = [current_level]
        self.current_direction = 'down'
        self.reversals = []
        self.continue_staircase = True
        self.threshold = []
        self.decoder_info = decoder_info
        self.run_info = {'decoder': [], 'ori': [], 'iv': [], 'threshold': []}  # appends during runs
        self.data = None
        self.extra_info = extra_info  # to be added externally when saving to pkl

    def stop(self):
        if not self.continue_staircase:
            self.run_info['decoder'].append(self.decoder_info)
            self.run_info['threshold'].append(self.threshold)
            # reset initial values to defaults
            self.is_corr = []  # tracks n of corr answers in a row
            self.current_level = self.start_level
            self.stepsizes_idx = 0
            self.current_stepsize = self.step_sizes[0]
            self.level_list = [self.current_level]
            self.current_direction = 'down'
            self.reversals = []

    def is_correct(self, correct_ans, given_ans):
        if correct_ans == given_ans:
            self.is_corr.append(1)  # correct answer
        else:
            self.is_corr.append(0)  # wrong answer
        self.update_staircase()

    def update_level(self, current_direction=None):
        if current_direction is None:
            current_direction = self.current_direction
        if current_direction == 'up':
            self.current_level *= 10 ** self.current_stepsize
            if self.current_level > self.max_level:
                self.current_level = self.max_level  # force back to max
        elif current_direction == 'down':
            self.current_level /= 10 ** self.current_stepsize
        if len(self.step_sizes) - 1 == self.stepsizes_idx:  # hold on last stepsize once reached
            pass
        else:
            self.stepsizes_idx += 1

    def calc_threshold(self):
        self.threshold = np.mean(self.reversals[-self.revs_per_thresh:])
        return self.threshold

    def update_staircase(self):
        self.current_stepsize = self.step_sizes[self.stepsizes_idx]
        # check if staircase needs to step down
        if len(self.is_corr) >= self.n_down:  # only check for step up when possible
            if all(i == 1 for i in self.is_corr[-self.n_down:]):
                if self.current_direction == 'up':  # if change in direction of staircase
                    self.reversals.append(self.current_level)
                self.current_direction = 'down'
                self.update_level()

        # check if staircase needs to step up
        if len(self.is_corr) >= self.n_up:  # only check for step up when possible
            if all(i == 0 for i in self.is_corr[-self.n_up:]):  # checks if incorrect answer
                if self.current_direction == 'down':  # if change in direction of staircase
                    self.reversals.append(self.current_level)
                self.current_direction = 'up'
                self.update_level()

        if len(self.reversals) == self.n_reversals:  # if predetermined n reversals reached, end exp
            self.continue_staircase = False
            self.calc_threshold()

        if self.continue_staircase:
            self.level_list.append(self.current_level)

    def info2dframe(self):
        self.data = pd.DataFrame(self.run_info)
        orders = dict(ori=self.data['ori'].unique(), iv=self.data['iv'].unique())
        custom_sort = {}
        for i_categ in orders:
            for idx, cond in enumerate(orders[i_categ]):
                custom_sort[cond] = idx
        self.data = self.data.sort_values(by=['ori', 'iv'], key=lambda x: x.map(custom_sort))


def perform_2afc_ori_contrast(staircase, popcode, decoder=None, oris=None, contrasts=None, nruns=None):
    if decoder is None:
        decoder = 'WTA'
    staircase.decoder_info = decoder
    if oris is None:
        oris = [0]
    if contrasts is None:
        contrasts = [100]
    if nruns is None:
        nruns = 1
    for i_run in range(nruns):
        for i_ori in oris:  # loop through each standard orientation
            for i_con in contrasts:  # loop through each value of iv
                staircase.run_info['ori'].append(i_ori)
                staircase.run_info['iv'].append(i_con)
                staircase.continue_staircase = True
                while staircase.continue_staircase:
                    level = staircase.current_level
                    standard = Stimuli(ori=i_ori, con=i_con, tiling=popcode.tiling)
                    if random.random() > 0.5:  # rand ori_diff direction of rotation (probs unnecess for model)
                        corr_ans = 'CW'
                        comparison = Stimuli(ori=(standard.ori + level),
                                             con=standard.con, tiling=popcode.tiling)
                    else:
                        corr_ans = 'CCW'
                        comparison = Stimuli(ori=(standard.ori - level),
                                             con=standard.con, tiling=popcode.tiling)
                    # generate response to stimulus ori & contrast
                    PopCode.gen_stim_response(standard.ori, standard.ori_idx, standard.con)
                    standard_decoded = PopCode.decode_response(decoder=staircase.decoder_info)
                    PopCode.gen_stim_response(comparison.ori, comparison.ori_idx, comparison.con)
                    comparison_decoded = PopCode.decode_response(decoder=staircase.decoder_info)

                    if comparison_decoded < standard_decoded:
                        this_ans = 'CCW'
                    else:
                        this_ans = 'CW'
                    staircase.is_correct(this_ans, corr_ans)  # checks if correct and updates staircase accordingly
                    if not staircase.continue_staircase:
                        staircase.stop()
    staircase.info2dframe()  # converts staircase information to dataframe
    return staircase


if __name__ == '__main__':

    # define feature space where orientations will be defined
    FeatureSpace = Tiling(min_tile=-270, max_tile=270, stepsize=0.05)

# todo find a way to get the full width at max/sqrt(2) for comparison with previous studies (from sigma)
    # initialise parameters of each ori pop
    ori_populations_info = dict(vertical={'sampling_freq': 1, 'sigma': 12,  # full-width half-max = 2.3548 * sigma
                                          'r_max': 60,
                                          'boundaries': get_boundaries([-90, 90], 45),
                                          'spont': 0.05, 'exponent': 3.4, 'semi_sat': 24, 'name': 'vertical'},

                                right_oblique={'sampling_freq': 2, 'sigma': 10, 'r_max': 60,
                                               'boundaries': get_boundaries([-45, 135], 45),
                                               'spont': 0.05, 'exponent': 3.4, 'semi_sat': 24, 'name': 'right_oblique'},

                                horizontal={'sampling_freq': 1, 'sigma': 12, 'r_max': 60,
                                            'boundaries': get_boundaries([-180, 0, 180], 45),
                                            'spont': 0.05, 'exponent': 3.4, 'semi_sat': 24, 'name': 'horizontal'},

                                left_oblique={'sampling_freq': 2, 'sigma': 12, 'r_max': 60,
                                              'boundaries': get_boundaries([-135, 45], 45),
                                              'spont': 0.05, 'exponent': 3.4, 'semi_sat': 24, 'name': 'left_oblique'})

    ori_populations = {}
    for key in ori_populations_info:
        ori_populations[key] = NeuralPopulation(info=ori_populations_info[key])

    PopCode = PopulationCode(tiling=FeatureSpace.tiling,
                             neural_populations=[ori_populations[key] for key in ori_populations])
    PopCode.adjust_boundaries()  # adjust boundaries based off average sampling rate of each neighbouring population
    # generate tunings & prefs for each pop
    PopCode.gen_tunings(stack=True, window=True)  # stack & calc rollingwindow of prefs

    tic = time.time()
    # use multiprocessing for different decoding methods over nruns
    # define variables with which to iterate over for all possible combinations
    n_runs = 6
    decoder = ['WTA', 'PV', 'ML']
    # define constants which will be iterated to match n_combinations
    Staircase = StaircaseHandler(start_level=20, step_sizes=[0.6, 0.4, 0.2, 0.1, 0.08],
                                 n_up=1, n_down=3, n_reversals=10, revs_per_thresh=6,
                                 extra_info=ori_populations_info)
    ori_std = [-45, 0, 45, 90]
    contrast = [2.5, 5, 10, 20, 40]
    # get all possible iterations / combinations of conditions for use in multiprocessing
    mp_iters = {'decoder': [], 'ori_std': [], 'contrast': [], 'popcode': [], 'staircase': []}
    for _ in range(n_runs):
        for i_decoder in decoder:
            mp_iters['decoder'].append(i_decoder)
            mp_iters['ori_std'].append(ori_std)
            mp_iters['contrast'].append(contrast)
            mp_iters['popcode'].append(PopCode)
            mp_iters['staircase'].append(Staircase)

    # make sure n_iters is equal for all params to be used in function
    # each decoder across all cond combos for n_runs
    with mp.Pool() as pool:
        mp_data = pool.starmap(perform_2afc_ori_contrast, zip(mp_iters['staircase'], mp_iters['popcode'],
                                                              mp_iters['decoder'], mp_iters['ori_std'],
                                                              mp_iters['contrast']))
        pool.close()
        pool.join()

    print(f"Multiprocessing of 3 staircases for {n_runs} runs each took:\n\t{(time.time() - tic):.2f} secs")

    all_data = pd.concat([i.data for i in mp_data])
    all_data = all_data.sort_values(by=['decoder', 'ori', 'iv'])

    output = dict(all_data=all_data, information=ori_populations_info)
    with open(f"find_a_better_way_to_name_files.pkl", "wb") as file:
        pickle.dump(output, file)

    with open(f"find_a_better_way_to_name_files.pkl", "rb") as file:
        test = pickle.load(file)
# todo pickle all data and appropriate info
    #     with open(f"{i_Staircase.decoder_info}_{tstamp}.pkl", "wb") as file:
    #         pickle.dump(i_Staircase, file)
    #         i_Staircase.data.to_csv(f"{i_Staircase.decoder_info}_{tstamp}_dframe.csv")
    #     # ensure file has saved and can be opened
    #     with open(f"{i_Staircase.decoder_info}_{tstamp}.pkl", "rb") as file:
    #         test = pickle.load(file)
