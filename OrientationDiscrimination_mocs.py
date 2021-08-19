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
import dill
import multiprocessing as mp
import os
from scipy.optimize import curve_fit
import pylab


"""Population Coding Model that performs an orientation discrimination two-alternative forced-choice task"""
# one interesting thing the psychometric functions show is that
# there is no point at which an ori_diff increase makes no difference
# with human observers an ogive is observed but with the model it's monotonic and gradual

# notes:
#       - METHOD OF CONSTANT STIMULI
#         > loop through each ori_std
#         > tuned response for standard and comparison for each orientation * nReps * nRuns
#         > proportion correct (always CW in model) for each run (blocks of nReps)
#         > use PCN curve fitting routine to pull threshold across all runs

# standard_oris = [-45, 0, 45, 90]
# ori_offset = np.linspace(0, 22.5, n_levels)  # n_levels can be huge number to ensure sensitive range for psych func

# # perform analysis on each ori_std at a time
# for i_std in standard_oris:
#     comparison_oris = [i + std for i in ori_offset]

#     # get noisy_resp and decoded for each i_std and i_comp for nReps
#     for j_comp in comparison_oris:
#         std_resp = gen_response(i_std, nreps=100)
#         std_decode = decode_resp(std_resp)
#         comp_resp = gen_response(j_comp, nreps=100)
#         comp_decode = decode_resp(comp_resp)

#         # calculate proportion CW by using comp > std
#         isCW = np.asarray(comp_decode) > np.asarray(std_decode)
#         proportion_CW = np.sum(isCW) / len(isCW)


def get_boundaries(base_ori, bound_range):
    base_ori = [base_ori - 180, base_ori, base_ori + 180]  # allows for moving prefs window across extended space
    boundaries = [[]] * len(base_ori)
    for idx, val in enumerate(base_ori):
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


class Params:
    def __init__(self, sampling_density=None, sampling_range=None, tuning_fwhm=None,
                 r_max=None, spont=None, exponent=None, semi_sat=None):
        # set model parameters to be varied
        self.sampling_density = sampling_density  # neurons per degree
        self.sampling_range = sampling_range
        self.tuning_fwhm = tuning_fwhm  # HWHM = s * sqrt(2ln2) = s * 2.3548
        # set constant model parameters`
        self.r_max = r_max
        self.spont = spont
        self.exponent = exponent
        self.semi_sat = semi_sat


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
            self.sigma = info['tuning_fwhm'] / 2.3548  # converts fwhm to standard deviation of gaussian
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
        tunings = tunings * self.r_max  # normalise to rmax
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

    def get_adjusted_tunings(self, contrast_adjustment=True, spontaneous_firing=True):
        stim_con = self.trial_stim_con
        adjusted_tunings = []
        all_prefs = []
        for i_pop in self.neural_populations:
            all_prefs = np.hstack([all_prefs, i_pop.prefs])  # use this to sort by
            i_tunings = i_pop.tunings  # initialise one ori_population tunings
            # contrast mod on each pops tunings, utilising each pops exponent and semi_sat
            if contrast_adjustment:
                i_tunings = self.contrast_response(i_pop.tunings, stim_contrast=stim_con,
                                                   n=i_pop.exponent, c50=i_pop.semi_sat)
            if spontaneous_firing:  # apply spontaneous firing rate after contrast adjustment
                i_tunings = i_tunings + (i_pop.r_max * i_pop.spont)
            else:
                pass
            adjusted_tunings.append(i_tunings)
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

    def gen_stim_response(self, stim_ori, stim_con=100, trial_decoder=None, n_reps=1, spontaneous_firing=True):
        # assign mutable stim vals
        self.trial_stim_ori = stim_ori
        _, stim_ori_idx = find_nearest(self.tiling, stim_ori)
        self.trial_stim_ori_idx = stim_ori_idx
        self.trial_stim_con = stim_con
        # adjust tunings dependent on CRF of each pop
        trial_tunings = self.get_adjusted_tunings(spontaneous_firing=spontaneous_firing)  # applies spont after contrast
        # save used window of prefs and tunings for later decoding
        trial_prefs, trial_prefs_idx = self.shift_prefs(stim_ori)
        # tunings within prefs window for this stim_ori
        self.trial_tunings = trial_tunings[trial_prefs_idx, :]  # save trial tunings for max likelihood

        # generate noiseless response from tunings
        # old # self.resp_tuned = self.trial_tunings[:, stim_ori_idx]  # old #
        resp_tuned = self.trial_tunings[:, stim_ori_idx]
        self.resp_tuned = np.transpose(resp_tuned * np.ones([n_reps, 1]))  # ***
        # generate noisy response using poisson noise
        self.resp_noisy = np.random.poisson(self.resp_tuned)

        # error checking that stim val results in a peak response - circular tuning maintained # ***
        maxidx_ = (len(resp_tuned) - 1) / 2
        maxidx_ = [int(np.floor(maxidx_)), int(np.ceil(maxidx_))]
        if all(i != resp_tuned.argmax() for i in maxidx_):
            if any(resp_tuned[i - 1] <= resp_tuned[i] <= resp_tuned[i + 1] for i in maxidx_):
                pass
            else:
                if show_errors:
                    print(f"\nCircular tuning may have failed:"
                          f"\n\t- Stim_ori {stim_ori} causes max tuned response at pref_idx {resp_tuned.argmax()}"
                          f" rather than centre_idx {int(np.round(len(trial_prefs) / 2))}"
                          f"\n\t- Stim does not result in a tuned response peak at central orientation"
                          f"\n\t  ...this may be a result of shared values due to integer values with Poisson noise")

        if trial_decoder is None:
            pass
        else:
            while all(trial_decoder != i for i in ['WTA', 'PV', 'ML']):
                trial_decoder = [input('Input any of the following:\n\tWTA\tPV\tML')]
                if not trial_decoder:  # if no keys entered
                    trial_decoder = ['WTA']
            self.decode_response(trial_decoder)
        # assign
        self.trial_prefs = trial_prefs
        self.trial_prefs_idx = trial_prefs_idx
        # return
        return self.resp_noisy

    def decode_response(self, response_decoder='WTA',
                        stim_ori=None, stim_con=None, n_reps=None, spontaneous_firing=True):
        if all(i or i == 0 for i in [stim_ori, stim_con, n_reps]):
            self.gen_stim_response(stim_ori, stim_con, response_decoder, n_reps, spontaneous_firing=spontaneous_firing)
        decoded_val = None
        if response_decoder == 'WTA':
            decoded_val = self.wta()
        elif response_decoder == 'PV':
            decoded_val = self.popvector()
        elif response_decoder == 'ML':
            decoded_val = self.maxlikelihood()
        self.decoded = decoded_val
        return decoded_val

    def wta(self, pop_resp=None, trial_prefs=None):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # trial_prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if trial_prefs is None:
            trial_prefs = self.trial_prefs
        trial_max = pop_resp.max(0)  # max response for each trial  *** .max()
        ismax = (trial_max == pop_resp).astype('int')  # bool 01 matrix of max for each trial
        trial_pref_idx = (ismax * np.random.random(size=ismax.shape)).argmax(0)  # idx of pref with max response  *** .argmax()
        wta_est = trial_prefs[trial_pref_idx]
        # which pref produced the strongest response each trial
        return wta_est

    def popvector(self, pop_resp=None, pop_prefs=None):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # trial_prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if pop_prefs is None:
            pop_prefs = self.trial_prefs
        cond_prefs = pop_prefs * (np.pi / 180)  # convert to radians
        # hori = np.sum((pop_resp * np.cos(cond_prefs)))  *** see below
        hori = np.sum((pop_resp.T * np.cos(cond_prefs)).T, 0)
        # vert = np.sum((pop_resp * np.sin(cond_prefs)))  *** see below
        vert = np.sum((pop_resp.T * np.sin(cond_prefs)).T, 0)
        popvector_est = (np.arctan2(vert, hori) * (180 / np.pi))  # inverse tangent in degrees
        return popvector_est

    def maxlikelihood(self, pop_resp=None, pop_tunings=None, tiling=None):
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if pop_tunings is None:
            pop_tunings = self.trial_tunings
        if tiling is None:
            tiling = self.tiling
        log_likelihood = pop_resp.T @ np.log10(pop_tunings)  # matrix multiplication; outputs: size=[trials, tiling]
        max_likelihood_idx = log_likelihood.argmax(1)  # gives idx in feature space of ML est  *** .argmax()
        maxlikelihood_est = tiling[max_likelihood_idx]
        return maxlikelihood_est


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
        self.run_info = {'decoder': [], 'ori_std': [], 'contrast': [], 'threshold': []}  # appends during runs
        self.data = None
        self.extra_info = extra_info  # to be added externally when saving to pkl

    def stop(self):
        if not self.continue_staircase:
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
        orders = dict(ori=self.data['ori_std'].unique(), iv=self.data['contrast'].unique())
        custom_sort = {}
        for i_categ in orders:
            for idx, cond in enumerate(orders[i_categ]):
                custom_sort[cond] = idx
        self.data = self.data.sort_values(by=['ori_std', 'contrast'], key=lambda x: x.map(custom_sort))


def logistic_function(x, x0, k):
    # Add 0.5 to the intercept and multiply the sigmoid by 0.5 since it now only spans a y-range half of before
    y = 0.5 + 0.5 / (1 + np.exp(-k * (x - x0)))
    return y


def fit_logistic(x_vals, y_vals):
    """ creates vals of logistic curve (params determined from psychometric data)"""

    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)
    # determine params of logistic curve using least squares method
    popt, _ = curve_fit(logistic_function, x_vals, y_vals, method='trf',
                        maxfev=4000,
                        bounds=([0, 0], [x_vals.max(), 10])  # gradient and middle of slope positive
                        )
    threshold = popt[0]

    # if threshold > (x_vals.max() - 0.5):
    #     print(f"{surname}\t{exp_name}\t{target_ori}\t{threshold:.2f}")
    #     global test
    #     test = dict(id=surname, exp=exp_name, ori=target_ori)

    slope = popt[1]
    # output of logistic function given x-vals and optimised params
    x = np.linspace(x_vals.min(), x_vals.max(), len(x_vals) * 100)
    y = logistic_function(x, *popt)

    # namedtuple does not work with multiprocessing
    return {'x': x, 'y': y, 'threshold': threshold, 'slope': slope}


def plot_psychometric(x_axis, y_axis, x_fit_axis, y_fit_axis, threshold_value, title='', close_all=True):
    # todo plot error bars
    threshold_idx = np.argmin(abs(x_fit_axis - threshold_value))
    if close_all:
        pylab.close('all')
    else:
        pass
    pylab.figure()
    pylab.plot(x_axis, y_axis, 'o', label='data')
    pylab.plot(x_fit_axis, y_fit_axis, label='fit')
    pylab.ylim(0 - .01, 1.01)
    pylab.xlim(x_axis.min() - 0.01, x_axis.max() + 0.01)
    pylab.plot([threshold_value, threshold_value], [0, y_fit_axis[threshold_idx]], 'k--', alpha=0.3)
    pylab.plot([0, threshold_value], [y_fit_axis[threshold_idx], y_fit_axis[threshold_idx]], 'k--', alpha=0.3)
    pylab.xlabel('Orientation Difference (deg)')
    pylab.ylabel('Proportion Correct')
    pylab.title(title)
    pylab.legend(loc='best')


def perform_oridis_2afc_mocs(mocs_popcode, mocs_decoder, mocs_ori_std, mocs_contrast, mocs_ori_offsets, mocs_n_reps):
    mocs_raw_data = dict(ori_offset=[], proportion_correct=[])
    for i_ori_offset in mocs_ori_offsets:
        i_comp_ori = mocs_ori_std + i_ori_offset
        std_decoded = mocs_popcode.decode_response(mocs_decoder, mocs_ori_std, mocs_contrast, mocs_n_reps)
        comp_decoded = mocs_popcode.decode_response(mocs_decoder, i_comp_ori, mocs_contrast, mocs_n_reps)
        isCW = comp_decoded > std_decoded  # oridis for each trial
        proportion_correct = (1 / len(isCW)) * np.sum(isCW)
        # append vals into data_dict
        mocs_raw_data['ori_offset'].append(i_ori_offset)
        mocs_raw_data['proportion_correct'].append(proportion_correct)

    if any(i >= 0.75 for i in mocs_raw_data['proportion_correct']):  # only get threshold if above chance performance
        mocs_curve_fit = fit_logistic(x_vals=mocs_ori_offsets, y_vals=mocs_raw_data['proportion_correct'])
        plot_psychometric(x_axis=ori_offset, y_axis=mocs_raw_data['proportion_correct'],
                          x_fit_axis=mocs_curve_fit['x'], y_fit_axis=mocs_curve_fit['y'],
                          threshold_value=mocs_curve_fit['threshold'],
                          title=f"{mocs_decoder} at {mocs_contrast}% contrast")
        # if mocs_curve_fit['slope'] > 7:
        #     print(f"slope: mocs_curve_fit['slope']")
        #     z = 1
        mocs_summary_data = {'decoder': mocs_decoder, 'ori_std': mocs_ori_std, 'contrast': mocs_contrast,
                             'threshold': mocs_curve_fit['threshold'], 'slope': mocs_curve_fit['slope']}
        return pd.DataFrame(mocs_summary_data, index=[''])
    else:
        return None


def perform_oridis_2afc_staircase(staircase, stair_popcode, stair_decoder='WTA', stair_oris=None,
                                  stair_contrasts=None):
    staircase.decoder_info = stair_decoder
    if stair_oris is None:
        stair_oris = [0]
    if stair_contrasts is None:
        stair_contrasts = [100]
    for i_ori in stair_oris:  # loop through each standard orientation
        for i_con in stair_contrasts:  # loop through each value of iv
            staircase.run_info['ori_std'].append(i_ori)
            staircase.run_info['contrast'].append(i_con)
            staircase.run_info['decoder'].append(stair_decoder)
            staircase.continue_staircase = True
            while staircase.continue_staircase:
                level = staircase.current_level
                standard = Stimuli(ori=i_ori, con=i_con, tiling=stair_popcode.tiling)
                if random.random() > 0.5:  # rand ori_diff direction of rotation (probs unnecess for model)
                    corr_ans = 'CW'
                    comparison = Stimuli(ori=(standard.ori + level),
                                         con=standard.con, tiling=stair_popcode.tiling)
                else:
                    corr_ans = 'CCW'
                    comparison = Stimuli(ori=(standard.ori - level),
                                         con=standard.con, tiling=stair_popcode.tiling)
                # generate response to stimulus ori & contrast
                std_decoded = stair_popcode.decode_response(stair_decoder, i_ori, i_con, n_reps=1,
                                                            spontaneous_firing=True)
                comp_decoded = stair_popcode.decode_response(stair_decoder, comparison.ori, i_con, n_reps=1,
                                                             spontaneous_firing=True)
                if comp_decoded < std_decoded:
                    this_ans = 'CCW'
                else:
                    this_ans = 'CW'
                staircase.is_correct(this_ans, corr_ans)  # checks if correct and updates staircase accordingly
                if not staircase.continue_staircase:
                    staircase.stop()
    staircase.info2dframe()  # converts staircase information to dataframe
    return staircase


if __name__ == '__main__':

    # all possible params
    density = [1, 1/1.5, 1/2, 1/2.5, 1/3, 1/3.5, 1/4, 1/4.5, 1/5, 1/5.5]
    fwhm = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
    bounds = [[45, 45], [30, 60], [20, 70], [10, 80], [5, 85]]
    rmax = [120, 60, 30]

    # loop_conds[0] = i_d   loop_conds[1] = i_t   loop_conds[2] = t_2
    d = [0, 6]  # sampling density of obliques
    t = [0, 8]  # tuning width fwhm
    b = [0]  # boundary
    f = [0, 1]  # firing rate  (Shen et al., 2014)

    counter = 0
    # get all possible combos of idcs for each paramater
    all_combos = [[d_, t_, b_, f_] for d_ in d for t_ in t for b_ in b for f_ in f]
    for i_combo in all_combos:
        i_d = i_combo[0]
        i_t = i_combo[1]
        i_b = i_combo[2]
        i_f = i_combo[3]

        cardinal = Params(sampling_density=density[0], sampling_range=bounds[i_b][0], tuning_fwhm=fwhm[i_t])
        oblique = Params(sampling_density=density[i_d], sampling_range=bounds[i_b][1], tuning_fwhm=fwhm[0])

        shared = Params(r_max=rmax[i_f - 1], spont=0.05, exponent=3.4, semi_sat=14)

        og_path = os.getcwd()
        data_path = os.path.join(og_path, 'data')
        if not os.path.exists(data_path):
            os.makedirs('data')

        # define feature space where orientations will be defined
        FeatureSpace = Tiling(min_tile=-270, max_tile=270, stepsize=0.05)

        # todo find a way to get the full width at max/sqrt(2) for comparison with previous studies (from sigma)
        # initialise parameters of each ori pop
        ori_populations_info = dict(vertical={'sampling_freq': 1 / cardinal.sampling_density,
                                              'tuning_fwhm': cardinal.tuning_fwhm,
                                              'boundaries': get_boundaries(0, cardinal.sampling_range),
                                              'r_max': shared.r_max,
                                              'spont': shared.spont,
                                              'exponent': shared.exponent,
                                              'semi_sat': shared.semi_sat,
                                              'name': 'vertical'},

                                    right_oblique={'sampling_freq': 1 / oblique.sampling_density,
                                                   'tuning_fwhm': oblique.tuning_fwhm,
                                                   'boundaries': get_boundaries(45, oblique.sampling_range),
                                                   'r_max': shared.r_max,
                                                   'spont': shared.spont,
                                                   'exponent': shared.exponent,
                                                   'semi_sat': shared.semi_sat,
                                                   'name': 'right_oblique'},

                                    horizontal={'sampling_freq': 1 / cardinal.sampling_density,
                                                'tuning_fwhm': cardinal.tuning_fwhm,
                                                'boundaries': get_boundaries(90, cardinal.sampling_range),
                                                'r_max': shared.r_max,
                                                'spont': shared.spont,
                                                'exponent': shared.exponent,
                                                'semi_sat': shared.semi_sat,
                                                'name': 'horizontal'},

                                    left_oblique={'sampling_freq': 1 / oblique.sampling_density,
                                                  'tuning_fwhm': oblique.tuning_fwhm,
                                                  'boundaries': get_boundaries(-45, oblique.sampling_range),
                                                  'r_max': shared.r_max,
                                                  'spont': shared.spont,
                                                  'exponent': shared.exponent,
                                                  'semi_sat': shared.semi_sat,
                                                  'name': 'left_oblique'}
                                    )

        ori_populations = {}
        for key in ori_populations_info:
            ori_populations[key] = NeuralPopulation(info=ori_populations_info[key])

        PopCode = PopulationCode(tiling=FeatureSpace.tiling,
                                 neural_populations=[ori_populations[key] for key in ori_populations])
        PopCode.adjust_boundaries()  # adjust boundaries based off average sampling rate of each neighbouring population
        # generate tunings & prefs for each pop
        PopCode.gen_tunings(stack=True, window=True)  # stack & calc rollingwindow of prefs

        # use multiprocessing for different decoding methods over nruns
        # define variables with which to iterate over for all possible combinations

        # at low contrasts the PV and ML decoders fail due to integer scale of poisson noise
        # (may have affected prev staircase results)
        use_staircase = True
        use_mocs = True
        use_mp = True
        use_single = False
        use_loops = False
        show_errors = False  # whether to print errors from inside functions

        ori_std = [0, 45]
        contrast = [2.5, 5, 10, 20, 40, 80]
        decoder = ['WTA', 'PV', 'ML']
        filename = f"B{i_b}_D{i_d}_T{i_t}_F{i_f}_{len(contrast)}cons"

        # # # METHOD OF CONSTANT STIMULI # # #
        if use_mocs:
            print('\n# # # RUNNING METHOD OF CONSTANT STIMULI # # #')

            n_level_reps = 100
            ori_offset = np.linspace(0, 90, 91)  # ensure sensitive psychometric function
            mocs_dframe = None

            if use_single:  # if just one run requested
                tic = time.time()
                mocs_dframe = perform_oridis_2afc_mocs(mocs_popcode=PopCode, mocs_decoder='WTA', mocs_ori_std=0,
                                                       mocs_contrast=2.5, mocs_ori_offsets=ori_offset,
                                                       mocs_n_reps=n_level_reps)
                print(f"\nSingle trial for {len(ori_offset)} levels:"
                      f"\n\t\t\t\t\t{time.time() - tic:.2f} seconds")

            if use_mp:  # if multiprocessing requested
                tic = time.time()
                mp_iters = {'popcode': [], 'decoder': [], 'ori_std': [], 'contrast': [], 'offsets': [],
                            'n_level_reps': []}
                n_iters = len(decoder) * len(ori_std) * len(contrast)  # all IVs
                for i_decoder in decoder:
                    for i_ori_std in ori_std:
                        for i_contrast in contrast:
                            # IVs
                            mp_iters['decoder'].append(i_decoder)
                            mp_iters['ori_std'].append(i_ori_std)
                            mp_iters['contrast'].append(i_contrast)
                            # constants
                            mp_iters['popcode'].append(PopCode)
                            mp_iters['offsets'].append(ori_offset)
                            mp_iters['n_level_reps'].append(n_level_reps)
                with mp.Pool() as pool:
                    mocs_data = pool.starmap(perform_oridis_2afc_mocs,
                                             zip(mp_iters['popcode'], mp_iters['decoder'],
                                                 mp_iters['ori_std'], mp_iters['contrast'],
                                                 mp_iters['offsets'], mp_iters['n_level_reps'])
                                             )
                    pool.close()
                    pool.join()

                print(f"\nMultiprocessing with {len(decoder) * len(contrast) * len(ori_std)} conds"
                      f" for {len(ori_offset)} levels:"
                      f"\n\t\t\t\t\t{time.time() - tic:.2f} seconds")
                mocs_dframe = pd.concat([i for i in mocs_data])
                mocs_dframe = mocs_dframe.sort_values(by=['decoder', 'ori_std', 'contrast'])

            if use_loops:  # if looping requested (no multiprocessing nor one-run requested)
                tic = time.time()
                mocs_data = []
                for i_std in ori_std:
                    for i_decoder in decoder:
                        for i_contrast in contrast:
                            mocs_data.append(perform_oridis_2afc_mocs(mocs_popcode=PopCode, mocs_decoder=i_decoder,
                                                                      mocs_ori_std=i_std, mocs_contrast=i_contrast,
                                                                      mocs_ori_offsets=ori_offset, mocs_n_reps=100)
                                             )
                mocs_dframe = pd.concat([i for i in mocs_data])
                print(f"\nLooping through {len(decoder) * len(contrast) * len(ori_std)} conds"
                      f" for {len(ori_offset)} levels:"
                      f"\n\t\t\t\t\t{time.time() - tic:.2f} seconds")

            # save out dframe for later analysis
            mocs_output = dict(all_data=mocs_dframe, information=ori_populations_info)
            with open(os.path.join(data_path, f"{filename}_mocs.pkl"), "wb") as file:
                dill.dump(mocs_output, file)

        # # # STAIRCASE # # #
        if use_staircase:
            print('\n# # # RUNNING STAIRCASE # # #')
            # define constants which will be iterated to match n_combos (stepsize has huge influence on stair time)
            n_runs = 5
            Staircase = StaircaseHandler(start_level=20, step_sizes=[0.6, 0.3, 0.25, 0.2, 0.19, 0.18, 0.17, 0.16],
                                         n_up=1, n_down=3, n_reversals=8, revs_per_thresh=5,
                                         extra_info=ori_populations_info)
            if use_single:
                tic = time.time()
                stair_dframe = perform_oridis_2afc_staircase(Staircase, PopCode, 'WTA', [0], [80])
                print(f"\nSingle run:"
                      f"\n\t\t\t\t\t{time.time() - tic:.2f} seconds")

            if use_mp:
                tic = time.time()
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
                    stair_data = pool.starmap(perform_oridis_2afc_staircase,
                                              zip(mp_iters['staircase'], mp_iters['popcode'],
                                                  mp_iters['decoder'], mp_iters['ori_std'],
                                                  mp_iters['contrast']))
                    pool.close()
                    pool.join()

                stair_dframe = pd.concat([i.data for i in stair_data])

                print(f"\nMultiprocessing with {len(decoder) * len(contrast) * len(ori_std)} conds"
                      f" for {n_runs} runs:"
                      f"\n\t\t\t\t\t{time.time() - tic:.2f} seconds")

                # save out dframe for later analysis
                stair_dframe = stair_dframe.sort_values(by=['decoder', 'ori_std', 'contrast'])
                stair_output = dict(all_data=stair_dframe, information=ori_populations_info)
                with open(os.path.join(data_path, f"{filename}_staircase.pkl"), "wb") as file:
                    dill.dump(stair_output, file)

            if use_loops:
                tic = time.time()
                stair_data = []
                for _ in range(n_runs):
                    for i_decoder in decoder:
                        stair_data.append(perform_oridis_2afc_staircase(Staircase, PopCode, i_decoder, ori_std, contrast)
                                          )
                print(f"\nLooping through {len(decoder) * len(contrast) * len(ori_std)} conds"
                      f" for {n_runs} runs:"
                      f"\n\t\t\t\t\t{time.time() - tic:.2f} seconds")

        z = 1

    z = 1
