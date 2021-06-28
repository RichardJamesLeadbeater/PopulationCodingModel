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
import random

import numpy as np
from scipy.linalg import hankel
import pandas as pd


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
    """attribute store"""

    def __init__(self):
        pass


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

    def __init__(self, stim_val=None, stim_contrast=None, tiling=None):
        self.tiling = tiling
        self.stim_val = stim_val
        self.stim_contrast = stim_contrast
        _, self.stim_idx = find_nearest(tiling, stim_val)

    def get_stim_idx(self, stim_val=None):
        if stim_val is None:
            stim_val = self.stim_val
        self.stim_val, self.stim_idx = find_nearest(self.tiling, stim_val)
        return self.stim_idx


class NeuralPopulation:
    """params and sampling for neural population with independent properties"""

    def __init__(self, name, boundaries, sampling_freq, r_max=60, semi_sat=10, exponent=4, sigma=10, spont=0.05):
        self.tunings = None
        self.prefs = None
        self.prefs_idx = None
        self.name = name
        self.boundaries = boundaries  # [min, max]
        self.sampling_freq = sampling_freq  # degrees per neuron pref
        self.sigma = sigma  # sigma of gaussian tuning
        self.r_max = r_max
        self.spont = spont
        self.semi_sat = semi_sat
        self.exponent = exponent

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

    def contrast_response(self, tuned_resp, stim_contrast, n=None, c50=None):
        # todo implement this
        # con is contrast of stimulus
        # n is exponent for steepness of curve
        if n is None:
            n = self.exponent
        # semi_saturation constant is contrast for half-max response
        if c50 is None:
            c50 = self.semi_sat
        response = tuned_resp * (stim_contrast ** n / (stim_contrast ** n + c50 ** n))  # hyperbolic ratio
        return response


class PopulationTuning:
    def __init__(self, all_prefs=None, all_prefs_idx=None, tunings=None,
                 window_prefs=None, window_prefs_idx=None):
        if tunings is None:  # attributes to be appended external from class
            tunings = []
        if all_prefs_idx is None:
            all_prefs_idx = []
        if all_prefs is None:
            all_prefs = []
        self.all_prefs = all_prefs
        self.all_prefs_idx = all_prefs_idx
        self.tunings = tunings
        self.window_prefs = window_prefs
        self.window_prefs_idx = window_prefs_idx

    def rolling_window(self, vector, window):
        # create rolling window across vector
        m = len(vector) - len(window)
        self.window_prefs_idx = hankel(np.arange(0, m + 1), np.arange(m, len(vector)))
        self.window_prefs = vector[self.window_prefs_idx]
        return self.window_prefs, self.window_prefs_idx


def shift_prefs(stim_val, possible_prefs, all_windows):
    # get central value of each prefs window
    centre_vals = np.mean([all_windows.min(1), all_windows.max(1)], 0)
    # find index of centre_val(pref_window) closest to stim_val
    window_idx = abs(centre_vals - stim_val).argmin()  # get row_idx (window) where centre_val is closest to stim_val
    # use prefs window which has stim_val at it's centre
    window_vals = all_windows[window_idx, :]
    # get range of idxs so we can index tunings array
    min_idx = abs(possible_prefs - window_vals.min()).argmin()  # start of window
    max_idx = abs(possible_prefs - window_vals.max()).argmin()  # end of window
    window_idxs = np.arange(min_idx, max_idx + 1)  # all idxs within window
    return window_vals, window_idxs


def get_boundaries(mid_values, bound_range):
    boundaries = [[]] * len(mid_values)
    for idx, val in enumerate(mid_values):
        boundaries[idx] = [val - (bound_range / 2), val + (bound_range / 2)]
    return boundaries


def adjust_adjacents(value_list, operation='mean'):
    """for each value, get average with each neighbouring value"""

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


def adjust_boundaries(boundaries, adjuster_values):
    # input bounds for each ori, and frequency averages for each ori
    # adjust boundaries to smoothly switch between different sampling rates

    for idx in range(len(boundaries)):
        boundaries[idx] = np.asarray(boundaries[idx])

    adjuster_values = np.asarray(adjuster_values)
    adjusted_bounds = [[]] * len(boundaries)

    for idx, ii in enumerate(boundaries):  # loop through each ori bounds
        adjuster = adjuster_values[idx]
        adjusted_bounds[idx] = ii + ((adjuster * [1, -1]) / 2)

    return adjusted_bounds


class PopResponse:
    def __init__(self, stim_val, stim_idx, tunings, prefs_windows, prefs_all):
        self.stim_val = stim_val
        self.stim_idx = stim_idx
        self.tunings = tunings
        self.prefs_windows = prefs_windows
        self.prefs_all = prefs_all
        self.resp_tuned = None
        self.resp_noisy = None
        self.trial_prefs = None
        self.trial_tunings = None

    def gen_stim_response(self):
        stim_val = self.stim_val
        stim_idx = self.stim_idx
        # save used window of prefs and tunings for later decoding
        self.trial_prefs, trial_prefs_idxs = shift_prefs(stim_val, self.prefs_all, self.prefs_windows)
        # tunings within prefs window for this stim_val
        self.trial_tunings = self.tunings[trial_prefs_idxs, :]  # save trial tunings for max likelihood
        # generate noiseless response from tunings
        self.resp_tuned = self.tunings[trial_prefs_idxs, stim_idx]
        # error checking that stim val is in centre of prefs window (circular tuning)
        if self.resp_tuned.argmax() != 87 and self.resp_tuned.argmax() != 88:
            print(f'stim_val not in centre of prefs window - in position {self.resp_tuned.argmax()}')
        # generate noisy response using poisson noise
        self.resp_noisy = np.random.poisson(self.resp_tuned)


# todo create trial handler class which can store data from multiple runs
class TrialHandler:
    def __init__(self, n_trials, stim_vals, stim_idxs, tunings, prefs_windows, prefs_all):
        self.n_trials = n_trials
        self.prefs_windows = prefs_windows
        self.prefs_all = prefs_all
        self.tunings = tunings
        self.stim_idxs = stim_idxs
        self.stim_vals = stim_vals
        self.resp_noiseless = []
        self.resp_noisy = []
        self.rolling_prefs = []
        self.rolling_tunings = []

    def run(self):
        resp_noisy = []
        resp_noiseless = []
        rolling_prefs = []
        rolling_tunings = []
        for idx, stim_val in enumerate(self.stim_vals):
            stim_idx = self.stim_idxs[idx]
            prefs_window_vals, prefs_window_idxs = shift_prefs(stim_val, self.prefs_windows, self.prefs_all)
            rolling_prefs.append(prefs_window_vals)  # save trial prefs window for later decoding
            rolling_tunings.append(self.tunings[prefs_window_idxs, :])  # save trial tunings for max likelihood
            noiseless = self.tunings[prefs_window_idxs, stim_idx]
            # only use tunings within prefs window for this stim_val
            # error checking that stim val is in centre of prefs window
            if noiseless.argmax() != 87 and noiseless.argmax() != 88:
                print(f'stim_val not in centre of prefs window - in position {noiseless.argmax()}')
            # repeat noiseless response for n_trials
            resp_noiseless.append(np.transpose(noiseless * np.ones([self.n_trials, 1])))
            # get noisy response for all prefs across all trials
            resp_noisy.append(np.random.poisson(resp_noiseless[-1]))

        self.rolling_prefs = rolling_prefs
        self.rolling_tunings = rolling_tunings
        self.resp_noiseless = resp_noiseless
        self.resp_noisy = resp_noisy

        return resp_noiseless, resp_noisy


def calc_error(decoded_vals, stim_val):
    abs_error = np.asarray(abs(decoded_vals - stim_val))
    mean_abs_error = np.mean(abs_error)
    return mean_abs_error


class Decoder:
    def __init__(self, resp_noisy, rolling_tunings, rolling_prefs, stim_vals, tiling):
        """initialise attributes of parent class"""
        self.rolling_prefs = rolling_prefs
        self.rolling_tunings = rolling_tunings
        self.resp_noisy = resp_noisy
        self.tiling = tiling
        self.decoded = {}
        self.stim_vals = stim_vals

    def wta(self, pop_resp=None, trial_prefs=None):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # trial_prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if trial_prefs is None:
            trial_prefs = self.rolling_prefs

        trial_max = pop_resp.max()  # max response for each trial
        ismax = (trial_max == pop_resp).astype('int')  # bool 01 matrix of max for each trial
        trial_pref_idx = (ismax * np.random.random(size=ismax.shape)).argmax()  # idx of pref with max response
        wta_est = trial_prefs[trial_pref_idx]
        # which pref produced the strongest response each trial
        self.decoded['wta'] = wta_est

        return wta_est

    def popvector(self, pop_resp=None, rolling_prefs=None):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # trial_prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if rolling_prefs is None:
            rolling_prefs = self.rolling_prefs
        trial_resp = pop_resp
        cond_prefs = rolling_prefs * (np.pi / 180)  # convert to radians
        hori = np.sum((trial_resp * np.cos(cond_prefs)))
        vert = np.sum((trial_resp * np.sin(cond_prefs)))
        popvector_est = (np.arctan2(vert, hori) * (180 / np.pi))  # inverse tangent in degrees
        self.decoded['pv'] = popvector_est
        return popvector_est

    def maxlikelihood(self, pop_resp=None, rolling_tunings=None, tiling=None):
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if rolling_tunings is None:
            rolling_tunings = self.rolling_tunings
        if tiling is None:
            tiling = self.tiling
        log_tunings = np.log10(rolling_tunings)
        log_likelihood = pop_resp.T @ log_tunings  # matrix multiplication; outputs: size=[trials, tiling]
        max_likelihood_idx = log_likelihood.argmax()  # gives idx in feature space of ML est
        maxlikelihood_est = (tiling[max_likelihood_idx])
        self.decoded['ml'] = maxlikelihood_est
        return maxlikelihood_est


class StaircaseHandler:
    def __init__(self, current_level, step_sizes, n_up, n_down, n_reversals):
        self.n_down = n_down
        self.n_up = n_up
        self.n_reversals = n_reversals
        self.is_corr = []  # tracks n of corr answers in a row
        self.current_level = current_level
        self.stepsizes = step_sizes
        self.stepsizes_idx = 0
        self.current_stepsize = step_sizes[0]
        self.level_list = [current_level]
        self.current_direction = 'down'
        self.reversals = []
        self.continue_staircase = True

    def is_correct(self, correct_ans, given_ans):
        if correct_ans == given_ans:
            self.is_corr.append(1)  # correct answer
        else:
            self.is_corr.append(0)  # wrong answer
        self.update_staircase()

    def update_level(self, direction=None):
        if direction is None:
            direction = self.current_direction
        if direction == 'up':
            self.current_level *= self.current_level * (self.current_stepsize ** 10)
        elif direction == 'down':
            self.current_level /= self.current_level * (self.current_stepsize ** 10)
        self.stepsizes_idx += 1
        if self.stepsizes_idx == self.n_reversals:
            self.continue_staircase = False

    def update_staircase(self):
        self.current_stepsize = self.stepsizes[self.stepsizes_idx]

        # check if staircase needs to step down
        if len(self.is_corr) >= self.n_down:  # only check for step up when possible
            if all(i == 1 for i in self.is_corr[-self.n_down:]):
                if self.current_direction == 'up':  # if change in direction of staircase
                    self.reversals.append(self.current_level)
                self.current_direction = 'down'
                self.update_level()

        # check if staircase needs to step up
        if len(self.is_corr) >= self.n_up:  # only check for step up when possible
            if all(i == 0 for i in self.is_corr[-self.n_up:]):
                if self.current_direction == 'down':  # if change in direction of staircase
                    self.reversals.append(self.current_level)
                self.current_direction = 'up'
                self.update_level()

        self.level_list.append(self.current_level)

cardinal = {'sampling_freq': 1, 'sigma': 10, 'r_max': 60, 'spont': 0.05}
oblique = {'sampling_freq': 1, 'sigma': 10, 'r_max': 60, 'spont': 0.05}

ori_populations = {'vertical': NeuralPopulation(name='vertical', boundaries=get_boundaries([-90, 90], 45),
                                                sampling_freq=cardinal['sampling_freq'], sigma=cardinal['sigma'],
                                                spont=cardinal['spont'], r_max=cardinal['r_max']),

                   'right_oblique': NeuralPopulation(name='right_oblique', boundaries=get_boundaries([-45, 135], 45),
                                                     sampling_freq=oblique['sampling_freq'], sigma=oblique['sigma'],
                                                     spont=oblique['spont'], r_max=oblique['r_max']),

                   'horizontal': NeuralPopulation(name='horizontal', boundaries=get_boundaries([-180, 0, 180], 45),
                                                  sampling_freq=cardinal['sampling_freq'], sigma=cardinal['sigma'],
                                                  spont=cardinal['spont'], r_max=cardinal['r_max']),

                   'left_oblique': NeuralPopulation(name='left_oblique', boundaries=get_boundaries([-135, 45], 45),
                                                    sampling_freq=oblique['sampling_freq'], sigma=oblique['sigma'],
                                                    spont=oblique['spont'], r_max=oblique['r_max']),
                   }

# adjust boundaries using avg sampling rates of neighbouring populations
bounds_adjusted = adjust_boundaries(boundaries=[ori_populations[x].boundaries for x in ori_populations],
                                    adjuster_values=adjust_adjacents(
                                        [ori_populations[x].sampling_freq for x in ori_populations]))
# assign adjusted boundaries to associated ori-tuned population
for idx, ori in enumerate(ori_populations):
    ori_populations[ori].boundaries = bounds_adjusted[idx]

# define params for tiling of extended feature space (e.g. ori is from -90 to 90, so have -180 to 180)
FeatureSpace = Tiling(min_tile=-180, max_tile=180, stepsize=0.05)

PopTuning = PopulationTuning()  # initialise attributes in PopulationTuning
for idx, ori in enumerate(ori_populations):
    ori_populations[ori].generate_prefs(tiling=FeatureSpace.tiling)  # prefs for each ori
    ori_populations[ori].generate_tunings(tiling=FeatureSpace.tiling)  # tunings for each ori
    # concatenate prefs, prefs_idx, tunings
    PopTuning.all_prefs = np.hstack([PopTuning.all_prefs, ori_populations[ori].prefs])
    PopTuning.all_prefs_idx = np.hstack([PopTuning.all_prefs_idx, ori_populations[ori].prefs_idx])
    PopTuning.tunings.append(ori_populations[ori].tunings)
PopTuning.tunings = np.vstack([i for i in PopTuning.tunings])  # concat all appended tunings
# sort prefs_idx and tunings by indices of prefs in ascending order
(PopTuning.all_prefs, PopTuning.all_prefs_idx, PopTuning.tunings) = sort_by_standard(PopTuning.all_prefs,                                                                          PopTuning.all_prefs_idx,                                                                   PopTuning.tunings)
# create array of all possible pref windows - switch between dependent on stim_val
# ensures population vector is equally accurate regardless of stim_val (circular tuning)
PopTuning.rolling_window(vector=PopTuning.all_prefs,
                         window=PopTuning.all_prefs[(PopTuning.all_prefs >= -90) & (PopTuning.all_prefs < 90)])

# model setup is complete, can begin trials
ori_std = [-45, 0, 45, 90]
contrast = [2.5, 5, 10, 20, 40]
start_val = 20
Staircase = StaircaseHandler(current_level=start_val, step_sizes=[1], n_up=1, n_down=3)
continue_exp = True

# todo contrast response function
# todo store absolute reversal value of level

while continue_exp:
    level = Staircase.current_level
    for i_ori in ori_std:  # loop through each standard orientation
        for i_con in contrast:  # loop through each value of iv
            StandardStim = Stimuli(stim_val=i_ori, stim_contrast=i_con, tiling=FeatureSpace.tiling)
            if random.random() > 0.5:  # rand ori_diff direction of rotation (probs unnecess for model)
                corr_ans = 'CW'
                ComparisonStim = Stimuli(stim_val=i_ori + level,
                                         stim_contrast=i_con, tiling=FeatureSpace.tiling)
            else:
                corr_ans = 'CCW'
                ComparisonStim = Stimuli(stim_val=i_ori - level,
                                         stim_contrast=i_con, tiling=FeatureSpace.tiling)

            StandardResp = PopResponse(stim_val=StandardStim.stim_val, stim_idx=StandardStim.stim_idx,
                                       tunings=PopTuning.tunings, prefs_windows=PopTuning.window_prefs,
                                       prefs_all=PopTuning.all_prefs)
            StandardResp.gen_stim_response()
            ComparisonResp = PopResponse(stim_val=ComparisonStim.stim_val, stim_idx=ComparisonStim.stim_idx,
                                         tunings=PopTuning.tunings, prefs_windows=PopTuning.window_prefs,
                                         prefs_all=PopTuning.all_prefs)
            ComparisonResp.gen_stim_response()

            StandardDecoded = Decoder(StandardResp.resp_noisy, StandardResp.trial_tunings,
                                      StandardResp.trial_prefs, StandardStim.stim_val, FeatureSpace.tiling)
            ComparisonDecoded = Decoder(ComparisonResp.resp_noisy, ComparisonResp.trial_tunings,
                                        ComparisonResp.trial_prefs, ComparisonStim.stim_val, FeatureSpace.tiling)

            if ComparisonDecoded.wta() < StandardDecoded.wta():
                this_ans = 'CCW'
            else:
                this_ans = 'CW'
            Staircase.is_correct(this_ans, corr_ans)  # checks if correct and updates staircase accordingly

            StandardDecoded.popvector()
            ComparisonDecoded.popvector()

            StandardDecoded.maxlikelihood()
            ComparisonDecoded.maxlikelihood()

            print('')
# todo

# Stimuli
# -	Contrast (constant)
# -	Orientation (change each loop in staircase)
# - Two intervals

# Staircase
# -	Set initial oridiff value.
# -	Change step size on each reversal
# -	Decision stage
# o	Decoded_1 < Decoded_2; n_corr += 1
# -	3 down 1 up
# o	if n_corr == 3; ori -= stepsize[i]
#
# Neural Population
# -	Contrast response function
# -	Density
# -	Tuning
# -	Max response
#
# Decoders
# -	WTA
# -	PV
# -	ML
