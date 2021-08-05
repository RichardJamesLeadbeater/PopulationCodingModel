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


import numpy as np
from scipy.linalg import hankel
import pandas as pd


def find_nearest(array, values):  # nearest value and index in array for each value
    nearest_idx = np.zeros(shape=values.shape, dtype='int')
    nearest_values = np.zeros(shape=values.shape)
    for idx, ii in enumerate(values):
        nearest_idx[idx] = (np.abs(array - ii)).argmin()
        nearest_values[idx] = array[nearest_idx[idx]]
    return nearest_values, nearest_idx


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

    def __init__(self, n_stim, min_stim, max_stim, tiling, distribution='rand'):
        self.n_stim = n_stim
        self.min_val = min_stim
        self.max_val = max_stim
        self.tiling = tiling

        self.distribution = distribution
        stim_vals = None
        if distribution == 'unif':  # achieves near-equal spacing of stim_vals
            stim_freq = abs(min_stim - max_stim) / n_stim
            stim_vals = np.linspace(min_stim, max_stim - stim_freq, n_stim)  # no repeats e.g. -90 and 90
        elif distribution == 'rand':  # random spacing of stim_vals
            stim_vals = np.random.uniform(min_stim, max_stim, n_stim)
        # nearest val & idx in array for each given value
        stim_vals, stim_indices = find_nearest(tiling, stim_vals)

        self.stim_vals = np.sort(stim_vals)
        self.stim_indices = np.sort(stim_indices)


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
        # con is contrast of stimulus
        # n is exponent for steepness of curve
        if n is None:
            n = self.exponent
        # semi_saturation constant is contrast for half-max response
        if c50 is None:
            c50 = self.semi_sat
        response = tuned_resp * (stim_contrast**n / (stim_contrast**n + c50**n))  # hyperbolic ratio
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


def shift_prefs(stim_val, all_windows, all_values):
    # get central value of each prefs window
    centre_vals = np.mean([all_windows.min(1), all_windows.max(1)], 0)
    # find index of centre_val(pref_window) closest to stim_ori
    window_idx = abs(centre_vals - stim_val).argmin()  # get row_idx (window) where centre_val is closest to stim_ori
    # use prefs window which has stim_ori at it's centre
    window_vals = all_windows[window_idx, :]
    # get range of idxs so we can index tunings array
    min_idx = abs(all_values - window_vals.min()).argmin()  # start of window
    max_idx = abs(all_values - window_vals.max()).argmin()  # end of window
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


class PopulationResponse:
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
        # for idx, stim_val in enumerate(self.stim_vals):
        for idx, stim_val in enumerate(self.stim_vals):
            stim_idx = self.stim_idxs[idx]
            # prefs_window_vals, prefs_window_idxs = shift_prefs(stim_val, self.prefs_windows, self.prefs_all)
            prefs_window_vals, prefs_window_idxs = shift_prefs(-80, self.prefs_windows, self.prefs_all)
            rolling_prefs.append(prefs_window_vals)  # save trial prefs window for later decoding
            rolling_tunings.append(self.tunings[prefs_window_idxs, :])  # save trial tunings for max likelihood
            noiseless = self.tunings[prefs_window_idxs, stim_idx]
            # only use tunings within prefs window for this stim_ori
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
        self.n_stim = len(stim_vals)

    def wta(self, pop_resp=None, trial_prefs=None):
        # pop_resp: 2D neuron(rows) x trial(cols) for resp(vals)
        # trial_prefs: 1D prefs(vals)
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if trial_prefs is None:
            trial_prefs = self.rolling_prefs
        wta_est = []
        for idx in range(len(pop_resp)):
            trial_max = pop_resp[idx].max(0)  # max response for each trial (col)
            ismax = (trial_max == pop_resp[idx]).astype('int')  # bool 01 matrix of max for each trial (col)
            # if two prefs with max response, randomly choose
            trial_pref_idx = (ismax * np.random.random(size=ismax.shape)).argmax(0)  # idx of pref (row) with max resp
            wta_est.append(trial_prefs[idx][trial_pref_idx])
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
        popvector_est = []
        for idx in range(len(pop_resp)):
            trial_resp = pop_resp[idx]
            cond_prefs = rolling_prefs[idx] * (np.pi / 180)  # convert to radians
            hori = np.sum((trial_resp.T * np.cos(cond_prefs)).T, 0)
            vert = np.sum((trial_resp.T * np.sin(cond_prefs)).T, 0)
            popvector_est.append(np.arctan2(vert, hori) * (180 / np.pi))  # inverse tangent in degrees
        self.decoded['pv'] = popvector_est
        return popvector_est

    def maxlikelihood(self, pop_resp=None, rolling_tunings=None, tiling=None):
        if pop_resp is None:
            pop_resp = self.resp_noisy
        if rolling_tunings is None:
            rolling_tunings = self.rolling_tunings
        if tiling is None:
            tiling = self.tiling
        maxlikelihood_est = []
        for idx, trial_resp in enumerate(pop_resp):
            log_tunings = np.log10(rolling_tunings[idx])
            log_likelihood = trial_resp.T @ log_tunings  # matrix multiplication; outputs: size=[trials, tiling]
            max_likelihood_idx = log_likelihood.argmax(1)  # gives idx in feature space of ML est
            maxlikelihood_est.append(tiling[max_likelihood_idx])
        self.decoded['ml'] = maxlikelihood_est
        return maxlikelihood_est


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
# define params for MyStimuli values within normal feature space (e.g. ori stim = [-90:90]
Stim = Stimuli(n_stim=20, min_stim=-90, max_stim=90-FeatureSpace.stepsize,
               tiling=FeatureSpace.tiling, distribution='rand')

PopTuning = PopulationTuning()
for idx, ori in enumerate(ori_populations):
    ori_populations[ori].generate_prefs(tiling=FeatureSpace.tiling)  # prefs for each ori
    ori_populations[ori].generate_tunings(tiling=FeatureSpace.tiling)  # tunings for each ori
    # concatenate prefs, prefs_idx, tunings
    PopTuning.all_prefs = np.hstack([PopTuning.all_prefs, ori_populations[ori].prefs])
    PopTuning.all_prefs_idx = np.hstack([PopTuning.all_prefs_idx, ori_populations[ori].prefs_idx])
    PopTuning.tunings.append(ori_populations[ori].tunings)  # rows=prefs, cols=tiling, vals=tuned_response
PopTuning.tunings = np.vstack([i for i in PopTuning.tunings])  # concat all appended tunings
# sort prefs_idx and tunings by indices of prefs in ascending order
(PopTuning.all_prefs, PopTuning.all_prefs_idx, PopTuning.tunings) = sort_by_standard(PopTuning.all_prefs,
                                                                                     PopTuning.all_prefs_idx,
                                                                                     PopTuning.tunings)
# create array of all possible pref windows - switch between dependent on stim_ori
# ensures population vector is equally accurate regardless of stim_ori (circular tuning)
PopTuning.rolling_window(vector=PopTuning.all_prefs,
                         window=PopTuning.all_prefs[(PopTuning.all_prefs >= -90) & (PopTuning.all_prefs < 90)])

# generate response for allPops from stim_vals
# [resp_n, noiseless, shift_prefs, shift_tunings] = genPopResponse(tunings, params);
# input tunings params
# output noisy and noiseless response; shifted prefs & tunings for each stim presentat

# get idx of min and max of pref_window, in all_prefs - index with these values
# loop through each stimulus:

PopResponse = PopulationResponse(n_trials=10, stim_vals=Stim.stim_vals, stim_idxs=Stim.stim_indices,
                                 tunings=PopTuning.tunings, prefs_windows=PopTuning.window_prefs,
                                 prefs_all=PopTuning.all_prefs)
PopResponse.run()

Decoded = Decoder(resp_noisy=PopResponse.resp_noisy, rolling_tunings=PopResponse.rolling_tunings,
                  rolling_prefs=PopResponse.rolling_prefs, tiling=FeatureSpace.tiling, stim_vals=Stim.stim_vals)

# decode (WTA, PV, ML, ?pooling?)
Decoded.wta()
Decoded.popvector()
Decoded.maxlikelihood()

analysis = {}
for method in Decoded.decoded:
    estimates = Decoded.decoded[method]
    idxs = range(len(estimates))
    analysis[method] = {'stdev': np.asarray([np.std(estimates[i]) for i in idxs]),
                        'mean_abserr': np.asarray([calc_error(estimates[i], Stim.stim_vals[i]) for i in idxs])
                        }

# stdDev = std(decoded, 0, 2);
# % calculate absolute error
# absErr = abs(decoded - transpose(ps.stim.v));

print('debug')

# todo decision stage

# todo staircase

