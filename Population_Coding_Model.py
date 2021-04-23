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


def find_nearest(array, values):  # nearest value and index in array for each value
    nearest_idx = np.zeros(shape=values.shape, dtype='int')
    nearest_values = np.zeros(shape=values.shape)
    for idx, ii in enumerate(values):
        nearest_idx[idx] = (np.abs(array - ii)).argmin()
        nearest_values[idx] = array[nearest_idx[idx]]
    return nearest_values, nearest_idx


def rolling_window(vector, window):
    # create rolling window across vector
    m = len(vector) - len(window)
    windows_idxs = hankel(np.arange(0, m + 1), np.arange(m, len(vector)))
    windows_vals = vector[windows_idxs]
    return windows_idxs, windows_vals


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

    def __init__(self, min_val, max_val, stepsize):
        self.min_val = min_val
        self.max_val = max_val
        self.range = abs(min_val - max_val)
        self.stepsize = stepsize
        self.tiling = np.arange(min_val, max_val, stepsize)


class Stimuli:
    """values of MyStimuli across feature space"""

    def __init__(self, n_stim, min_val, max_val, tiling, distribution='rand'):
        self.n_stim = n_stim
        self.min_val = min_val
        self.max_val = max_val
        self.tiling = tiling

        self.distribution = distribution
        if distribution == 'unif':  # achieves near-equal spacing of stim_vals
            stim_freq = abs(min_val - max_val) / n_stim
            self.stim_vals = np.linspace(min_val, max_val - stim_freq, n_stim)  # no repeats e.g. -90 and 90
        elif distribution == 'rand':  # random spacing of stim_vals
            self.stim_vals = np.random.uniform(min_val, max_val, n_stim)

        # nearest val & idx in array for each given value
        self.stim_vals, self.stim_indices = find_nearest(tiling, self.stim_vals)


class NeuralPopulation:
    """params and sampling for neural population with independent properties"""

    def __init__(self, name, boundaries, sampling_freq, sigma=10, rmax=60, spont=0.05):
        self.tunings = None
        self.prefs = None
        self.prefs_idx = None
        self.name = name
        self.boundaries = boundaries  # [min, max]
        self.sampling_freq = sampling_freq  # degrees per neuron pref
        self.sigma = sigma  # sigma of gaussian tuning
        self.rmax = rmax
        self.spont = spont

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
        tunings = tunings * self.rmax + (self.spont * self.rmax)  # normalise to rmax then + spont firing
        self.tunings = tunings
        return self.tunings

# todo conversion of prefs to nearest val in tiling - can be done once prefs vector is joined together


def get_boundaries(mid_values, bound_range):
    boundaries = [[]] * len(mid_values)
    for idx, val in enumerate(mid_values):
        boundaries[idx] = [val-(bound_range/2), val+(bound_range/2)]
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


# input bounds for each ori, and frequency averages for each ori
def adjust_boundaries(boundaries, adjuster_values):
    # adjust boundaries to smoothly switch between different sampling rates

    for idx in range(len(boundaries)):
        boundaries[idx] = np.asarray(boundaries[idx])

    adjuster_values = np.asarray(adjuster_values)
    adjusted_bounds = [[]] * len(boundaries)

    for idx, ii in enumerate(boundaries):  # loop through each ori bounds
        adjuster = adjuster_values[idx]
        adjusted_bounds[idx] = ii + ((adjuster * [1, -1]) / 2)

    return adjusted_bounds


ori_populations = {'vertical': NeuralPopulation(name='vertical', boundaries=get_boundaries([-90, 90], 45),
                                                sampling_freq=1, sigma=10,
                                                spont=0.05, rmax=60),

                   'right_oblique': NeuralPopulation(name='right_oblique', boundaries=get_boundaries([-45, 135], 45),
                                                     sampling_freq=1, sigma=10,
                                                     spont=0.05, rmax=60),

                   'horizontal': NeuralPopulation(name='horizontal', boundaries=get_boundaries([-180, 0, 180], 45),
                                                  sampling_freq=1, sigma=10,
                                                  spont=0.05, rmax=60),

                   'left_oblique': NeuralPopulation(name='left_oblique', boundaries=get_boundaries([-135, 45], 45),
                                                    sampling_freq=1, sigma=10,
                                                    spont=0.05, rmax=60)
                   }

sampling_freqs = [ori_populations[x].sampling_freq for x in ori_populations]

freq_avgs = adjust_adjacents(value_list=[ori_populations[x].sampling_freq for x in ori_populations])

bounds_adjusted = adjust_boundaries(boundaries=[ori_populations[x].boundaries for x in ori_populations],
                                    adjuster_values=freq_avgs)

for idx, ori in enumerate(ori_populations):
    ori_populations[ori].boundaries = bounds_adjusted[idx]

# define params for tiling of feature space
FeatureSpace = Tiling(min_val=-180, max_val=180, stepsize=0.05)

# define params for MyStimuli values within feature space
MyStimuli = Stimuli(n_stim=100, min_val=-90, max_val=90-FeatureSpace.stepsize, tiling=FeatureSpace.tiling,
                    distribution='rand')

PopulationResponse = Params()
PopulationResponse.prefs = np.asarray([])
# all_prefs = np.asarray([])
PopulationResponse.prefs_idx = np.asarray([])
# all_prefs_idx = np.asarray([])
PopulationResponse.tunings = []
# all_tunings = []
for idx, ori in enumerate(ori_populations):
    ori_populations[ori].generate_prefs(tiling=FeatureSpace.tiling)  # prefs for each ori
    ori_populations[ori].generate_tunings(tiling=FeatureSpace.tiling)  # tunings for each ori

    PopulationResponse.prefs = np.hstack([PopulationResponse.prefs, ori_populations[ori].prefs])  # all prefs
    PopulationResponse.prefs_idx = np.hstack([PopulationResponse.prefs_idx, ori_populations[ori].prefs_idx])  # all idx
    PopulationResponse.tunings.append(ori_populations[ori].tunings)  # rows=prefs, cols=tiling, vals=tuned_response

PopulationResponse.tunings = np.vstack([i for i in PopulationResponse.tunings])
print('debug')

# todo gen_tunings function (outside of class) use boundaries to define sigma vals
# e.g. for each pref loop through ori_populations to find appropriate boundary (long-winded)
#   or create gaussian tunings within NeuralPopulation then piece together (issue with indexing)
#   if you know index of each preference then piecing together is no issue

# each neural population as instance of class
# boundaries
# bandwidth
# sampling frequency
# max firing rate
# preferences
# genTunings()

# todo extend preferences to repeat across feature space
# ext_prefs = np.asarray(list(prefs-180) + list(prefs) + list(prefs+180))
# ext_prefs[(ext_prefs > -180) & (ext_prefs < 180)]

# define params for tuning of neural populations

# define params for response of neural populations


# adjust boundaries of each population based on sampling frequencies

# piece together tuningArrays and prefVectors from each population into allPops

# generate response for allPops from stim_vals

# decode allPops responses (WTA, PV, ML)

print('debug')
