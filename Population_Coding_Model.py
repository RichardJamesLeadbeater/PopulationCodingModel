#
# Population Coding Model
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

    def __init__(self, name, boundaries, sampling_freq, tuning_bandwidth=10, max_firing=60, spontaneous_firing=0.05):
        self.prefs = None
        self.name = name
        self.boundaries = boundaries  # [min, max]
        self.sampling_freq = sampling_freq  # degrees per neuron
        self.tuning_sigma = tuning_bandwidth  # sigma of gaussian tuning
        self.max_firing = max_firing
        self.spontaneous_firing = spontaneous_firing

    def generate_prefs(self):
        # produce vector of orientation preferences for each neuron in population
        # adjust sampling rate to find nearest whole number of nNrns

        n_neurons = abs(self.boundaries[0] - self.boundaries[1]) / self.sampling_freq
        n_neurons = int(np.round(n_neurons, 0))  # round to nearest possible n_neurons for desired sampling rate
        prefs = np.linspace(self.boundaries[0], self.boundaries[1], n_neurons)
        adjusted_freq = abs(self.boundaries[0] - self.boundaries[1]) / (len(prefs) - 1)

        self.prefs = prefs
        self.sampling_freq = adjusted_freq

        return prefs, adjusted_freq

    def gen_tunings(self, prefs, sigma):
        """pass in prefs and generate tuning"""
        pass


# todo calculate prefs from boundaries and sampling frequency
# get_prefs input: adj_bounds, sampling_freq..  output: prefs, adj_sampling_freq
# todo conversion of prefs to nearest val in tiling
# todo rollingWindow


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
    bounds = np.asarray(boundaries)
    adjuster_values = np.asarray(adjuster_values)
    adjusted_bounds = [[]] * len(bounds)

    for idx, ii in enumerate(boundaries):  # loop through each ori bounds
        adjuster = adjuster_values[idx]
        adjusted_bounds[idx] = ii + ((adjuster * [1, -1]) / 2)

    return adjusted_bounds


ori_populations = {'vertical': NeuralPopulation(name='vertical', boundaries=[-67.5, 67.5], sampling_freq=1, tuning_bandwidth=10,
                                                spontaneous_firing=0.05, max_firing=60),

                   'right_oblique': NeuralPopulation(name='right_oblique', boundaries=[22.5, 67.5], sampling_freq=4, tuning_bandwidth=15,
                                                     spontaneous_firing=0.05, max_firing=50),

                   'horizontal': NeuralPopulation(name='horizontal', boundaries=[-22.5, 22.5], sampling_freq=1, tuning_bandwidth=10,
                                                  spontaneous_firing=0.05, max_firing=60),

                   'left_oblique': NeuralPopulation(name='left_oblique', boundaries=[-67.5, -22.5], sampling_freq=4, tuning_bandwidth=15,
                                                    spontaneous_firing=0.05, max_firing=50)
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
MyStimuli = Stimuli(n_stim=100, min_val=-90, max_val=90 - FeatureSpace.stepsize, tiling=FeatureSpace.tiling,
                    distribution='rand')

# generate preferences for each orientation
for ori in ori_populations:
    ori_populations[ori].generate_prefs()



print('debug')

# todo fix vertical generate_prefs
# ... v.bounds = [[-112.5, -67.5], [67.5, 112.5]]
# ... loop through to generate prefs
# ... with adjust bounds loop you know the lower bound adjustment and the upper bound adjustment
# ... so just use the lower and upper adjustments equally on each bound  - see notes

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
