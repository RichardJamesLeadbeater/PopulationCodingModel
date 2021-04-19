#
# Population Coding Model
#
# Important:
# Feature space extended from (-90:+90) to (-180:+180) for a moving prefs window which provides circular tuning;
# ...stimuli are only presented within (-90:+90) range
# Preferences for each trial only contain values within a 180 degree range;
# ...these values change depending upon the stimulus value;
# ...e.g. if stimulus value == -45, the preferences used will be (-135:135)
# ...     if stimulus value == 85, the preferences used will be (-5:175)


import numpy as np


def find_nearest(array, values):  # nearest value and index in array for each value
    nearest_idx = np.zeros(shape=values.shape, dtype='int')
    nearest_values = np.zeros(shape=values.shape)
    for idx, ii in enumerate(values):
        nearest_idx[idx] = (np.abs(array - ii)).argmin()
        nearest_values[idx] = array[nearest_idx[idx]]
    return nearest_values, nearest_idx


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
    """values of stimuli across feature space"""

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

    def __init__(self, boundaries, sampling_freq, tuning_bandwidth=10, max_firing=60, spontaneous_firing=0.05):
        self.boundaries = boundaries  # [min, max]
        self.sampling_freq = sampling_freq  # degrees per neuron
        self.tuning_sigma = tuning_bandwidth  # sigma of gaussian tuning
        self.max_firing = max_firing
        self.spontaneous_firing = spontaneous_firing

    def gen_tunings(self, prefs, sigma):
        """pass in prefs and generate tuning"""
        pass


# todo calculate prefs from boundaries and sampling frequency
# todo prefs as nearest vals in tiling


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


vertical = NeuralPopulation(boundaries=[-67.5, 67.5], sampling_freq=1, tuning_bandwidth=10,
                            spontaneous_firing=0.05, max_firing=60)

right_oblique = NeuralPopulation(boundaries=[22.5, 67.5], sampling_freq=8, tuning_bandwidth=15,
                                 spontaneous_firing=0.05, max_firing=50)

horizontal = NeuralPopulation(boundaries=[-22.5, 22.5], sampling_freq=5, tuning_bandwidth=10,
                              spontaneous_firing=0.05, max_firing=60)

left_oblique = NeuralPopulation(boundaries=[-67.5, -22.5], sampling_freq=3, tuning_bandwidth=15,
                                spontaneous_firing=0.05, max_firing=50)

# todo outside of class adjust boundaries based on relative sampling frequencies
# calculate average sampling rate for transition between neighbouring populations
# calculate avg_neighbour input sampling rates for each neural pop

freq_avgs = adjust_adjacents(value_list=[vertical.sampling_freq, right_oblique.sampling_freq,
                                         horizontal.sampling_freq, left_oblique.sampling_freq])


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


bounds_adjusted = adjust_boundaries(boundaries=[vertical.boundaries, right_oblique.boundaries,
                                     horizontal.boundaries, left_oblique.boundaries],
                         adjuster_values=freq_avgs)



# define params for tiling of feature space
feature_space = Tiling(min_val=-180, max_val=180, stepsize=0.05)

# define params for stimuli values within feature space
stimuli = Stimuli(n_stim=100, min_val=-90, max_val=90 - feature_space.stepsize, tiling=feature_space.tiling,
                  distribution='rand')

print('debug')

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
