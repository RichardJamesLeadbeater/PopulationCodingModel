import numpy as np


def find_nearest(array, value):  # return value and index of nearest value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_indices(array, values):  # nearest index of array for each value
    indices = [None] * len(values)
    for idx, ii in enumerate(values):
        _, indices[idx] = find_nearest(array, ii)
    return indices


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
    def __init__(self, n_stim, min_val, max_val, tiling, distribution):
        self.n_stim = n_stim
        self.min_val = min_val
        self.max_val = max_val

        self.distribution = distribution
        if distribution == 'unif':
            self.stim_vals = np.linspace(min_val, max_val, n_stim)
        elif distribution == 'rand':
            self.stim_vals = np.random.uniform(min_val, max_val, n_stim)

        self.stim_idx =
        # self.stim_vals = np.random.choice(self.stim_tiling, n_stim)
        # _, self.min_idx = find_nearest(tiling, min_val)
        # _, self.max_idx = find_nearest(tiling, max_val)
        # self.stim_tiling = tiling[self.min_idx: self.max_idx + 1]  # index into feature space for stimuli range


# define params for tiling of feature space
feature_space = Tiling(min_val=-90, max_val=90, stepsize=0.05)

# define params for stimuli values

# define params for tuning of neural populations

# define params for response of neural populations

# each neural population as instance of class
# boundaries
# bandwidth
# sampling frequency
# max firing rate
# preferences
# genTunings()

# adjust boundaries of each population based on sampling frequencies

# piece together tuningArrays and prefVectors from each population into allPops

# generate response for allPops from stim_vals

# decode allPops responses (WTA, PV, ML)
