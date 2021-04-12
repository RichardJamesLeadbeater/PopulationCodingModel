import numpy as np


class Tile():
    """tiling of feature space"""

    def __init__(self, min, max, stepsize):
        self.min = min
        self.max = max
        self.range = abs(min - max)
        self.stepsize = stepsize
        self.tiling = np.arange(min, max, stepsize)


# define params for tiling of feature space
feature_space = Tile(min=-90, max=90, stepsize=0.05)

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
