import numpy as np
from scipy.stats import norm

class Filter:

    def __init__(self, mean, min, max, id, ftr_pt):
        self.mean = 1/mean
        self.min = 1/min
        self.max = 1/max
        self.variance = np.power(self.min, 2) / 36
        self.a = 10
        self.b = 10
        self.ref_keyframe = id # keyframe id that the filter is associated to
        self.feature_point = ftr_pt # image point in the keyframe that the filter is associated to

    def update(self, estimated_depth, tau_sq):

        s_sq = 1./((1. / self.variance) + (1./tau_sq))
        m = s_sq * ((self.mean / self.variance) + (estimated_depth/tau_sq))
        C1 = (self.a / (self.a + self.b)) * norm.pdf(estimated_depth, self.mean, np.sqrt(self.variance + tau_sq))
        C2 = (self.b / (self.a + self.b)) * 1/(self.min)

        # normalization = C1 + C2
        # C1 /= normalization
        # C2 /= normalization

        new_mean = C1 * m + C2 * self.mean
        new_variance = C1 * (s_sq + m * m) + C2 * (self.variance + self.mean * self.mean) - new_mean * new_mean

        eq_25 = C1 * (self.a + 1.)/(self.a + self.b + 1.) + C2* (self.a)/(self.a + self.b + 1.)
        eq_26 = C1 * (self.a + 1.) * (self.a + 2.) / ((self.a + self.b + 1.) * (self.a + self.b + 2.)) \
            + C2 * self.a * (self.a + 1.) / ((self.a + self.b + 1.) * (self.a + self.b + 2.))

        # Update the self
        self.mean = new_mean
        self.variance = new_variance
        self.a = (eq_26-eq_25)/(eq_25-eq_26/eq_25)
        self.b = self.a*(1. - eq_25)/eq_25