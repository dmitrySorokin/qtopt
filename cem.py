import numpy as np


class CEM(object):
    """ cross-entropy method, as optimization of the action policy """

    def __init__(self, theta_dim, ini_mean_scale=0.0, ini_std_scale=1.0):
        self.theta_dim = theta_dim
        self.initialize(ini_mean_scale=ini_mean_scale, ini_std_scale=ini_std_scale)
        self.mean = None
        self.std = None

    def initialize(self, ini_mean_scale=0.0, ini_std_scale=0.33):
        self.mean = ini_mean_scale * np.ones(self.theta_dim)
        self.std = ini_std_scale * np.ones(self.theta_dim)

    def sample(self):
        # theta = self.mean + np.random.randn(self.theta_dim) * self.std
        theta = self.mean + np.random.normal(size=self.theta_dim) * self.std
        return theta

    def sample_multi(self, n):
        theta_list = []
        for i in range(n):
            theta_list.append(self.sample())
        return np.array(theta_list)

    def update(self, selected_samples):
        self.mean = np.mean(selected_samples, axis=0)
        self.std = np.std(selected_samples, axis=0)  # plus the entropy offset, or else easily get 0 std

        return self.mean, self.std
