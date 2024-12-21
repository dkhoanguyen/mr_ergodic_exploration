import numpy as np
from gymnasium.spaces import Box

from double_integrator import DoubleIntegrator
from ergodic_control import RTErgodicController

from scipy.stats import multivariate_normal as mvn

def main():
    mean1 = np.array([0.35, 0.38])
    cov1 = np.array([
        [0.01, 0.004],
        [0.004, 0.01]
    ])
    w1 = 0.5

    mean2 = np.array([0.68, 0.25])
    cov2 = np.array([
        [0.005, -0.003],
        [-0.003, 0.005]
    ])
    w2 = 0.2

    mean3 = np.array([0.56, 0.64])
    cov3 = np.array([
        [0.008, 0.0],
        [0.0, 0.004]
    ])
    w3 = 0.3


    # Define the Gaussian-mixture density function here
    def pdf(x):
        return w1 * mvn.pdf(x, mean1, cov1) + \
            w2 * mvn.pdf(x, mean2, cov2) + \
            w3 * mvn.pdf(x, mean3, cov3)