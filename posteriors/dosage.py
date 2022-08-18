"""
posterior joint probability function wrt., Ex 3.7, page 84 in Gelman

Note:
    * all mutliplication and power operations take place in log space 
"""
import typing
from decimal import *

import numpy as np


def posterior_p(grid_point: typing.Iterable, data: list[tuple]):
    """Returns the posterior probabtility for some given alpha, beta, data

    The mechanics of this function are specific to the data layout,
    and assumed latent model

    Args:
        data: a list of (dosage, negative count, total count)
    """

    a, b = grid_point

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    # likelihoods wrt., each datapoint
    likelihoods = []
    for dose, n, death_count in data:

        # model
        theta = sigmoid(a + multiply([b, dose]))

        # likelihood wrt., datapoint
        l = multiply([pow(theta, int(death_count)), pow(1 - theta, int(n - death_count))])
        likelihoods.append(l)

    # posterior = likelihood
    P = np.exp(np.sum(np.log(likelihoods)))

    return P


def multiply(x: list) -> float:
    """Multiply 2 numbers in log space."""

    # what sign should the result be?
    resultant_sign = np.sign(x).prod()

    # multiplication in log space on absolute values
    return resultant_sign * np.exp(np.sum(np.log(np.abs(x))))


def pow(base: float, power: int) -> float:
    """power operation in logspace

    requires the power to be an int
    """

    return multiply([base] * power)
