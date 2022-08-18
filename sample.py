""" sample the posterior distribution ... see configs

Note: 
    * probabtility distribution operations are carried out by logsumexp
    * mutliplications, power operations are dealt with in log space
"""

import itertools
import json
import pathlib
import typing

import numpy as np
import pandas as pd

# load all of the defined posterior_f functions
for p in pathlib.Path("posteriors").glob("*.py"):
    exec(f"import posteriors.{p.stem}")


def main():

    with open("sample_configs.json", "r") as f:
        configs = json.load(f)

    for config in configs:

        # ------
        # get options
        # ------
        csv = pd.read_csv(pathlib.Path(config["csv"]).expanduser())
        posterior_p: typing.Callable = eval(config["posterior_p"])
        latents_ranges: dict = config["latents_ranges"]
        n = int(config["n"])

        # ------
        # convert csv to a list of csv rows
        # ------
        data: list[tuple] = [tuple(r) for r in csv.to_numpy()]

        # ------
        # sample a grid of posterior probability values for defined latent
        # variable combinations given in latents_ranges
        # ------

        # init the grid as a ndarray
        latent_vars = list(latents_ranges.keys())  # latent variable labls

        # get a list of axis (latent variable) tick values to consider for each latent var
        grid_ords = [
            np.arange(*latents_ranges[latent_var]) for latent_var in latent_vars
        ]

        # get a list of number of ticks per each
        grid_dims = [len(grid_ord) for grid_ord in grid_ords]

        # init the grid
        grid = np.zeros(grid_dims)

        # build an iterator of axies tick combinations (i.e., grid coords)
        grid_point_coords = itertools.product(*grid_ords)

        # build an interator of grid array indices corresponding to grid_points
        grid_point_axes_indices = itertools.product(
            *[range(len(grid_ord)) for grid_ord in grid_ords]
        )

        # iterate over each combinations of discrete latent variable positions on the grid
        for grid_point_coords, grid_point_indices in zip(
            grid_point_coords, grid_point_axes_indices
        ):

            # populate probability wrt., grid point
            grid[tuple(grid_point_indices)] = posterior_p(grid_point_coords, data)

        # normalise the grid
        grid = np.exp(grid - logsumexp(grid))

        # ------
        # sample the latent vars based on the posterior joint grid
        # ------
        samples: np.ndarray = sample(grid, grid_ords, n)

        # ------
        # return probability interval
        # ------
        for i, latent_var in enumerate(latent_vars):
            pi = [np.percentile(samples[:, i], 2.5), np.percentile(samples[:, i], 97.5)]
            print(f"{latent_var} 95% PI: {pi}")


def sample(grid: np.ndarray, grid_ords, n: int):
    """return an ndarray joint samples of latent vars

    Args:
        grid (ndarray) of latent var joint vars
        grid_ords (list[list[float]]), where e.g., grid_ords[17] is the latent
            vars values corresponding to axis=17 of grid
        n (int): the number of posterior samples to take

    Return:
        n x len(grid_ords) ndarray, where returned[0] is the posterior samples
        of latent var 0
    """

    # init container to store samples
    n_latent_vars = len(grid_ords)
    samples = np.zeros([n, n_latent_vars])

    for i in range(n):

        # stack
        # list of grid axes (i.e., latent vars) still to be sampled
        axes_yet_to_sample: list[int] = list(range(n_latent_vars))

        # record sampled values of latent variables
        # e.g., sampled_values=[0.7, 0.8] means latent_var 0, has the value 0.7
        # sampled; and latent_var1 has the value 0.8 sampled in the current
        # sample set
        sampled_axes_indices = []

        while axes_yet_to_sample:

            # axis to sample
            axis: int = axes_yet_to_sample.pop(0)

            # estimate conditional probatility of axis to be samples
            non_normalised_conditional: np.ndarray = grid[
                tuple(sampled_axes_indices)
            ].sum(axis=tuple(axes_yet_to_sample))

            # normalise the conditional probability
            conditional = np.exp(
                non_normalised_conditional - logsumexp(non_normalised_conditional)
            )

            sampled_grid_index = np.random.choice(
                range(len(grid_ords[axis])), p=conditional
            )

            sampled_axes_indices.append(sampled_grid_index)

            # populate samples with actual latent variable value
            sampled_grid_value = grid_ords[axis][sampled_grid_index]
            samples[i, axis] = sampled_grid_value

    return samples


def logsumexp(x):
    """
    where x is a vector.

    see https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    i.e., normalised_probabilities = np.exp(x - logsumexp(x)), where x are log likelihoods
    """
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


if __name__ == "__main__":
    main()
