import math
import random
from typing import *
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import pathlib


def repeat_expand(x: np.ndarray) -> np.ndarray:
    return np.repeat(x.reshape((-1, 1)), 1, axis=1)


@numba.njit(fastmath=True)
def make_ixs(init_buckets: np.ndarray, buckets: int):
    ixs = np.full((init_buckets.size, buckets), 0)
    for i in range(buckets):
        mask = init_buckets == i
        ixs[mask, i] = 1
    return ixs


@numba.njit(fastmath=True, parallel=False)
def mutate(critter: np.ndarray, mutation_p: float) -> np.ndarray:
    for _ in range(mutation_p):
        r = random.randint(0, critters.shape[1] - 1)
        np.random.shuffle(critter[r])
    return critter


@numba.njit(fastmath=True, parallel=False)
def k_point_crossover_random(
    critter: np.ndarray, k: int, parent_ixs: np.ndarray
) -> np.ndarray:
    points = np.sort(np.random.randint(0, critters.shape[1] - 1, k + 2))

    points[0] = 0
    points[-1] = critters.shape[1]

    for i in range(1, k + 2):
        start, end = points[i - 1], points[i]

        parent = critters[parent_ixs[0]]
        parent_ixs = np.roll(parent_ixs, 1)

        critter[start:end] = parent[start:end]
    return critter


@numba.njit(fastmath=True, parallel=False)
def k_point_crossover_uniform(
    critter: np.ndarray, k: int, parent_ixs: np.ndarray
) -> np.ndarray:
    delta = len(costs) // (k * len(parent_ixs))
    start, end = 0, delta

    for _ in range(k):
        for ix in parent_ixs:
            critter[start:end] = critters[ix][start:end]
            start = end
            end += delta
    return critter


@numba.njit(fastmath=True)
def select_parents(critters: np.ndarray, top_size: int) -> np.ndarray:
    parent_count = min(top_size, 6)
    p_ixs = np.random.randint(0, top_size - 1, parent_count)
    return np.unique(p_ixs)


@numba.njit(fastmath=True)
def get_mutation_count(critters: np.ndarray, mutation_p: np.ndarray) -> int:
    return max(1, math.ceil(len(critters) * mutation_p))


@numba.njit(fastmath=True)
def mate(critters: np.ndarray, top_size: int, mutation_p: float) -> np.ndarray:
    mutation_count = get_mutation_count(critters, mutation_p)
    t_mutation_count = get_mutation_count(critters, mutation_p / 4)

    k = 4
    for n, critter in enumerate(critters):
        if n > top_size:
            parent_ixs = select_parents(critters, top_size)
            k_point_crossover_random(critter, k, parent_ixs)
            mutate(critter, mutation_count)
        elif n != 0:
            mutate(critter, t_mutation_count)

    return critters


@numba.njit(fastmath=True, parallel=False)
def promulgate_critter(max_critter: np.ndarray, critters: np.ndarray) -> np.ndarray:
    critters[:] = max_critter
    return critters


@numba.njit(fastmath=True, parallel=False)
def norm_fitnessess(fitnessess: np.ndarray) -> np.ndarray:
    return fitnessess ** 2


@numba.njit(fastmath=True, parallel=False)
def cull_mating_pool(
    critters: np.ndarray, fitnessess: np.ndarray, mating_pool_size: int
) -> np.ndarray:
    normed_fitnessess = norm_fitnessess(fitnessess / fitnessess[0])
    total_fitness = normed_fitnessess.sum()

    probs = np.cumsum(normed_fitnessess / total_fitness)

    rs = np.random.random(mating_pool_size)
    ixs = np.searchsorted(probs, rs)
    p_ixs = set(ixs)

    all_ixs = set(range(len(critters)))
    other_ixs = all_ixs.difference(p_ixs)

    ixs = np.array(list(sorted(p_ixs)) + list(other_ixs))
    return critters[ixs]


@numba.njit(fastmath=True, parallel=False)
def life(
    critters: np.ndarray,
    n: int,
    pop_size: int,
    fitness_func: Callable[[np.ndarray], float],
) -> np.ndarray:
    top_size = max(1, pop_size // 5)

    mutation_p = 0.01
    a, b = mutation_p, 0.2
    t_mutation_p = mutation_p

    delta = 0

    threshold = 100
    t_threshold = threshold

    max_threshold = n // 4

    max_fitness = 0
    fitnessess = np.zeros(pop_size)

    max_critter = None

    i = 0
    while True:
        for j, critter in enumerate(critters):
            fitnessess[j] = fitness_func(critter)

        ixs = np.argsort(-fitnessess)
        fitnessess = fitnessess[ixs]
        critters = critters[ixs]

        t_max_fitness = fitnessess[0]

        if i >= n:
            print(i, "final:", max_fitness)
            break
        elif t_max_fitness > max_fitness:
            max_critter = critters[0].copy()

            print(i, t_max_fitness, "max delta:", t_max_fitness - max_fitness)
            max_fitness = t_max_fitness

            delta = 0
            t_threshold = threshold
            t_mutation_p = mutation_p
        else:
            critters = mate(critters, top_size, t_mutation_p)

            if delta > t_threshold:
                print(i, " skipping, delta is:", delta, t_threshold, t_mutation_p)

                t_mutation_p = min(random.random() * (b - a) + a, t_mutation_p * 1.05)
                t_threshold = min(t_threshold * 2, max_threshold)

                if delta >= max_threshold:
                    print(" ***promulgating critter")
                    critters = promulgate_critter(max_critter, critters)

                    t_threshold = threshold
                    t_mutation_p = mutation_p
                else:
                    critters = cull_mating_pool(critters, fitnessess, top_size)
                delta = 0
            else:
                delta += 1

            critters[-1] = max_critter

        i += 1

    return max_critter


def set_buckets(ixs: np.ndarray, df: pd.DataFrame):
    for i in range(ixs.shape[1]):
        mask = ixs[..., i] == 1
        df["bucket"].values[mask] = i
    return df


use_last = True

t = int(datetime.timestamp())
now = datetime.now().isoformat()

random.seed(t)
np.random.seed(t)

buckets = 4

dirpath = pathlib.Path("data/2021-optimization")

tmp_filepath = dirpath.joinpath("tmp.csv")

in_filepath = dirpath.joinpath("in.csv") if not use_last else tmp_filepath
out_filepath = dirpath.joinpath(f"out-{now}-{t}.csv")

df = pd.read_csv(in_filepath)

init_buckets = df["bucket"].values
costs = repeat_expand(df["cost"].values)
discounts = repeat_expand(df["discount"].values)


@numba.njit(fastmath=True, parallel=False)
def calc_cost(ixs):
    total = 0
    for ix in ixs.T:
        ix = np.expand_dims(ix, 1)
        z_count = np.count_nonzero(ix)

        if z_count > 0:
            avg_discounts = np.round(np.sum(discounts * ix) / z_count) / 100.0
            bucket_costs = np.sum(costs * ix)

            total += avg_discounts * bucket_costs
    return total


n = 1 * (10 ** 7)
pop_size = 200
fitness_func = calc_cost

critters = np.asarray([make_ixs(init_buckets, buckets) for _ in range(pop_size)])
max_critter = life(critters, n, pop_size, fitness_func)

df = set_buckets(max_critter, df)
df.to_csv(out_filepath, index=False)
df.to_csv(tmp_filepath, index=False)
