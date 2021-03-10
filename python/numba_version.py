import math
import random
from typing import *

import numba
import numpy as np
import pandas as pd

random.seed(1)
np.random.seed(1)


def repeat_expand(x):
    return np.repeat(x.reshape((-1, 1)), 1, axis=1)


@numba.njit(fastmath=True)
def make_ixs(init_ixs, buckets):
    ixs = np.full((init_ixs.size, buckets), 0)
    for i in range(buckets):
        mask = init_ixs == i
        ixs[mask, i] = 1
    return ixs


@numba.njit(fastmath=True, parallel=False)
def mutate(critter, mutation_p: float):
    for _ in range(mutation_p):
        r = random.randint(0, init_ixs.size - 1)
        np.random.shuffle(critter[r])


@numba.njit(fastmath=True, parallel=False)
def k_point_crossover(critter, k, parents):
    delta = len(costs) // (k * len(parents))
    start, end = 0, delta

    for _ in range(k):
        for i in range(len(parents)):
            critter[start:end] = parents[i][start:end]
            start = end
            end += delta


@numba.njit(fastmath=True)
def select_parents(critters, top_size):
    parent_count = 2
    return [critters[random.randint(0, top_size - 1)] for _ in range(parent_count)]


@numba.njit(fastmath=True, parallel=False)
def cull_mating_pool(critters, fitnessess, mating_pool_size):
    total_fitness = fitnessess.sum()

    probs = np.cumsum(fitnessess / total_fitness)

    rs = np.random.random(mating_pool_size)
    ixs = np.searchsorted(probs, rs)
    p_ixs = set(ixs)

    all_ixs = set(range(len(critters)))
    other_ixs = all_ixs.difference(p_ixs)

    ixs = np.array(list(p_ixs) + list(other_ixs))

    return critters[ixs]


@numba.njit(fastmath=True)
def mate(critters, fitnessess, top_size, mutation_count):
    # critters = cull_mating_pool(critters, fitnessess, top_size)

    k = 4
    for n, critter in enumerate(critters):
        if n > top_size:
            parents = select_parents(critters, top_size)
            k_point_crossover(critter, k, parents)
            mutate(critter, mutation_count)
        # else:
        #     t_mutation_count = mutation_count // 4
        #     mutate(critter, t_mutation_count)

    return critters


@numba.njit(fastmath=True, parallel=False)
def life(critters, n, pop_size, fitness_func):
    top_size = max(1, pop_size // 20)

    mutation_p = 0.01
    t_mutation_p = mutation_p

    delta = 0

    threshold = 50
    t_threshold = threshold

    prev = 0
    fitnessess = np.zeros(pop_size)

    for i in range(n):
        for j, critter in enumerate(critters):
            fitnessess[j] = fitness_func(critter)

        ixs = np.argsort(-fitnessess)
        fitnessess = fitnessess[ixs]
        critters = critters[ixs]
        total = fitnessess[0]

        if total > prev:
            print(i, total)
            prev = total

            delta = 0
            t_threshold = threshold
            t_mutation_p = mutation_p
        else:
            if delta > t_threshold:
                t_mutation_p = min(0.1, t_mutation_p * 1.01)
                t_threshold = min(t_threshold * 1.1, 99999.0)

                critters = cull_mating_pool(critters, fitnessess, top_size)
            else:
                delta += 1
                

        mutation_count = math.ceil(len(critters) * t_mutation_p)
        critters = mate(critters, fitnessess, top_size, mutation_count)

    return critters


def set_buckets(ixs, df: pd.DataFrame):
    for i in range(ixs.shape[1]):
        mask = ixs[..., i] == 1
        df["bucket"].values[mask] = i
    return df


buckets = 4

df = pd.read_csv("data/2021Internet-CharterSplit.csv")

init_ixs = df["bucket"].values
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


out_filepath = "data/2021-out.csv"
n = 1 * (10 ** 6)
pop_size = 500
fitness_func = calc_cost

critters = np.asarray([make_ixs(init_ixs, buckets) for _ in range(pop_size)])
critters = life(critters, n, pop_size, fitness_func)


df = set_buckets(critters[0], df)
df.to_csv(out_filepath, index=False)
