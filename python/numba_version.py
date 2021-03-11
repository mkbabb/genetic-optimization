import math
import os
import random
from typing import *

import numba
import numpy as np
import pandas as pd

import time


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
        r = random.randint(0, init_ixs.size)
        np.random.shuffle(critter[r])


@numba.njit(fastmath=True, parallel=False)
def k_point_crossover(critter, k, parent_ixs):
    delta = len(costs) // (k * len(parent_ixs))
    start, end = 0, delta

    points = np.sort(np.random.randint(0, critters.shape[0], k + 2))
    
    points[0] = 0
    points[-1] = critters.shape[0]

    for i in range(1, k):
        start, end = points[i - 1], points[i]
        for ix in parent_ixs:
            critter[start: end] = critters[ix][start:end]

@numba.njit(fastmath=True)
def select_parents(critters, top_size):
    parent_count = min(top_size, 4)

    p_ixs = set([random.randrange(0, top_size) for _ in range(parent_count)])
    # other_ixs = set([random.randrange(0, len(critters)) for _ in range(parent_count//2)])

    return np.array(list(p_ixs))


@numba.njit(fastmath=True)
def get_mutation_count(critters, mutation_p):
    return max(1, math.ceil(len(critters) * mutation_p))


@numba.njit(fastmath=True)
def mate(critters, top_size, mutation_p):
    mutation_count = get_mutation_count(critters, mutation_p)
    t_mutation_count = get_mutation_count(critters, mutation_p / 2)

    k = 2
    for n, critter in enumerate(critters):
        if n > top_size:
            parent_ixs = select_parents(critters, top_size)
            k_point_crossover(critter, k, parent_ixs)
            mutate(critter, mutation_count)
        else:
            mutate(critter, t_mutation_count)

    return critters


@numba.njit(fastmath=True, parallel=False)
def promulgate_critter(max_critter, critters):
    critters.fill(max_critter)
    return critters


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


@numba.njit(fastmath=True, parallel=False)
def life(critters, n, pop_size, fitness_func):
    top_size = max(1, pop_size // 10)

    mutation_p = 0.01
    a, b = mutation_p, 0.1
    t_mutation_p = mutation_p

    delta = 0

    threshold = 100
    t_threshold = threshold

    max_threshold = n // 8

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
            max_critter = critters[0]

            print(i, t_max_fitness, "max delta:", t_max_fitness - max_fitness)
            max_fitness = t_max_fitness

            delta = 0
            t_threshold = threshold
            t_mutation_p = mutation_p
        else:
            if delta > t_threshold:
                print(" skipping, delta is:", delta, t_threshold, t_mutation_p)

                t_mutation_p = min(random.random() * (b - a) + a, t_mutation_p * 1.1)
                t_threshold = min(t_threshold * 1.5, max_threshold)

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

        critters = mate(critters, top_size, t_mutation_p)

        i += 1

    return max_critter


def set_buckets(ixs, df: pd.DataFrame):
    for i in range(ixs.shape[1]):
        mask = ixs[..., i] == 1
        df["bucket"].values[mask] = i
    return df


use_last = True

t = int(time.time())
random.seed(t)
np.random.seed(t)

buckets = 4

tmp_filepath = "data/2021-optimization/tmp.csv"

in_filepath = "data/2021-optimization/in.csv" if not use_last else tmp_filepath
out_filepath = f"data/2021-optimization/out-{t}.csv"


df = pd.read_csv(in_filepath)

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


n = 1 * (10 ** 5)
pop_size = 100
fitness_func = calc_cost

critters = np.asarray([make_ixs(init_ixs, buckets) for _ in range(pop_size)])
max_critter = life(critters, n, pop_size, fitness_func)

df = set_buckets(max_critter, df)
df.to_csv(out_filepath, index=False)
df.to_csv(tmp_filepath, index=False)