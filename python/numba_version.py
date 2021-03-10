import numpy as np
import pandas as pd
import random
import math
from datetime import datetime
import numba
import csv
import time

random.seed(1)
np.random.seed(1)


def repeat_expand(x):
    return np.repeat(x.reshape((-1, 1)), 1, axis=1)


buckets = 4

df = pd.read_csv("data/2021Internet-CharterSplit.csv")

df_row = df.iloc[0].to_dict()

init_ixs = df["bucket"].values
costs = repeat_expand(df["cost"].values)
discounts = repeat_expand(df["discount"].values)


# @numba.njit(fastmath=True)
def make_ixs():
    ixs = np.full((init_ixs.size, buckets), 0)
    for i in range(buckets):
        mask = init_ixs == i
        ixs[mask, i] = 1
    return ixs


def make_csv(ixs):
    for i in range(buckets):
        mask = ixs[..., i] == 1
        df["bucket"].values[mask] = i
    return df


# @numba.njit(fastmath=True)
def calc_fitness(critter_ixs: np.ndarray):
    n = critter_ixs.shape[-1]

    total = 0
    for i in range(n):
        ixs = np.expand_dims(critter_ixs.T[i], 1)
        m = np.count_nonzero(ixs)
        t_d = discounts * ixs

        if m > 0:
            d = np.round(np.sum(t_d) / m) / 100.0
            c = np.sum(costs * ixs)
            s_i = c * d
            total += s_i

    return total


# @numba.njit(fastmath=True)
def mutate(critter, mutation_amount: float):
    for _ in range(mutation_amount):
        r = random.randint(0, init_ixs.size - 1)
        np.random.shuffle(critter[r])


# @numba.njit(fastmath=True)
def k_point_crossover(critter, k, parents):
    delta = len(costs) // (k * len(parents))
    start, end = 0, delta

    for _ in range(k):
        for i in range(len(parents)):
            critter[start:end] = parents[i][start:end]
            start = end
            end += delta


# @numba.njit(fastmath=True)
def select_parents(critters, top_size):
    parent_count = 4
    return [critters[random.randint(0, top_size - 1)] for _ in range(parent_count)]


# @numba.njit(fastmath=True)
def mate(critters, top_size, mutation_amount):
    k = 4
    for i in range(len(critters)):
        critter = critters[i]
        if i > top_size:
            k_point_crossover(critter, k, select_parents(critters, top_size))
            mutate(critter, mutation_amount)


def exp_backoff(threshold, c):
    m = 2 ** (c - 1)
    r = random.randint(0, m) / m
    return r * threshold


# @numba.njit(fastmath=True)
def life(critters, n, pop_size):
    top_size = max(1, pop_size // 10)

    mutation_amount = 1
    t_mutation_amount = mutation_amount

    delta = 0.0

    threshold = 100.0
    t_threshold = threshold

    prev = 0
    fitnessess = np.zeros(pop_size)

    for i in range(n):
        for j, critter in enumerate(critters):
            fitnessess[j] = calc_fitness(critter)
        ixs = np.argsort(-fitnessess)
        fitnessess = fitnessess[ixs]
        critters = critters[ixs]
        total = fitnessess[0]

        if total > prev:
            print(i, total)
            df.to_csv("tmp.csv", index=False)

            prev = total
            delta = 0
            t_threshold = threshold
            t_mutation_amount = mutation_amount
        else:
            if delta > t_threshold:
                delta = 0
                t_mutation_amount = min(
                    (len(costs) - 1) // 4, math.ceil(t_mutation_amount * 1.1)
                )
                t_threshold = min(t_threshold * 1.1, 99999.0)
            else:
                delta += 1

        mate(critters, top_size, t_mutation_amount)

    return critters


n = 100
pop_size = 10
print(df)
critters = np.asarray([make_ixs() for _ in range(pop_size)])
critters = life(critters, n, pop_size)
