import numpy as np
import pandas as pd
import random
import math
from datetime import datetime

random.seed(1)
np.random.seed(1)

buckets = 4

df = pd.read_csv("data/2021Internet-CharterSplit.csv")

repeat_expand = lambda x: np.repeat(x.reshape((-1, 1, 1)), 1, axis=1)

init_buckets = df["bucket"].values
costs = repeat_expand(df["cost"].values)
discounts = repeat_expand(df["discount"].values)


def make_ixs():
    ixs = np.full((init_buckets.size, buckets, 1), np.nan)
    for i in range(buckets):
        mask = init_buckets == i
        ixs[mask, i,] = 1
    return ixs


def calc_fitness(critter_ixs):
    d = np.round(np.nanmean(discounts * critter_ixs, axis=0)) / 100.0
    c = np.nansum(costs * critter_ixs, axis=0)
    return np.nansum(d * c, axis=0)


def mutate(critter, mutation_amount: float):
    j = 0
    while j < mutation_amount:
        r = random.randint(0, critter.shape[0] - 1)
        np.random.shuffle(critter[r])
        j += 1


def k_point_crossover(critter, k, parent_ixs, critters):
    delta = len(costs) // (k * len(parent_ixs))
    start, end = 0, delta

    for _ in range(k):
        for i in range(len(parent_ixs)):
            p_i = (..., parent_ixs[i])
            critter[start:end] = critters[p_i][start:end]
            start = end
            end += delta


def select_parent_ixs(critters, top_size):
    parent_count = 4
    return [random.randint(0, critters.shape[0] - 1) for _ in range(parent_count)]


def mate(critters, top_size, mutation_amount):
    k = 2
    for i in range(critters.shape[-1]):
        critter = critters[..., i]
        if i > top_size:
            parent_ixs = select_parent_ixs(critters, top_size)
            k_point_crossover(critter, k, parent_ixs, critters)
        mutate(critter, mutation_amount)


def life(n, pop_size):
    top_size = max(1, pop_size // 100)

    mutation_amount = 1
    t_mutation_amount = mutation_amount

    delta = 0.0

    threshold = 100.0
    t_threshold = threshold

    prev = 0

    critters = np.concatenate([make_ixs() for _ in range(pop_size)], axis=2)

    for i in range(n):
        fitnessess = calc_fitness(critters)
        ixs = np.argsort(-fitnessess)

        fitnessess = fitnessess[ixs]
        critters = critters[..., ixs]
        s = fitnessess[0]

        if s > prev:
            print(i, s)

            prev = s
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


life(100000, 10000)

