import math
import random
from typing import *

import numba
import numpy as np


@numba.njit(fastmath=True, parallel=False)
def mutate(critter, mutation_p: float):
    for _ in range(mutation_p):
        r = random.randint(0, critter.shape[0] - 1)
        np.random.shuffle(critter[r])


@numba.njit(fastmath=True, parallel=False)
def k_point_crossover(critter, k, parents):
    delta = critter.shape[0] // (k * len(parents))
    start, end = 0, delta

    for _ in range(k):
        for i in range(len(parents)):
            critter[start:end] = parents[i][start:end]
            start = end
            end += delta


@numba.njit(fastmath=True)
def select_parents(critters, top_size):
    parent_count = min(top_size, 4)
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

    k = 4
    for n, critter in enumerate(critters):
        if n > top_size:
            parents = select_parents(critters, top_size)
            k_point_crossover(critter, k, parents)
            mutate(critter, mutation_count)
        else:
            t_mutation_count = mutation_count // 4
            mutate(critter, t_mutation_count)

    return critters


@numba.njit(fastmath=True, parallel=False)
def life(critters, n, pop_size, fitness_func):
    top_size = max(1, pop_size // 10)

    mutation_p = 0.01
    a, b = mutation_p, 0.1
    t_mutation_p = mutation_p

    delta = 0

    threshold = 100
    t_threshold = threshold

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

        if i > n:
            print("final:", max_fitness)
            break
        elif t_max_fitness > max_fitness:
            max_critter = critters[0]

            print(i, t_max_fitness, t_max_fitness - max_fitness)
            max_fitness = t_max_fitness

            delta = 0
            t_threshold = threshold
            t_mutation_p = mutation_p
        else:
            if delta > t_threshold:
                print("\tskipping, delta is:", delta, round(t_threshold, 4), round(t_mutation_p, 4))
                t_mutation_p = min(random.random() * (b - a) + a, t_mutation_p * 1.1)
                t_threshold = min(t_threshold * 1.5, 99999.0)

                critters = cull_mating_pool(critters, fitnessess, top_size)

                delta = 0
            else:
                delta += 1

        mutation_count = math.ceil(len(critters) * t_mutation_p)
        critters = mate(critters, fitnessess, top_size, mutation_count)

        critters[0] = max_critter

        i += 1

    return max_critter