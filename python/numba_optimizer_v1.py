import math
import random
from typing import *

import numba
import numpy as np


@numba.njit(cache=False, fastmath=True, parallel=False)
def mutate(critter: np.ndarray, mutation_count: int) -> np.ndarray:
    # cost, discount_costs = calc_cost(critter)
    # discount_costs /= cost
    # discount_costs = 1 - discount_costs

    for _ in range(mutation_count):
        r = random.randint(0, critter.shape[0] - 1)
        np.random.shuffle(critter[r])

        # critter[r] = 0
        # i = choice(p=discount_costs)
        # critter[r][i] = 1

    return critter


@numba.njit(cache=False, fastmath=True, parallel=False)
def k_point_crossover_random(
    critter: np.ndarray, k: int, parent_ixs: np.ndarray, critters: np.ndarray
) -> np.ndarray:
    points = np.sort(np.random.randint(0, critter.shape[0] - 1, k + 2))

    points[0] = 0
    points[-1] = critter.shape[0]

    for i in range(1, k + 2):
        start, end = points[i - 1], points[i]

        parent = critters[parent_ixs[0]]
        parent_ixs = np.roll(parent_ixs, 1)

        critter[start:end] = parent[start:end]
    return critter


@numba.njit(cache=False, fastmath=True, parallel=False)
def k_point_crossover_uniform(
    critter: np.ndarray, k: int, parent_ixs: np.ndarray, critters: np.ndarray
) -> np.ndarray:
    delta = len(costs) // (k * len(parent_ixs))
    start, end = 0, delta

    for _ in range(k):
        for ix in parent_ixs:
            critter[start:end] = critters[ix][start:end]
            start = end
            end += delta
    return critter


@numba.njit(cache=False, fastmath=True)
def select_parents(critters: np.ndarray, top_size: int) -> np.ndarray:
    parent_count = min(top_size, 2)
    p_ixs = np.random.randint(0, top_size - 1, parent_count)
    return p_ixs


@numba.njit(cache=False, fastmath=True)
def get_mutation_count(critters: np.ndarray, mutation_p: np.ndarray) -> int:
    return max(1, math.ceil(len(critters) * mutation_p))


@numba.njit(cache=False, fastmath=True)
def mate(critters: np.ndarray, top_size: int, mutation_p: float) -> np.ndarray:
    mutation_count = get_mutation_count(critters, mutation_p)
    t_mutation_count = get_mutation_count(critters, mutation_p / 4)

    k = 4
    for n, critter in enumerate(critters):
        if n >= top_size:
            parent_ixs = select_parents(critters, top_size)
            critter = k_point_crossover_random(critter, k, parent_ixs, critters)
            critter = mutate(critter, mutation_count)
        else:
            critter = mutate(critter, t_mutation_count)
        critters[n] = critter

    return critters


@numba.njit(cache=False, fastmath=True, parallel=False)
def promulgate_critter(max_critter: np.ndarray, critters: np.ndarray) -> np.ndarray:
    critters[:] = max_critter
    return critters


@numba.njit(cache=False, fastmath=True, parallel=False)
def norm_fitnessess(fitnessess: np.ndarray) -> np.ndarray:
    return fitnessess


@numba.njit(cache=False, fastmath=True, parallel=False)
def choice(p: np.ndarray, values: np.ndarray = None, size=1):
    probs = np.cumsum(p)
    probs /= probs[-1]

    rs = np.random.random(size)
    ixs = np.searchsorted(probs, rs, side="left")

    if values is not None:
        return values[ixs]
    else:
        return ixs


@numba.njit(cache=False, fastmath=True, parallel=False)
def proportionate_selection(
    critters: np.ndarray, fitnessess: np.ndarray, mating_pool_size: int
) -> np.ndarray:
    normed_fitnessess = norm_fitnessess(fitnessess / fitnessess[0])
    ixs = choice(normed_fitnessess, size=mating_pool_size)
    p_ixs = set(ixs)

    all_ixs = set(range(len(critters)))
    other_ixs = all_ixs.difference(p_ixs)

    ixs = np.array(list(sorted(p_ixs)) + list(other_ixs))
    return critters[ixs]


@numba.njit(cache=False, fastmath=True, parallel=False)
def life(
    critters: np.ndarray,
    n: int,
    pop_size: int,
    fitness_func: Callable[[np.ndarray], float],
) -> np.ndarray:
    top_size = max(1, pop_size // 10)

    mutation_p = 0.01
    a, b = mutation_p, 0.1
    t_b = a

    t_mutation_p = mutation_p

    delta = 0

    threshold = 100
    t_threshold = threshold

    max_threshold = n // 6

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
            if delta > t_threshold:
                print("\t", i, " skipping, delta is:", delta, t_threshold, t_mutation_p)

                t_mutation_p = random.random() * (t_b - a) + a
                t_threshold = min(t_threshold * 1.5, max_threshold)

                t_b = min(t_b * 1.1, b)

                if delta >= max_threshold:
                    print("\t***promulgating critter")
                    critters = promulgate_critter(max_critter, critters)

                    t_threshold = threshold
                    t_mutation_p = mutation_p
                else:
                    critters = proportionate_selection(critters, fitnessess, top_size)
                delta = 0
            else:
                delta += 1

            critters = mate(critters, top_size, t_mutation_p)
            critters[0] = max_critter

        i += 1

    return max_critter