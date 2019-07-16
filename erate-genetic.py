import math
import random
import timeit
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

SEED = 0xDEADBEEF

np.random.seed(SEED)
random.seed(SEED)


def swap(arr, ix1, ix2):
    t = arr[ix1]
    arr[ix1] = arr[ix2]
    arr[ix2] = t


def knuth_shuffle(arr):
    for i in range(len(arr)):
        swap(arr, i, random.randint(0, i))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Critter(object):
    def __init__(self, N):
        self.genes: List[List[int]] = [[0, 0] for i in range(N)]
        self.fitness: float = 0

    def __repr__(self):
        return f"{self.fitness}"


def calc_erate_stats(data: pd.DataFrame,
                     critter: Critter,
                     buckets: Dict[int, Dict[str, float]],
                     max_bucket: int,
                     full_buckets: List[bool]) -> None:
    N = len(data)
    bucket_count = len(buckets)
    for j in range(N):
        k = critter.genes[j][1]\
            if critter.genes[j][0]\
            else random.randint(0, bucket_count - 1)

        if (full_buckets[k]):
            while (full_buckets[k]):
                k = random.randint(0, bucket_count - 1)

        buckets[k]["total_cost"] += data.loc[j, "cost"]
        buckets[k]["total_discount"] += data.loc[j, "discount"]
        buckets[k]["count"] += 1

        if (buckets[k]["count"] > max_bucket):
            full_buckets[k] = True

        critter.genes[j][1] = k

    t: float = 0
    for j in range(bucket_count):
        b = buckets[j]
        if (b["count"] > 0):
            b["average_discount"] = round(
                b["total_discount"] / (100 * b["count"]), 2)
            b["discount_cost"] = round(
                b["average_discount"] * b["total_cost"])

        t += b["discount_cost"]

        for key in b.keys():
            b[key] = 0

        full_buckets[j] = False

    critter.fitness = float(t)


def and_critter(critter1: Critter,
                critter2: Critter) -> None:
    for i in range(len(critter1.genes)):
        critter1.genes[i][0] &= critter2.genes[i][0]


def mate_critters(data: pd.DataFrame,
                  critters: List[Critter],
                  max_fitness: float,
                  mutation_count: int,
                  parent_count: int = 2) -> List[Critter]:
    N = len(data)
    L = len(critters)

    ratio: float = 0
    for i in range(L):
        if (critters[i].fitness > 0):
            critters[i].fitness = (critters[i].fitness / max_fitness - 0.9) * 10
            if (critters[i].fitness > 0):
                ratio += 1 / critters[i].fitness
            else:
                ratio += 0
    critters = sorted(critters, key=lambda x: x.fitness, reverse=True)

    M = int(L * ratio)
    weights: Dict[int, int] = {}
    k = 0
    for i in range(L):
        count = int(M * critters[i].fitness)
        for j in range(count):
            weights[k] = i
            k += 1

    children = critters[:parent_count]
    for i in range(L - parent_count):
        child = Critter(N)

        for j in range(parent_count):
            r = random.randrange(0, L)
            parent = critters[r]
            if (j == 0):
                child.genes = list(parent.genes)
            else:
                and_critter(child, parent)

        for j in range(mutation_count):
            k = random.randrange(0, N)
            child.genes[k][0] = random.randint(0, 1)

        children.append(child)

    return children


def optimize_buckets(data: pd.DataFrame,
                     bucket_count: int,
                     max_bucket: int,
                     population_size: int = 10,
                     mutation_rate: float = 0.1,
                     parent_count: int = 2,
                     iterations: int = 1000) -> List[Dict[str, Union[int, float]]]:
    N = len(data)
    max_bucket = max(max_bucket, N // bucket_count)
    buckets: Dict[int, Dict[str, float]] = {i: {"average_discount": 0,
                                                "total_discount": 0,
                                                "total_cost": 0,
                                                "discount_cost": 0,
                                                "count": 0} for i in range(bucket_count)}
    full_buckets = [False for i in range(bucket_count)]

    extremal_fitness: float = 0.1
    mutation_count = int(math.ceil(N * mutation_rate))

    critters = [Critter(N) for i in range(population_size)]

    for i in range(iterations):
        for j in range(population_size):
            critter = critters[j]

            calc_erate_stats(data,
                             critter,
                             buckets,
                             max_bucket,
                             full_buckets)

            if (critter.fitness > extremal_fitness):
                extremal_fitness = critter.fitness
                print(f"-{i}-")
                print(f"discount total: {extremal_fitness}")
                for i in range(N):
                    print(
                        f"lea: {data.loc[i, 'lea-number']}, bucket: {critter.genes[i][1]}")
                    critter.genes[i][0] = 1

        critters = mate_critters(data,
                                 critters,
                                 extremal_fitness,
                                 mutation_count,
                                 parent_count)


data = pd.read_csv("erate-data.csv",
                   names=["lea-number", "discount", "cost"])[:30]

optimize_buckets(data,
                 bucket_count=3,
                 max_bucket=2,
                 population_size=10,
                 mutation_rate=0.3,
                 parent_count=2,
                 iterations=1000)
