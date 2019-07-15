import random
import timeit
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from dataclasses import dataclass


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


def shuffle_buckets(arr: np.ndarray,
                    intervals: List[int]) -> List[np.ndarray]:
    np.random.shuffle(arr)
    buckets = []
    start = 0
    for i in intervals:
        buckets.append(arr[start:i + start].astype(int))
        start += i
    return buckets


def calc_erate_stats(bucket: np.ndarray,
                     data: pd.DataFrame) -> Dict[str, float]:
    bucket_data = data.loc[data.index.isin(bucket)]
    M = len(bucket_data)
    stats = {"average_discount": 0,
             "total_discount": 0,
             "total_cost": 0,
             "discount_cost": 0,
             "bucket_size": 0}

    stats["total_discount"] = np.sum(bucket_data["discount"])
    stats["average_discount"] = round(stats["total_discount"] / (M * 100), 2)
    stats["total_cost"] = np.sum(bucket_data["cost"])
    stats["discount_cost"] = round(
        stats["average_discount"] * stats["total_cost"])
    stats["bucket_size"] = M

    return stats


def optimize_discount(N, intervals, data, iterations=100000):
    arr = np.arange(N)

    max_buckets = [[]]
    max_cost = 11810384
    M = len(intervals)

    for i in range(iterations):
        buckets = shuffle_buckets(arr, intervals)
        t_stats = []
        cost_i = 0
        for j in range(M):
            stats = calc_erate_stats(buckets[j], data)
            t_stats.append(stats)
            cost_i += stats["discount_cost"]

        if (cost_i > max_cost):
            print(cost_i, max_cost, cost_i - max_cost)
            print(t_stats)
            max_cost = cost_i
            max_buckets[0] = list(buckets)


@dataclass
class Critter:
    genes: List[int]
    description: Dict[str, int]
    fitness: int





def optimize_t(data: pd.DataFrame,
               bucket_count: int,
               max_bucket: int,
               mutation_count: int = 10,
               iterations: int = 1000) -> List[Dict[str, Union[int, float]]]:
    N = len(data)
    buckets = {i: {"average_discount": 0,
                   "total_discount": 0,
                   "total_cost": 0,
                   "discount_cost": 0,
                   "count": 0} for i in range(bucket_count)}
    parent_bucket = {"average_discount": 0,
                     "total_discount": 0,
                     "total_cost": 0,
                     "discount_cost": 0,
                     "count": 0}
    extrema = []

    full_buckets = [False for i in range(bucket_count)]
    genes: List[List] = [[0, 0] for i in range(N)]

    for i in range(iterations):
        print(f"Starting iteration {i}\n")
        child_bucket = {"average_discount": 0,
                        "total_discount": 0,
                        "total_cost": 0,
                        "discount_cost": 0,
                        "count": 0}

        for j in range(N):
            k = genes[j][1]\
                if genes[j][0]\
                else random.randrange(0, bucket_count)

            if (full_buckets[k]):
                while (full_buckets[k]):
                    k = (k + 1) % bucket_count

            buckets[k]["total_cost"] += data.loc[j, "cost"]
            buckets[k]["total_discount"] += data.loc[j, "discount"]
            buckets[k]["count"] += 1

            if (buckets[k]["count"] > max_bucket):
                full_buckets[k] = True

            genes[j][1] = k

        for j in range(bucket_count):
            b = buckets[j]
            child_bucket["total_cost"] += b["total_cost"]
            child_bucket["total_discount"] += b["total_discount"]
            child_bucket["count"] += b["count"]

            if (b["count"] > 0):
                b["average_discount"] = round(
                    b["total_discount"] / (100 * b["count"]), 2)
                b["discount_cost"] = round(
                    b["average_discount"] * b["total_cost"])

            child_bucket["discount_cost"] += b["discount_cost"]

            for key in b.keys():
                b[key] = 0
            full_buckets[j] = False

        print(f"child bucket discount: {child_bucket['discount_cost']}")
        print(f"parent bucket discount: {parent_bucket['discount_cost']}")

        if (child_bucket["discount_cost"] > parent_bucket["discount_cost"]):
            parent_bucket = child_bucket

            print(f"Found extremal value at iteration {i}")
            print("---")

            for key, value in parent_bucket.items():
                print(f"\t{key}: {value}")

            extrema.append(parent_bucket)

            for i in range(N):
                genes[i][0] = 1

            for i in range(mutation_count):
                j = random.randrange(0, N)
                genes[j][0] ^= 1

            print(f"\n\tgenes of iteration {i}:")
            print(f"{genes}")
            print("---\n")
            a = 1


data = pd.read_csv("erate-data.csv",
                   names=["lea-number", "discount", "cost"])[:50]

optimize_t(data, 5, 10, 25, 100000)
