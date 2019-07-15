import timeit

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import random

np.random.seed(10)
random.seed(10)


def swap(arr, ix1, ix2):
    t = arr[ix1]
    arr[ix1] = arr[ix2]
    arr[ix2] = t


def knuth_shuffle(arr):
    for i in range(len(arr)):
        swap(arr, i, random.randint(0, i))


def case1(N, intervals):
    arr = np.arange(N)
    np.random.shuffle(arr)

    buckets = []
    start = 0
    stop = 0
    for i in intervals:
        print(start, start + i)
        buckets.append(arr[start:i + start])
        start += i
    print(buckets)


def case2(N, intervals):
    buckets = [[] for i in range(len(intervals))]
    ix1 = 0
    ix2 = 0
    for i in range(N):
        buckets[ix2].append(random.randint(0,i))
        if ix1 > intervals[ix2]:
            ix1 = 0
            ix2 += 1
        else:
            ix1 += 1

    print(buckets)


case1(30, [10, 10, 10])
case2(30, [10, 10, 10])
