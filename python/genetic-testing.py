import math
import random
import timeit
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt

random.seed(10)


@dataclass
class Critter:
    fitness: float = 0


fs = [1, 2, 3, 4]

pop_size = 281
critters = []
total_fitness: float = 0
multiplier = 1 / 2810

for i in range(pop_size):
    r = random.randrange(1, 99) / 100
    print(r)
    # r = fs[i] * multiplier
    critters.append(Critter(fitness=r))
    total_fitness += r


print("---")
d = {}

prev_p: float = 0
for n, c in enumerate(critters):
    p = prev_p + c.fitness / total_fitness

    start = int(prev_p / multiplier)
    end = int(p / multiplier)
    print(n, prev_p, p)
    d[n] = end

    prev_p = p

M = int(1 / multiplier)
d2 = {}
for i in range(M):
    r = random.randrange(0, int(1 / multiplier))
    for key, value in d.items():
        if r < value:
            if (key not in d2):
                d2[key] = 1
            else:
                d2[key] += 1
            break

for key, value in sorted(d2.items()):
    print(key, value / M)
