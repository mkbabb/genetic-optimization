import numpy as np
from scipy.signal import fftconvolve
import pandas as pd


def coin_poly(c):
    t = [0] * (c + 1)
    t[0] = 1
    t[c] = 1
    return np.asarray(t)


def to_coin(arr):
    c = []
    for i in range(1, len(arr)):
        if (arr[i] > 0):
            c.append(i)
    return c


c1 = coin_poly(5)
c2 = coin_poly(2)
c3 = coin_poly(1)

coins = [1, 2, 5]
polys = [coin_poly(i) for i in coins]
dp = list(polys)
N = 33

start = False
stop = False


# coins = [1, 2, 5]
# polys = [[i, coin_poly(i)] for i in coins]
# N = 12
# conv = c3

# for i in range(12):
#     conv = fftconvolve(conv, c3).astype(int)
#     t = to_coin(conv)
#     print(t)


# knap([5, 2, 1], 10)

# df = pd.read_csv("erate-data.csv",
#                  columns=["lea-number", "discount", "cost"])






