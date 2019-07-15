import itertools

import numpy as np
from scipy.signal import fftconvolve


def powerset(items):
    s = list(items)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r
                                         in range(len(s) + 1))


def powerset_knapsack(M, t):
    max_happiness = -1
    for S in powerset(M):
        total_cost = sum([cost for cost, happiness in S])
        total_happiness = sum([happiness for cost, happiness in S])
        if total_cost == t and total_happiness > max_happiness:
            max_happiness = total_happiness
    if max_happiness == -1:
        print('Warning: no subsets had sum, t, (i.e., subset - sum of M, t was False)')
    return max_happiness


def naive_max_convolve(x, y):
    x = np.asarray(x)
    x = np.asarray(y)

    x_n = len(x)
    y_n = len(y)
    result_n = x_n + y_n - 1
    result = np.zeros(result_n)

    for i in range(x_n):
        for j in range(y_n):
            result_index = i + j
            result[result_index] = max(result[result_index], x[i] * y[j])

    return result


def numeric_max_convolve(x, y, log_p_max, epsilon=1e-10):
    x = np.asarray(x)
    x = np.asarray(y)
    x_n = len(x)
    y_n = len(y)

    result_n = x_n + y_n - 1
    result = np.zeros(result_n)
    all_p = 2.0**np.arange(log_p_max)

    for p in all_p:
        result_for_p = fftconvolve(x**p, y**p)
        stable = result_for_p > epsilon
        result[stable] = result_for_p[stable]**(1.0 / p)

    return result


# M = [(3, 1), (5, 10), (7, 6), (9, 2), (11, 3), (13, 8), (109, 11),
#      (207, 4), (113, 7), (300, 18)]


# M = [(5, 5), (2, 2), (1, 1)]
# N = 8
# print(M)
# t = powerset_knapsack(M, N)
# print(t)
x = np.array([0.1, 0.3, 0.8, 0.5])
y = np.array([0.2, 0.15, 0.725, 0.4])


# print(fftconvolve(x, y))

# for p in 2**np.arange(10):
#     print(f"{p}", np.round(np.fabs(fftconvolve(x**p, y**p))**(1.0 / p), 3))

print(naive_max_convolve(x, y))
t = numeric_max_convolve(x, y, 10)
print(np.round(t, 4))
