import itertools
import math


# N = 2
# min_size = 0
# bucket_size = 2

# n = 0
# t = []

# shape = [N] * bucket_size
# axis_counter = [0] * bucket_size

# i = 0
# while (i < N**bucket_size):

#     if (sum(axis_counter) == N):
#         c1 = [True if i >= min_size else False for i in axis_counter]
#         if (all(c1)):
#             c2 = [True if i in t else False for i in
#                   itertools.permutations(axis_counter, bucket_size)]
#             if(not any(c2)):
#                 t.append(tuple(axis_counter))
#                 n += 1

#     axis_counter[0] += 1
#     for j in range(1, bucket_size):
#         if axis_counter[j - 1] >= shape[j - 1]:
#             axis_counter[j - 1] = 0
#             axis_counter[j] += 1

#     i += 1

# print(n + 1)

# t = []
# d = {}
def choose(n, m):
    return math.factorial(n) / (math.factorial(n - m) * math.factorial(m))



def partition_function_P(n, k):
    if (n == k):
        return 1 + partition_function_P(n, k - 1)
    elif (k == 0 or n < 0):
        return 0
    elif(n == 0 or k == 1):
        return 1
    else:
        return partition_function_P(n, k - 1) + partition(n - k, k)



def partition_function_Q(n, k):
    return partition_function_P(n - int(choose(k, 2)), k)


for i in range(6, 20):
    print(q(i, 3))

