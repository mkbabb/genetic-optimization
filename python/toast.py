# from decimal import *
# import math

# getcontext().prec = 64


# print("{")
# for i in range(64):
#     t = Decimal(Decimal(2)**(Decimal(2)**Decimal(-i)))
#     print(f"{t:.52f},")
# print("}")

import math

for i in range(45):
    t = math.exp(i * 1.0)
    print(i, f"{t:52f}")
