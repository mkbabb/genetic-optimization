from decimal import *
import math

getcontext().prec = 64


print("{")
for i in range(64):
    # t = Decimal(1) / Decimal(Decimal(2)**(Decimal(2)**Decimal(-i)))
    t = Decimal(2)**Decimal(-i)
    print(f"{t:.52f},")
print("}")
