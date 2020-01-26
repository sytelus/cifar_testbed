import numpy as np
import itertools

#lrs = np.array([10**p for p in range(-8, 1)])
#lrs = np.concatenate((lrs, lrs*2.5, lrs*5, lrs*7.5))
lrs = np.array([10**p for p in range(-5, 1)])
lrs = np.concatenate((lrs,))


#moms = np.concatenate(([1-10**p for p in range(-3, -1)],[p/10 for p in range(1, 10,)], [0.85]))
moms = np.concatenate(([1-10**p for p in range(-2, -1)],[p/10 for p in range(1, 10,2)]))

# wd = np.array([10**p for p in range(-6, 0)])
# wd = np.concatenate((wd, wd*2.5, wd*5, wd*7.5))
wd = np.array([10**p for p in range(-6, -2)])
wd = np.concatenate((wd, wd*5))

print(lrs)
print(moms)
print(wd)

c = list(itertools.product(lrs, moms, wd))
print(len(c), c[0])
