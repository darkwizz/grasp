# ---- ITERATION -----------------------------------------------------------------------------------
import random

import itertools


def shuffled(a):
    """ Returns an iterator of values in the list, in random order.
    """
    for v in sorted(a, key=lambda v: random.random()):
        yield v


def chunks(a, n=2):
    """ Returns an iterator of tuples of n consecutive values.
    """
    return zip(*(a[i::n] for i in range(n)))

# for v in chunks([1, 2, 3, 4], n=2): # (1, 2), (3, 4)
#     print(v)


def nwise(a, n=2):
    """ Returns an iterator of tuples of n consecutive values (rolling).
    """
    a = itertools.tee(a, n)
    a = (itertools.islice(a, i, None) for i, a in enumerate(a))
    a = zip(*a)
    a = iter(a)
    return a

for v in nwise([1, 2, 3, 4, 5, 6, 7, 8, 9], n=3): # (1, 2), (2, 3), (3, 4)
    print(v)


def choice(a, p=[]):
    """ Returns a random element from the given list,
        with optional (non-negative) probabilities.
    """
    p = list(p)
    n = sum(p)
    x = random.uniform(0, n)
    if n == 0:
        return random.choice(a)
    for v, w in zip(a, p):
        x -= w
        if x <= 0:
            return v

# f = {'a': 0, 'b': 0}
# for i in range(100):
#     v = choice(['a', 'b'], p=[0.9, 0.1])
#     f[v] += 1
#
# print(f)
