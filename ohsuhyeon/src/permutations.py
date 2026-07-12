import itertools

ALL_PERMS = list(itertools.permutations([1, 2, 3, 4]))   # 24 tuples, fixed order
PERM_TO_IDX = {p: i for i, p in enumerate(ALL_PERMS)}
IDX_TO_PERM = {i: p for i, p in enumerate(ALL_PERMS)}
