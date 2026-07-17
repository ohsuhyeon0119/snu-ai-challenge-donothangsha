"""Exhaustive unit tests for the permutation math in common.py.

Pure CPU / no model. Run locally before anything touches a GPU:
    python test_perms.py
"""
from itertools import permutations

from common import (ALL_ORDERS, SIGMAS, answer_from_image_order,
                    image_order_from_answer, image_order_from_slot_order,
                    inv_perm, slot_order, target_text)


def test_roundtrip_answer_image_order():
    for a in permutations([1, 2, 3, 4]):
        a = list(a)
        io = image_order_from_answer(a)
        assert answer_from_image_order(io) == a, (a, io)


def test_sigma_inverse():
    for s in permutations([1, 2, 3, 4]):
        inv = inv_perm(list(s))
        for j, v in enumerate(s, start=1):
            assert inv[v - 1] == j


def test_slot_order_matches_direct_recompute():
    """slot_order(io, sigma) must equal what you get by physically permuting
    the inputs with sigma and recomputing the order from the permuted answer."""
    for a in permutations([1, 2, 3, 4]):
        a = list(a)
        io = image_order_from_answer(a)
        for sigma in permutations([1, 2, 3, 4]):
            sigma = list(sigma)
            # Physically permuted row: slot j displays Input_{sigma[j]},
            # so the slot's temporal position is a[sigma[j]-1].
            answer_pres = [a[sigma[j] - 1] for j in range(4)]
            io_pres = image_order_from_answer(answer_pres)
            assert slot_order(io, sigma) == io_pres, (a, sigma)
            # and the inverse mapping returns to canonical space
            assert image_order_from_slot_order(io_pres, sigma) == io


def test_all_orders_cover_and_unique():
    assert len(ALL_ORDERS) == 24
    assert len({tuple(o) for o in ALL_ORDERS}) == 24
    assert all(sorted(o) == [1, 2, 3, 4] for o in ALL_ORDERS)


def test_sigmas_valid():
    assert SIGMAS[0] == [1, 2, 3, 4], "first sigma must be identity"
    assert all(sorted(s) == [1, 2, 3, 4] for s in SIGMAS)
    assert len({tuple(s) for s in SIGMAS}) == len(SIGMAS)


def test_target_text_equal_char_layout():
    texts = [target_text(o) for o in ALL_ORDERS]
    assert len({len(t) for t in texts}) == 1, "completions must be equal length"
    assert len({t.replace("1", "N").replace("2", "N").replace("3", "N")
                .replace("4", "N") for t in texts}) == 1, \
        "completions must differ only in the digits"


if __name__ == "__main__":
    import sys
    mod = sys.modules["__main__"]
    fns = [v for k, v in vars(mod).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} permutation tests passed")
