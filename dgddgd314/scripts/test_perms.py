import random
import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snu_frame_ordering.orders import (  # noqa: E402
    ALL_IMAGE_ORDERS,
    SIGMAS,
    adjacent_transpositions,
    aggregate_scores,
    answer_from_image_order,
    broadened_candidate_set,
    hard_candidate_set,
    image_order_from_answer,
    image_order_from_slot_order,
    inv_perm,
    kendall_distance,
    log_prior_vector,
    slot_order,
    target_text,
)


def test_roundtrip_answer_image_order():
    for answer in permutations([1, 2, 3, 4]):
        answer = list(answer)
        image_order = image_order_from_answer(answer)
        assert answer_from_image_order(image_order) == answer, (answer, image_order)


def test_sigma_inverse():
    for sigma in permutations([1, 2, 3, 4]):
        sigma = list(sigma)
        inv = inv_perm(sigma)
        for slot, value in enumerate(sigma, start=1):
            assert inv[value - 1] == slot


def test_slot_order_matches_physical_recompute():
    for answer in permutations([1, 2, 3, 4]):
        answer = list(answer)
        image_order = image_order_from_answer(answer)
        for sigma in permutations([1, 2, 3, 4]):
            sigma = list(sigma)
            presented_answer = [answer[sigma[slot] - 1] for slot in range(4)]
            presented_order = image_order_from_answer(presented_answer)
            assert slot_order(image_order, sigma) == presented_order, (answer, sigma)
            assert image_order_from_slot_order(presented_order, sigma) == image_order


def test_all_orders_cover_and_unique():
    assert len(ALL_IMAGE_ORDERS) == 24
    assert len({tuple(order) for order in ALL_IMAGE_ORDERS}) == 24
    assert all(sorted(order) == [1, 2, 3, 4] for order in ALL_IMAGE_ORDERS)


def test_sigmas_valid():
    assert SIGMAS[0] == [1, 2, 3, 4]
    assert all(sorted(sigma) == [1, 2, 3, 4] for sigma in SIGMAS)
    assert len({tuple(sigma) for sigma in SIGMAS}) == len(SIGMAS)


def test_target_text_equal_layout():
    texts = [target_text(order) for order in ALL_IMAGE_ORDERS]
    normalized = {
        text.replace("1", "N").replace("2", "N").replace("3", "N").replace("4", "N")
        for text in texts
    }
    assert len({len(text) for text in texts}) == 1
    assert len(normalized) == 1


def test_log_prior_vector():
    prior = log_prior_vector()
    assert len(prior) == 24
    assert prior[ALL_IMAGE_ORDERS.index([1, 2, 3, 4])] > prior[ALL_IMAGE_ORDERS.index([1, 2, 4, 3])]


def test_aggregate_scores_mean_and_prior():
    mat = [[0.0] * 24, [2.0] * 24]
    totals = aggregate_scores(mat)
    assert totals == [1.0] * 24

    mat = [[0.0] * 24]
    totals = aggregate_scores(mat, prior_alpha=1.0)
    assert max(range(24), key=lambda idx: totals[idx]) == ALL_IMAGE_ORDERS.index([1, 2, 3, 4])


def test_kendall_distance_basic():
    assert kendall_distance([1, 2, 3, 4], [1, 2, 3, 4]) == 0
    assert kendall_distance([1, 2, 3, 4], [2, 1, 3, 4]) == 1
    assert kendall_distance([1, 2, 3, 4], [4, 3, 2, 1]) == 6


def test_kendall_distance_symmetric_and_counts():
    counts = {distance: 0 for distance in range(7)}
    base = [1, 2, 3, 4]
    for order in ALL_IMAGE_ORDERS:
        distance = kendall_distance(base, order)
        assert distance == kendall_distance(order, base)
        counts[distance] += 1
    assert counts == {0: 1, 1: 3, 2: 5, 3: 6, 4: 5, 5: 3, 6: 1}


def test_adjacent_transpositions():
    truth = [1, 2, 3, 4]
    adjacent = adjacent_transpositions(truth)
    assert adjacent == [[2, 1, 3, 4], [1, 3, 2, 4], [1, 2, 4, 3]]
    assert all(kendall_distance(truth, order) == 1 for order in adjacent)


def test_hard_candidate_set():
    truth = [1, 2, 3, 4]
    candidates = hard_candidate_set(truth, n_random=2, rng=random.Random(0))
    assert candidates[0] == truth
    assert len(candidates) == 6
    assert len({tuple(order) for order in candidates}) == 6
    assert all(sorted(order) == [1, 2, 3, 4] for order in candidates)


def test_broadened_candidate_set():
    truth = [1, 2, 3, 4]
    candidates = broadened_candidate_set(truth, rng=random.Random(0))
    assert candidates[0] == truth
    assert len(candidates) == 7
    assert len({tuple(order) for order in candidates}) == 7
    distances = [kendall_distance(truth, order) for order in candidates[1:]]
    assert sum(distance == 1 for distance in distances) == 3
    assert sum(distance in (2, 3) for distance in distances) >= 2


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nall {len(tests)} permutation tests passed")
