import sys
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snu_frame_ordering.orders import (  # noqa: E402
    ALL_IMAGE_ORDERS,
    SIGMAS,
    answer_from_image_order,
    image_order_from_answer,
    image_order_from_slot_order,
    inv_perm,
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


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nall {len(tests)} permutation tests passed")
