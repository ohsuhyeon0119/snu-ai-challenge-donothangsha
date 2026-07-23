from itertools import permutations


ALL_IMAGE_ORDERS = [list(p) for p in permutations([1, 2, 3, 4])]

# Deterministic presentation orders used for train-time augmentation and
# inference-time TTA. sigma[j] is the 1-based canonical Input index displayed
# in contact-sheet slot j+1.
SIGMAS = [
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    [2, 4, 1, 3],
    [3, 1, 4, 2],
    [2, 1, 4, 3],
    [4, 2, 3, 1],
    [3, 4, 1, 2],
    [1, 3, 2, 4],
]


def image_order_from_answer(answer):
    """Competition Answer -> chronological image order."""
    answer = list(answer)
    return [answer.index(pos) + 1 for pos in range(1, 5)]


def answer_from_image_order(image_order):
    """Chronological image order -> competition Answer."""
    answer = [0, 0, 0, 0]
    for temporal_pos, input_index in enumerate(image_order, start=1):
        answer[input_index - 1] = temporal_pos
    return answer


def inv_perm(sigma):
    """Return inverse permutation: inv[v - 1] = 1-based slot of value v."""
    inv = [0, 0, 0, 0]
    for slot, value in enumerate(sigma, start=1):
        inv[value - 1] = slot
    return inv


def slot_order(image_order, sigma):
    """Canonical chronological order -> displayed slot-space order."""
    inv = inv_perm(sigma)
    return [inv[input_index - 1] for input_index in image_order]


def image_order_from_slot_order(order, sigma):
    """Displayed slot-space order -> canonical chronological image order."""
    return [sigma[slot - 1] for slot in order]


def target_text(image_order):
    return str(list(image_order))
