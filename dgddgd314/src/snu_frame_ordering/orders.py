from itertools import permutations


ALL_IMAGE_ORDERS = [list(p) for p in permutations([1, 2, 3, 4])]
ALL_ORDERS = ALL_IMAGE_ORDERS

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


# Fraction of identity answers in the raw train.csv, copied from the trial4
# train-side statistic. The non-identity permutations are treated uniformly.
TRAIN_P_IDENTITY = 1478 / 9535
IDENTITY_IDX = ALL_IMAGE_ORDERS.index([1, 2, 3, 4])


def log_prior_vector(p_identity=TRAIN_P_IDENTITY):
    """Log train prior aligned to ALL_IMAGE_ORDERS in canonical space."""
    import math

    other = (1.0 - p_identity) / (len(ALL_IMAGE_ORDERS) - 1)
    return [
        math.log(p_identity) if idx == IDENTITY_IDX else math.log(other)
        for idx in range(len(ALL_IMAGE_ORDERS))
    ]


def aggregate_scores(score_matrix, prior_alpha=0.0, p_identity=TRAIN_P_IDENTITY):
    """Mean over TTA sigmas plus an optional train-derived identity prior."""
    if not score_matrix:
        raise ValueError("score_matrix must contain at least one TTA row")
    totals = [sum(col) / len(score_matrix) for col in zip(*score_matrix)]
    if prior_alpha:
        totals = [
            score + prior_alpha * prior
            for score, prior in zip(totals, log_prior_vector(p_identity))
        ]
    return totals


def exact_match(preds_answer, truths_answer):
    correct = sum(
        1 for pred, truth in zip(preds_answer, truths_answer)
        if list(pred) == list(truth)
    )
    return correct / max(1, len(truths_answer))


def kendall_distance(a, b):
    """Pairwise-inversion distance between two canonical image orders."""
    pa = {value: idx for idx, value in enumerate(a)}
    pb = {value: idx for idx, value in enumerate(b)}
    return sum(
        1
        for i in range(1, 5)
        for j in range(i + 1, 5)
        if (pa[i] - pa[j]) * (pb[i] - pb[j]) < 0
    )


def adjacent_transpositions(order):
    out = []
    for idx in range(3):
        candidate = list(order)
        candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
        out.append(candidate)
    return out


def hard_candidate_set(truth, n_random=2, rng=None):
    """truth + adjacent swaps + random negatives, with truth first."""
    import random

    rng = rng or random
    truth = list(truth)
    seen = {tuple(truth)}
    candidates = [truth]

    for order in adjacent_transpositions(truth):
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            candidates.append(order)

    pool = [order for order in ALL_IMAGE_ORDERS if tuple(order) not in seen]
    rng.shuffle(pool)
    candidates.extend(pool[:n_random])
    return candidates


def broadened_candidate_set(
    truth,
    n_adjacent=3,
    n_near=2,
    n_random=1,
    rng=None,
):
    """trial4 stage-2 candidate set covering adjacent and near confusions."""
    import random

    rng = rng or random
    truth = list(truth)
    seen = {tuple(truth)}
    candidates = [truth]

    adjacent = adjacent_transpositions(truth)
    rng.shuffle(adjacent)
    for order in adjacent[:n_adjacent]:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            candidates.append(order)

    near_pool = [
        order for order in ALL_IMAGE_ORDERS
        if tuple(order) not in seen and kendall_distance(truth, order) in (2, 3)
    ]
    rng.shuffle(near_pool)
    for order in near_pool[:n_near]:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            candidates.append(order)

    random_pool = [
        order for order in ALL_IMAGE_ORDERS
        if tuple(order) not in seen
    ]
    rng.shuffle(random_pool)
    candidates.extend(random_pool[:n_random])
    return candidates
