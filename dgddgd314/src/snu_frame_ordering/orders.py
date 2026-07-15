from itertools import permutations


ALL_IMAGE_ORDERS = [list(p) for p in permutations([1, 2, 3, 4])]


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


def target_text(image_order):
    return str(list(image_order))

