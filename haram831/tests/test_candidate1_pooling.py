import pytest


torch = pytest.importorskip("torch")

from snu_ordering.candidate1.pooling import pool_last_non_padding


def test_pool_last_non_padding_handles_right_padding():
    hidden = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    pooled = pool_last_non_padding(hidden, attention_mask)

    assert torch.equal(pooled, torch.tensor([[2.0, 20.0], [7.0, 70.0]]))


def test_pool_last_non_padding_rejects_an_empty_sequence():
    hidden = torch.zeros((1, 2, 3))
    attention_mask = torch.zeros((1, 2), dtype=torch.long)

    with pytest.raises(ValueError, match="non-padding"):
        pool_last_non_padding(hidden, attention_mask)
