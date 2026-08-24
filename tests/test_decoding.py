import pytest
import torch

from gemma_eval.decoding import filter_logits, sample_next_token


def test_top_k_keeps_only_k_logits() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    filtered = filter_logits(logits, top_k=2, top_p=None)
    assert torch.isfinite(filtered).sum().item() == 2
    assert torch.isfinite(filtered[0, 1])
    assert torch.isfinite(filtered[0, 2])


def test_top_p_keeps_first_token_crossing_threshold() -> None:
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0]])
    filtered = filter_logits(logits, top_k=None, top_p=0.8)
    assert torch.isfinite(filtered).sum().item() == 2


def test_greedy_returns_argmax() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    token = sample_next_token(
        logits,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        generator=None,
    )
    assert token.tolist() == [[1]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_p": 0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
    ],
)
def test_invalid_filter_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        filter_logits(torch.ones(1, 4), **kwargs)
