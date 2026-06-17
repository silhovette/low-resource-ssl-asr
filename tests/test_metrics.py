from src.metrics import cer, wer


def test_metrics_perfect_match():
    assert wer("hello world", "hello world") == 0.0
    assert cer("hello world", "hello world") == 0.0


def test_metrics_nonnegative():
    assert wer("hello world", "hello there") >= 0.0
    assert cer("hello world", "hello there") >= 0.0
