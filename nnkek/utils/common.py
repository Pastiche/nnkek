from typing import Sequence, Any


def make_batches(sequence: Sequence[Any], batch_size=None) -> Sequence[Sequence[Any]]:
    if not batch_size:
        batch_size = len(sequence) // 10
    if batch_size == 0:
        batch_size = 1

    return [sequence[i : i + batch_size] for i in range(0, len(sequence), batch_size)]
