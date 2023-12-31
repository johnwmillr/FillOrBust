from collections import Counter
from typing import Literal, TypedDict


ActionT = Literal[
    "roll",
    "stop",
    "hold",
]

ArrayShape = tuple[int, int]

StateT = TypedDict(
    "StateT",
    {
        "banked_score": int,
        "turn_score": int,
        "rolled": Counter[int],
        "held": Counter[int],
    },
)

QTableT = dict[tuple[str, str, str, str], dict[ActionT, float]]
