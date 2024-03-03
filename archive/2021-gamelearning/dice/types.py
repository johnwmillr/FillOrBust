from collections import Counter
from typing import Literal, TypedDict
from dice.cards import Card


ActionT = Literal[
    "draw",
    "roll",
    "stop",
    "hold",
]

ArrayShape = tuple[int, int]

StateT = TypedDict(
    "StateT",
    {
        "card": Card | None,
        # "discard": Counter[Card],
        "banked_score": int,
        "turn_score": int,
        "rolled": Counter[int],
        "held": Counter[int],
    },
)


QTableKeyT = tuple[str, str, str, str]
QTableT = dict[QTableKeyT, dict[ActionT, float]]
