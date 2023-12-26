from collections import Counter
from typing import Literal, TypedDict


ActionT = Literal[
    None,
    "roll",
    "stop",
    "hold",
    "bust",
    "fill",
]

ArrayShape = tuple[int, int]

StateT = TypedDict(
    "StateT",
    {
        "game_score": int,
        "num_dice_remaining": int,
        "rolled": Counter[int],
        "held": Counter[int],
    },
)

QTableT = dict[str, dict[str, dict[ActionT, float]]]
