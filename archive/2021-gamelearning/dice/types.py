from typing import Literal, TypedDict


ActionT = Literal[
    None,
    "roll",
    "stop",
]

StateT = TypedDict(
    "StateT",
    {
        "score": int,
        "num_dice": int,
        "rolled": dict[int, int],
        "held": dict[int, int],
    },
)

QTableT = dict[str, dict[str, dict[ActionT, float]]]
