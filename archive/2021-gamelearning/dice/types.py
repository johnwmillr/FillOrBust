from typing import Literal, TypedDict


ActionT = Literal[
    None,
    "roll",
    "stop",
    "hold_01",
    "hold_02",
    "hold_03",
    "hold_04",
    "hold_05",
    "hold_06",
    "hold_10",
    "hold_11",
    "hold_12",
    "hold_13",
    "hold_14",
    "hold_15",
    "hold_20",
    "hold_21",
    "hold_22",
    "hold_23",
    "hold_24",
    "hold_30",
    "hold_31",
    "hold_32",
    "hold_33",
    "hold_40",
    "hold_41",
    "hold_42",
    "hold_50",
    "hold_51",
    "hold_60",
]

# StateT = tuple[int, int, int, int, int, int]
StateT = TypedDict(
    "StateT",
    {
        "hold_score": int,
        "final_score": int,
        "num_dice": int,
        "rolled": dict[int, int],
        "held": dict[int, int],
    },
)

QTableT = dict[str, dict[str, dict[ActionT, float]]]
# QTableT = dict[str, dict[ActionT, float]]


VALID_HOLDS = (
    "hold_01",
    "hold_02",
    "hold_03",
    "hold_04",
    "hold_05",
    "hold_06",
    "hold_10",
    "hold_11",
    "hold_12",
    "hold_13",
    "hold_14",
    "hold_15",
    "hold_20",
    "hold_21",
    "hold_22",
    "hold_23",
    "hold_24",
    "hold_30",
    "hold_31",
    "hold_32",
    "hold_33",
    "hold_40",
    "hold_41",
    "hold_42",
    "hold_50",
    "hold_51",
    "hold_60",
)
