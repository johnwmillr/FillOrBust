from collections import Counter
from typing import Literal, NamedTuple, TypedDict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from dice.cards import Card


ActionT = Literal[
    "draw",
    "roll",
    "stop",
    "hold",
]

ArrayShape = tuple[int, int]


class State(NamedTuple):
    card: Card | None
    # banked_score: int
    turn_score: int
    dice: Counter[int]
    holds: Counter[int]


StateT = TypedDict(
    "StateT",
    {
        "card": Card | None,
        # "discard": Counter[Card],
        # "banked_score": int,
        "turn_score": int,
        "rolled": Counter[int],
        "held": Counter[int],
    },
)


# QTableKeyT = tuple[str, str, str, str]
QTableKeyT = str
QTableT = dict[QTableKeyT, dict[ActionT, float]]

Estimator = SVR | RandomForestRegressor | SGDRegressor
