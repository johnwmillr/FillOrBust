from typing import Iterable, Literal, NamedTuple

from dice.cards import Card
from dice.utils import count_dice


ActionT = Literal[
    "draw",
    "roll",
    "stop",
    "hold",
]


class Dice:

    def __init__(self, dice: Iterable[int]) -> None:
        self._dice = tuple(dice)
        self._counts = count_dice(self._dice)

    @property
    def dice(self) -> tuple[int, ...]:
        return self._dice

    @property
    def counts(self) -> tuple[int, ...]:
        return self._counts

    def __hash__(self) -> int:
        return hash(self.counts)







class DiceCounts(NamedTuple):
    ones: int = 0
    twos: int = 0
    threes: int = 0
    fours: int = 0
    fives: int = 0
    sixes: int = 0

    def total(self) -> int:
        return (
            self.ones + self.twos + self.threes + self.fours + self.fives + self.sixes
        )

    def __repr__(self) -> str:
        return f"<DiceCounts: {(self.ones, self.twos, self.threes, self.fours, self.fives, self.sixes)}>"

    def __getitem__(self, die: int) -> int:
        if die == 1:
            return self.ones
        elif die == 2:
            return self.twos
        elif die == 3:
            return self.threes
        elif die == 4:
            return self.fours
        elif die == 5:
            return self.fives
        elif die == 6:
            return self.sixes
        raise Exception(f"Invalid die: {die}")

    def __add__(self, other: object) -> "DiceCounts":
        assert isinstance(other, DiceCounts)
        return DiceCounts(
            self.ones + other.ones,
            self.twos + other.twos,
            self.threes + other.threes,
            self.fours + other.fours,
            self.fives + other.fives,
            self.sixes + other.sixes,
        )

    def __len__(self) -> int:
        return self.total()


class State(NamedTuple):
    card: Card | None
    turn_score: int
    dice: tuple[int, ...]
    holds: tuple[int, ...]


QTableT = dict[State, dict[ActionT, float]]
