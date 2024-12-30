from typing import Literal, NamedTuple

from dice.cards import Card


# Create a `Dice` class that contains the counts of each die value.
class DiceCounts(NamedTuple):
    one: int = 0
    two: int = 0
    three: int = 0
    four: int = 0
    five: int = 0
    six: int = 0


ActionStr = Literal["inherit", "draw", "roll", "hold", "stop"]
ActionT = tuple[
    ActionStr, int | DiceCounts | None
]  # The action name and any accompanying info (e.g. ("hold", DiceCounts(1, 1, 0, 0, 0, 0)))


class TrainingInfo:
    """Track training information for a single step."""

    def __init__(
        self,
        num_turns: int,
        reward: float,
        game_score: int,
    ) -> None:
        self.num_turns = num_turns
        self.reward = reward
        self.game_score = game_score


class Observation(NamedTuple):
    last_action: ActionStr | None
    hold_score: int
    inheritance: int
    dice: DiceCounts
    card: Card | None

    def __str__(self) -> str:
        return (
            f"Last action: {self.last_action}\n"
            f"Hold score: {self.hold_score}\n"
            f"Inheritance: {self.inheritance}\n"
            f"Dice: {self.dice}\n"
            f"Card: {self.card}"
        )
