from typing import Literal, NamedTuple

from dice.cards import Card


class DiceCounts(tuple[int, ...]): ...  # Should actually be length six


ActionT = tuple[
    Literal["inherit", "draw", "roll", "hold", "stop"], int | DiceCounts | None
]  # The action name and any accompanying info (e.g. ("roll", 4))


class StepInfo:
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
    hold_score: int
    num_dice_remaining: int
    dice: DiceCounts
    card: Card | None
    inheriting: bool
