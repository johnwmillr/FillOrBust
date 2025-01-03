from typing import Literal, NamedTuple


class TrainingStats(NamedTuple):
    """A class representing the statistics of a training episode."""

    game_score: int
    turn_count: int


class DiceCounts(NamedTuple):
    """A class representing the counts of each dice value."""

    ones: int = 0
    twos: int = 0
    threes: int = 0
    fours: int = 0
    fives: int = 0
    sixes: int = 0


ActionName = Literal["hold", "roll", "stop"]
Action = tuple[ActionName, DiceCounts | None]
Reward = float
Truncated = bool
Terminated = bool


class Observation(NamedTuple):
    """A class representing the state of the game."""

    dice: DiceCounts
    num_dice: int
    held_score: int  # The yet-to-be-banked score for the current turn
    last_action: ActionName | None

    def __str__(self) -> str:
        return f"<Observation: dice: {self.dice} | num_dice: {self.num_dice} | held_score: {self.held_score} | last_action: {self.last_action}>"
