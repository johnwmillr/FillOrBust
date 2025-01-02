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


class Observation(NamedTuple):
    """A class representing the state of the game."""

    num_dice_remaining: int
    held_score: int  # The yet-to-be-banked score for the current turn


Action = Literal["roll", "stop"]
Reward = float
Truncated = bool
Terminated = bool
