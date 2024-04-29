from typing import NamedTuple


class DiceCounts(tuple[int, ...]): ...  # Should actually be length six


ActionT = tuple[
    str, int | DiceCounts | None
]  # The action name and any accompanying info (e.g. ("roll", 4))


class Observation(NamedTuple):
    hold_score: int
    dice: DiceCounts
