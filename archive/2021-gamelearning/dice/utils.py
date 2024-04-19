from typing import Iterable


# Hold string to counts
# counts = tuple(map(int, hold_action.split("_")[-1]))


# Counts to dice
# [die for num, count in enumerate(counts, start=1) for die in count * (num,)]


def count_dice(dice: Iterable[int]) -> tuple[int, ...]:
    # This is stupid, but it was faster than collections.Counter
    ones, twos, threes, fours, fives, sixes = 0, 0, 0, 0, 0, 0
    for die in dice:
        if die == 1:
            ones += 1
        elif die == 2:
            twos += 1
        elif die == 3:
            threes += 1
        elif die == 4:
            fours += 1
        elif die == 5:
            fives += 1
        else:
            sixes += 1
    return ones, twos, threes, fours, fives, sixes
