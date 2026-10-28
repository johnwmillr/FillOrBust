from random import choices
from typing import List, Optional, Tuple

import numpy as np


class Die(int):
    POINTERS = [1, 5]

    def __init__(self, value: int) -> None:
        self.value = value
        self._is_held = False

    def hold(self) -> None:
        self._is_held = True

    @property
    def is_held(self) -> bool:
        return self._is_held

    @property
    def is_pointer(self) -> bool:
        return self in self.POINTERS

    def __repr__(self) -> str:
        return f"D{self.value}"


class Dice:
    def __init__(self, num_dice: int = 6, num_sides: int = 6) -> None:
        self.num_dice = num_dice
        self.num_sides = num_sides
        self.active: Optional[List[Die]] = None
        self.held: Optional[List[Die]] = None
        self.state: Optional[np.ndarray] = None

    def roll(self) -> None:
        self.active = sorted(map(Die, choices(range(1, self.num_sides + 1), k=self.num_active)))
        self.encode_state()

    @property
    def num_active(self) -> int:
        if self.active is None:
            return self.num_dice
        return len([die for die in self.active if not die.is_held])

    @property
    def num_held(self) -> int:
        if self.held is None:
            return 0
        return len([die for die in self.held if die.is_held])

    def encode_state(self) -> None:
        """One-hot encode the number of pointers"""
        new_state = np.zeros((1, 2 * self.num_dice))
        new_state[0, sum(die.is_pointer for die in self.active)] = 1
        new_state[0, self.num_dice + self.num_held] = 1
        if self.state is None:
            self.state = new_state
        else:
            self.state = np.concatenate([self.state, new_state])


def get_possible_actions_from_state(state: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    print(state)
    if state[0]:
        return "bust", np.zeros_like(state)
    else:
        return "hold", state


def main() -> None:
    num_rolls = 5
    dice = Dice()
    for n in range(num_rolls):
        dice.roll()
        print(dice.active)
    print(dice.state)
    get_possible_actions_from_state(dice.state[-1])
    print()


if __name__ == "__main__":
    main()
