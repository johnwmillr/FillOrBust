from collections import Counter
import random
import numpy as np
from dice.q import UnknownActionError

from dice.types import ActionT, StateT


class Game:
    NUM_DICE = 6
    TRIPLE_VALUES = {1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}
    STRAIGHT_DICE = Counter(range(1, NUM_DICE + 1))

    def __init__(self, max_score: int = 10_000) -> None:
        self.turn_score = 0
        self.game_score = 0
        self.dice = self._get_fresh_dice()
        self.holds = self._get_fresh_holds()  # For the turn
        self.history: list[ActionT | str] = []
        self.max_score = max_score

    def _get_fresh_dice(self) -> Counter[int]:
        return Counter()

    def _get_fresh_holds(self) -> Counter[int]:
        return Counter()

    @property
    def num_dice_remaining(self) -> int:
        return Game.NUM_DICE - self.holds.total()

    @property
    def state(self) -> StateT:
        return self.to_state()

    @property
    def stopped(self) -> bool:
        return self.turn_score > 0 and sum(self.holds) == 0

    @property
    def busted(self) -> bool:
        # We rolled and got no pointers
        return self.state["rolled"].total() > 0 and self.calc_score(self.dice) == 0

    @property
    def filled(self) -> bool:
        return self.num_dice_remaining == 0

    @property
    def num_ones_rolled(self) -> int:
        return np.count_nonzero(self.dice == 1, axis=1)[0]

    @property
    def num_ones_held(self) -> int:
        return self.holds[1]

    @property
    def num_twos_rolled(self) -> int:
        return np.count_nonzero(self.dice == 2, axis=1)[0]

    @property
    def num_twos_held(self) -> int:
        return self.holds[2]

    @property
    def num_threes_rolled(self) -> int:
        return np.count_nonzero(self.dice == 3, axis=1)[0]

    @property
    def num_threes_held(self) -> int:
        return self.holds[3]

    @property
    def num_fours_rolled(self) -> int:
        return np.count_nonzero(self.dice == 4, axis=1)[0]

    @property
    def num_fours_held(self) -> int:
        return self.holds[4]

    @property
    def num_fives_rolled(self) -> int:
        return np.count_nonzero(self.dice == 5, axis=1)[0]

    @property
    def num_fives_held(self) -> int:
        return self.holds[5]

    @property
    def num_sixes_rolled(self) -> int:
        return np.count_nonzero(self.dice == 6, axis=1)[0]

    @property
    def num_sixes_held(self) -> int:
        return self.holds[6]

    def to_state(self) -> StateT:
        return {
            "game_score": self.game_score,
            "num_dice_remaining": self.num_dice_remaining,
            "rolled": self.dice,
            "held": self.holds,
        }

    def roll_dice(self, num_dice: int) -> None:
        self.dice.clear()
        self.dice.update(random.randint(1, 6) for _ in range(num_dice))

    @classmethod
    def calc_score(cls, dice: Counter[int]) -> int:
        assert len(dice), "Can't score empty dice"

        # Check for a straight
        if dice == Game.STRAIGHT_DICE:
            return 1500

        # Check for triples
        score = 0
        for die, count in dice.items():
            if count == 6:
                return 2 * Game.TRIPLE_VALUES[die]
            if count >= 3:
                score += Game.TRIPLE_VALUES[die]
                dice[die] = count - 3

        # Check for any single pointers
        score += 100 * dice[1] + 50 * dice[5]
        return score

    def handle_action(self, action: ActionT) -> None:
        self.history.append(action)
        if action is None:
            self.handle_bust()
        elif action == "fill":
            self.handle_fill()
        elif action == "roll":
            self.handle_roll()
        elif action.startswith("hold"):
            self.handle_hold(action)  # type: ignore
        elif action == "stop":
            self.handle_stop()
        else:
            raise UnknownActionError(action)

    def handle_roll(self) -> None:
        self.roll_dice(self.num_dice_remaining)
        if self.busted:
            self.handle_bust()

    def handle_hold(self, hold_action: ActionT) -> None:
        assert hold_action is not None

        # Add the new held dice from this roll to the turn's held dice
        # Also remove the held dice from `self.dice`
        for die, hold_count in enumerate(hold_action.split("_")[-1], start=1):
            if hold_count != "0":
                self.holds[die] += int(hold_count)
                self.dice[die] -= int(hold_count)

        if self.filled:
            self.handle_action("fill")  # TODO: Filling shouldn't be an action
            return

        # Reset dice to indicate that they're ready to be rolled
        self.dice = self._get_fresh_dice()

    def handle_bust(self) -> None:
        self.turn_score = 0

        # New turn
        self.dice = self._get_fresh_dice()
        self.holds = self._get_fresh_holds()

    def handle_stop(self) -> None:
        self.game_score += self.calc_score(self.holds)
        self.holds = self._get_fresh_holds()

    def handle_fill(self) -> None:
        self.turn_score += self.calc_score(
            self.holds
        )  # TODO: Do I need a separate self.turn_score variable?

        # New dice
        self.dice = self._get_fresh_dice()
        self.holds = self._get_fresh_holds()

    @property
    def turn_is_over(self) -> bool:
        return self.filled or self.stopped or self.busted
        # return self.busted or self.stopped or self.filled

    @property
    def game_is_over(self) -> bool:
        return self.game_score >= self.max_score
