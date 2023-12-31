from collections import Counter
from copy import deepcopy
import random

from dice.q import UnknownActionError
from dice.types import ActionT, StateT


class Game:
    NUM_DICE = 6
    TRIPLE_VALUES = {1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}
    STRAIGHT_DICE = Counter(range(1, NUM_DICE + 1))

    def __init__(self, max_score: int = 10_000) -> None:
        self.turn_score = 0
        self.banked_score = 0
        self.dice: Counter[int] = Counter()
        self.holds: Counter[int] = Counter()
        self.history: list[ActionT | str] = ["start"]
        self.max_score = max_score
        self.states: list[StateT] = [deepcopy(self.state)]

    @property
    def num_dice_remaining(self) -> int:
        return Game.NUM_DICE - self.holds.total()

    @property
    def state(self) -> StateT:
        return self.to_state()

    def to_state(self) -> StateT:
        return {
            "turn_score": self.turn_score,
            "rolled": self.dice,
            "held": self.holds,
        }

    @classmethod
    def calc_score(cls, dice: Counter[int]) -> int:
        assert len(dice), "Can't score empty dice"
        dice = deepcopy(dice)

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
        if action == "roll":
            self.handle_roll()
        elif action.startswith("hold"):
            self.handle_hold(action)  # type: ignore
        elif action == "stop":
            self.handle_stop()
        else:
            raise UnknownActionError(action)
        self.history.append(action)
        self.states.append(deepcopy(self.state))

    def roll_dice(self, num_dice: int) -> Counter[int]:
        return Counter(random.randint(1, 6) for _ in range(num_dice))

    def handle_roll(self) -> None:
        self.dice = self.roll_dice(self.num_dice_remaining)
        if self.busted:
            self.handle_bust()

    def handle_hold(self, hold_action: ActionT) -> None:
        assert hold_action is not None

        # Add the new held dice from this roll to the turn's held dice
        # Also remove the held dice from `self.dice`
        for die, hold_count in enumerate(hold_action.split("_")[-1], start=1):
            if hold_count != "0":
                self.holds[die] += int(hold_count)

        if self.holds.total() == self.NUM_DICE:
            self.handle_fill()
        self.dice = Counter()  # Reset dice to indicate we're ready to roll again

    def handle_stop(self) -> None:
        self.turn_score += self.calc_score(self.holds)
        self.banked_score += self.turn_score

    @property
    def stopped(self) -> bool:
        return self.banked_score > 0

    def handle_bust(self) -> None:
        self.history.append("bust")
        self.turn_score = 0

    @property
    def busted(self) -> bool:
        return self.state["rolled"].total() > 0 and self._no_pointers_on_dice(self.dice)

    def _no_pointers_on_dice(self, dice: Counter[int]) -> bool:
        return (dice[1] + dice[5]) == 0 and max(dice.values()) < 3

    def handle_fill(self) -> None:
        self.history.append("fill")
        self.turn_score += self.calc_score(self.holds)
        self.dice = Counter()
        self.holds = Counter()

    @property
    def turn_is_over(self) -> bool:
        return self.stopped or self.busted

    @property
    def game_is_over(self) -> bool:
        return self.banked_score >= self.max_score
