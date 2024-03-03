from collections import Counter
from copy import deepcopy
import random

from dice.cards import Card, Deck
from dice.q import UnknownActionError
from dice.types import ActionT, StateT


class Game:
    NUM_DICE = 6
    TRIPLE_VALUES = {1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}
    STRAIGHT_DICE = Counter(range(1, NUM_DICE + 1))

    def __init__(self, max_score: int | None = None) -> None:
        self.turn_score = 0
        self.turn_scores = []
        self.banked_score = 0
        self.num_turns = 0
        self.dice: Counter[int] = Counter()
        self.holds: Counter[int] = Counter()
        self.deck = Deck()
        self.card: Card | None = None
        self.discard: Counter[Card] = Counter()
        self.history: list[ActionT | str] = ["start"]
        self.max_score = max_score if max_score else 5_000
        self._start_new_turn()

    def _start_new_turn(self) -> None:
        self.history.append(f"NEW TURN, Score: {self.banked_score}")
        self.num_turns += 1
        self.turn_score = 0
        self._reset_dice_holds_and_card()

    def _reset_dice_holds_and_card(self) -> None:
        self.dice = Counter()
        self.holds = Counter()
        if self.card and self.card.category == "no_dice":
            self.discard[self.card] += 1
        self.card = None

    @property
    def num_dice_remaining(self) -> int:
        return Game.NUM_DICE - self.holds.total()

    @property
    def state(self) -> StateT:
        return self.to_state()

    def to_state(self) -> StateT:
        return {
            "card": self.card,
            # "discard": self.discard,
            "banked_score": self.banked_score,
            "turn_score": self.turn_score,
            "rolled": self.dice,
            "held": self.holds,
        }

    @property
    def stopped(self) -> bool:
        return self.banked_score > 0

    @property
    def busted(self) -> bool:
        def no_pointers_on_dice(dice: Counter[int]) -> bool:
            return (dice[1] + dice[5]) == 0 and max(dice.values()) < 3

        return self.state["rolled"].total() > 0 and no_pointers_on_dice(self.dice)

    def roll_dice(self, num_dice: int) -> Counter[int]:
        return Counter(random.randint(1, 6) for _ in range(num_dice))

    @classmethod
    def calc_score(cls, dice: Counter[int]) -> int:
        assert len(dice), "Can't score empty dice"

        # Check for a straight
        if dice == Game.STRAIGHT_DICE:
            return 1500

        # Check for triples and singles
        score = 0
        for die, count in dice.items():
            if count == 6:
                return 2 * Game.TRIPLE_VALUES[die]
            elif count >= 3:
                score += Game.TRIPLE_VALUES[die]

            # Check for any single pointers
            if die == 1:
                score += 100 * (count % 3)
            elif die == 5:
                score += 50 * (count % 3)
        return score

    def handle_action(self, action: ActionT) -> None:
        if action == "draw":
            self.handle_draw()
        elif action == "roll":
            self.handle_roll()
        elif action.startswith("hold"):
            self.handle_hold(action)  # type: ignore
        elif action == "stop":
            self.handle_stop()
        else:
            raise UnknownActionError(action)
        if action == "draw":
            self.history.append(f"{action}: {self.card}")
        elif action == "roll":
            dice = "".join(sorted(str(val) for die, count in self.dice.items() for val in [die] * count))
            self.history.append(f"{action}: {dice} ({len(dice)} dice)")
        else:
            self.history.append(action)

    def handle_draw(self) -> None:
        self.card = self.deck.draw()
        if self.card.category == "no_dice":
            self.handle_bust(history=self.card.category)

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

        # Compute the incremental turn score (to deal with triples same roll requirement)
        # TODO: Just store the holds incrementally, don't calc the score each time
        holds = Counter({die: int(count) for die, count in enumerate(hold_action.split("_")[-1], start=1)})
        self.turn_score += self.calc_score(holds)  # type:ignore

        if self.holds.total() == self.NUM_DICE:
            self.handle_fill()
        self.dice = Counter()  # Reset dice to indicate we're ready to roll again

    def handle_stop(self) -> None:
        self.banked_score += self.turn_score
        self.turn_scores.append(self.turn_score)
        self._start_new_turn()

    def handle_bust(self, history: str = "bust") -> None:
        self.history.append(history)
        self.turn_score = 0
        self._start_new_turn()

    def handle_fill(self) -> None:
        self.history.append("fill")
        if self.card and self.card.bonus:
            self.turn_score += self.card.bonus
        self._reset_dice_holds_and_card()

    @property
    def turn_is_over(self) -> bool:
        return self.stopped or self.busted

    @property
    def game_is_over(self) -> bool:
        return self.banked_score >= self.max_score
