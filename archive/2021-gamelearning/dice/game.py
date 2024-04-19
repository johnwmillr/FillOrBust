import random
from collections import Counter
from typing import Callable

from dice.cards import Card, Deck
from dice.q import UnknownActionError
from dice.types import ActionT, State
from dice.utils import count_dice


class Game:
    NUM_DICE = 6
    TRIPLE_VALUES = {1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}
    STRAIGHT_DICE = [1, 2, 3, 4, 5, 6]

    def __init__(self, max_score: int | None = None) -> None:
        self.turn_score = 0
        self.turn_scores: list[int] = []
        self.banked_score = 0
        self.num_turns = 0
        self.dice: tuple[int, ...] = ()
        self.holds: tuple[int, ...] = ()
        self.deck = Deck()
        self.card: Card | None = None
        self.discard: Counter[Card] = Counter()
        self.history: list[ActionT | str] = ["start"]
        self.max_score = max_score if max_score else 5_000
        self.state_before_action_cleanup = []
        self._start_new_turn()

    def _start_new_turn(self) -> None:
        self.history.append(f"NEW TURN, Score: {self.banked_score}")
        self.num_turns += 1
        self.turn_score = 0
        self._reset_dice_holds_and_card()

    def _reset_dice(self) -> None:
        self.dice = ()

    def _reset_dice_holds_and_card(self) -> None:
        self.dice = ()
        self.holds = ()
        if self.card and self.card.category == "no_dice":
            self.discard[self.card] += 1
        self.card = None

    @property
    def num_dice_remaining(self) -> int:
        return Game.NUM_DICE - sum(self.holds)

    @property
    def state(self) -> State:
        return State(
            card=self.card,
            turn_score=self.turn_score,
            dice=self.dice,
            holds=self.holds,
        )

    @property
    def stopped(self) -> bool:
        return self.banked_score > 0

    @property
    def busted(self) -> bool:
        def no_pointers_on_dice(dice: tuple[int, ...]) -> bool:
            counts = count_dice(dice)
            return (counts[0] + counts[4]) == 0 and max(counts) < 3

        return sum(self.dice) > 0 and no_pointers_on_dice(self.dice)

    def roll_dice(self, num_dice: int) -> tuple[int, ...]:
        return tuple(sorted((random.randint(1, 6) for _ in range(num_dice))))

    @classmethod
    def calc_score(cls, dice: tuple[int, ...]) -> int:
        assert len(dice), "Can't score empty dice"

        # Check for a straight
        if sorted(dice) == Game.STRAIGHT_DICE:
            return 1500

        # Check for triples and singles
        counts = count_dice(dice)
        score = 0
        for die, count in enumerate(counts, start=1):
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
            response = self.handle_draw()
        elif action == "roll":
            response = self.handle_roll()
        elif action.startswith("hold"):
            response = self.handle_hold(action)  # type: ignore
        elif action == "stop":
            response = self.handle_stop()
        else:
            raise UnknownActionError(action)

        # Record the state before cleaning up after the action
        self.state_before_action_cleanup.append(str(self.state))
        if action == "draw":
            self.history.append(f"{action}: {self.card}")
        elif action == "roll":
            dice = "".join(map(str, self.dice))
            self.history.append(f"{action}: {dice} ({len(dice)} dice)")
        else:
            self.history.append(action)

        if response:
            response()

    def handle_draw(self) -> Callable | None:
        self.card = self.deck.draw()
        if self.card.category == "no_dice":
            # TODO: No Dice cards never get recorded in state, fix this
            # Must record the state somehow before it gets reset.
            # Otherwise the Agent never sees the state that occurs
            # directly after the action.
            self.turn_score = 0
            return self._start_new_turn

    def handle_roll(self) -> Callable | None:
        self.dice = self.roll_dice(self.num_dice_remaining)
        if self.busted:
            return self.handle_bust()

    def handle_hold(self, hold_action: ActionT) -> Callable | None:
        assert hold_action is not None

        # Add the new held dice from this roll to the turn's held dice
        # Also remove the held dice from `self.dice`
        counts = tuple(map(int, hold_action.split("_")[-1]))
        dice = [
            die for num, count in enumerate(counts, start=1) for die in count * (num,)
        ]

        # Compute the incremental turn score (to deal with triples same roll requirement)
        # TODO: Just store the holds incrementally, don't calc the score each time
        self.turn_score += self.calc_score(dice)  # type:ignore
        if self.holds:
            self.holds = tuple(hold + count for hold, count in zip(self.holds, counts))
        else:
            self.holds = counts

        if sum(self.holds) == self.NUM_DICE:
            return self.handle_fill()
        return self._reset_dice  # Reset dice to indicate we're ready to roll again

    def handle_stop(self) -> Callable:
        self.banked_score += self.turn_score
        self.turn_scores.append(self.turn_score)
        return self._start_new_turn

    def handle_bust(self, history: str = "bust") -> Callable:
        self.history.append(history)
        self.turn_score = 0
        return self._start_new_turn

    def handle_fill(self) -> Callable:
        self.history.append("fill")
        if self.card and self.card.bonus:
            self.turn_score += self.card.bonus
        return self._reset_dice_holds_and_card

    @property
    def turn_is_over(self) -> bool:
        return self.stopped or self.busted

    @property
    def game_is_over(self) -> bool:
        return self.banked_score >= self.max_score
