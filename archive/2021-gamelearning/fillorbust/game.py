from collections import Counter
from inspect import Attribute
from uuid import uuid4
from random import choice, sample, shuffle
from typing import List


try:
    from fillorbust.card import Card, FULL_DECK
except ModuleNotFoundError:
    from card import Card, FULL_DECK


class Dice:
    TRIPLE_VALUES = {1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}
    SINGLE_VALUES = {1: 100, 5: 50}

    def __init__(self, num_dice: int = 6, num_sides: int = 6) -> None:
        self.num_dice = num_dice
        self.num_sides = num_sides
        self.active = self.roll()
        self.held: List[List[int]] = []

    def roll(self) -> List[int]:
        dice = sample(range(1, self.num_sides + 1), self.num_dice_remaining)
        return dice

    @property
    def active_counts(self) -> Counter:
        return Counter(self.active)

    def hold_dice(self, numbers_to_hold: List[int]) -> None:
        counts = Counter(numbers_to_hold)
        for number in numbers_to_hold:
            if counts[number] > self.active_counts.get(number, 0):
                raise ValueError(f"Can't remove {counts[number]} {number}'s from {self.active}.")

        self.held.append([self.active.pop(self.active.index(number)) for number in numbers_to_hold])

    def compute_score(self, dice: List[int]) -> int:
        counts = Counter(dice)

        # Check for a straight
        if len(dice) == self.num_dice and set(counts.values()) == {1}:
            return 1500

        # Check for triples and singles
        score = 0
        for die, count in counts.items():
            if count >= 3:
                score += Dice.TRIPLE_VALUES[die]
            if count != 3:
                score += Dice.SINGLE_VALUES.get(die, 0) * (count % 3)
        return score

    def compute_score_from_held_dice(self) -> int:
        return sum(self.compute_score(dice) for dice in self.held)

    @property
    def num_dice_remaining(self) -> int:
        try:
            return self.num_dice - len(self.held)
        except AttributeError:
            return self.num_dice

    def __len__(self) -> int:
        return self.num_dice_remaining


class Deck:
    def __init__(self) -> None:
        self.cards: List[Card] = FULL_DECK.copy()
        self.discard: List[Card] = []
        self.shuffle()

    def draw(self) -> Card:
        if self.num_cards_remaining:
            self.discard.append(self.cards.pop())
            return self.discard[-1]
        self.shuffle()
        return self.draw()

    def shuffle(self) -> None:
        if self.discard:
            self.cards = self.discard
        shuffle(self.cards)
        self.discard = []

    @property
    def num_cards_remaining(self) -> int:
        return len(self.cards)


class Player:
    def __init__(self, name) -> None:
        self.name = name
        self.score = 0
        self.held_dice = []
        self.id = uuid4()

    def choose_action(self, actions: List[str]) -> str:
        return choice(actions)

    def __repr__(self) -> str:
        return f"<Player: {self.name}, ({self.score})>"


class FillOrBust:
    def __init__(self, players: List[Player], max_score: int = 10_000) -> None:
        assert len(players) > 1, "Must supply at least two players."
        shuffle(players)
        self.players = players
        self._current_player_idx = 0
        self.max_score = max_score
        self.deck = Deck()
        self.dice = Dice()
        self.turn_is_in_progress = False
        self.busted = False

    def play(self) -> None:
        while not self.is_over:
            self.take_turn_for_current_player()

    @property
    def current_player(self) -> Player:
        return self.players[self._current_player_idx]

    def take_turn_for_current_player(self) -> None:
        self.turn_is_in_progress = True
        while self.turn_is_in_progress:
            actions = self.get_available_actions()
            action = self.current_player.choose_action(actions)
            self.perform_action(action)
            self.check_action_result()
        self._end_turn()

    def get_available_actions(self) -> List[str]:
        actions = []
        if self.dice.num_dice_remaining == 6:
            actions.extend(["roll", "draw"])

        actions.extend(self.get_dice_actions())
        self.dice.dice
        return actions

    def get_dice_actions(self) -> List[str]:
        actions: List[str] = []
        counts = Counter(self.dice.dice)

        # Check for a straight
        if len(self.dice.dice) == Dice.NUM_DICE and set(counts.values()) == {1}:
            actions.append("123456")

        # Check for triples and singles
        for key, count in counts.items():
            if count == 3:
                Dice.TRIPLE_VALUES[key.value]  # type: ignore
                actions.append(f"{str(key.value) * count}")
            elif key.value in Dice.SINGLE_VALUES:
                actions.append(f"{str(key.value) * count}")
        return [f"take_pointers_{action}" for action in actions]

    def perform_action(self, action: str) -> None:
        if action == "draw":
            self.draw()
        elif action == "roll":
            self.roll()
        elif action.startswith("take_pointers_"):
            self.take_pointers(action)
        elif action == "stop":
            self.stop()

    def draw(self) -> None:
        card = self.deck.draw()

    def roll(self) -> None:
        self.dice.roll()

    def take_pointers(self, pointers: str) -> None:
        pass

    def stop(self) -> None:
        self.turn_is_in_progress = False

    def check_action_result(self) -> None:
        if self.dice.compute_score(self.dice.held) == 0:
            self.busted = True
            self.turn_is_in_progress = False

    def _end_turn(self) -> None:
        self.compute_turn_score_for_current_player()
        self._increment_player_index()

    def compute_turn_score_for_current_player(self) -> None:
        score = 0
        if not self.busted:
            score = self.dice.compute_score_from_held_dice()
        self.current_player.score += score

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def num_dice_remaining(self) -> int:
        return len(self.dice)

    @property
    def is_over(self) -> bool:
        over_max_score = any(player.score >= self.max_score for player in self.players)
        no_tie = len(set(player.score for player in self.players)) > 1
        is_over = over_max_score and no_tie
        return is_over

    def _increment_player_index(self) -> None:
        self._current_player_idx = (self._current_player_idx + 1) % self.num_players

    def _dice_to_display_str(self, dice: Dice) -> str:
        return str(dice)

    def __str__(self):
        return self._dice_to_display_str(self.dice)

    def __repr__(self) -> str:
        return f"FillOrBust: {self.__str__()}"


def main() -> None:
    john = Player("John")
    anna = Player("Anna")
    game = FillOrBust(players=[john, anna])
    while not game.is_over:
        game.play()


if __name__ == "__main__":
    main()
