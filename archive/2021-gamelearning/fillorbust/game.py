from collections import Counter
from itertools import combinations
from random import choice, choices, shuffle
from typing import List, Optional
from uuid import uuid4

try:
    from fillorbust.card import FULL_DECK, Card
except ModuleNotFoundError:
    from card import FULL_DECK, Card


class Dice:
    TRIPLE_VALUES = {1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}
    SINGLE_VALUES = {1: 100, 5: 50}

    def __init__(self, num_dice: int = 6, num_sides: int = 6) -> None:
        self.num_dice = num_dice
        self.num_sides = num_sides
        self.active: Optional[List[int]] = None
        self.held: List[List[int]] = []

    def roll(self) -> None:
        self.active = choices(range(1, self.num_sides + 1), k=self.num_dice_remaining)

    @property
    def active_counts(self) -> Counter:
        return Counter(self.active)

    def hold_dice(self, numbers_to_hold: List[int]) -> None:
        counts = Counter(numbers_to_hold)
        for number in numbers_to_hold:
            if counts[number] > self.active_counts.get(number, 0):
                raise ValueError(f"Can't remove {counts[number]} {number}'s from {self.active}.")

        self.held.append([self.active.pop(self.active.index(number)) for number in numbers_to_hold])  # type: ignore

    def get_possible_actions(self, dice: List[int]) -> List[str]:
        counts = Counter(dice)
        options: List[str] = []

        # Check for a straight
        if dice and len(dice) == self.num_dice and set(counts.values()) == {1}:
            options.append("123456")

        # Check for triples and singles
        for die, count in counts.items():
            if count >= 3:
                if count == self.num_dice:
                    options.append(6 * str(die))
                else:
                    options.append(3 * str(die))
            if count != 3 and die in Dice.SINGLE_VALUES:
                for n in range(1, (count % 3) + 1):
                    options.append(str(die))
        options = [",".join(option) for option in options]

        # Enumerate all possible combinations
        combos = set()
        for n in range(1, len(options) + 1):
            combos = combos.union(set(",".join(sorted(combo)) for combo in combinations(options, n)))

        # Prevent pulling out more dice than allowed (e.g. ['1,1,2,3,4,5,6'])
        combos = filter(lambda x: len(x.replace(",", "")) <= len(dice), combos)
        return sorted("take_" + option for option in combos)

    def compute_score(self, dice: List[int]) -> int:
        counts = Counter(dice)

        # Check for a straight
        if len(dice) == self.num_dice and set(counts.values()) == {1}:
            return 1500

        # Check for triples and singles
        score = 0
        for die, count in counts.items():
            if count >= 3:
                if count == self.num_dice:
                    score += 2 * Dice.TRIPLE_VALUES[die]
                else:
                    score += Dice.TRIPLE_VALUES[die]
            if count != 3:
                score += Dice.SINGLE_VALUES.get(die, 0) * (count % 3)
        return score

    def compute_score_from_held_dice(self) -> int:
        return sum(self.compute_score(dice) for dice in self.held)

    @property
    def num_dice_remaining(self) -> int:
        return len(self.active) if self.active else self.num_dice

    def reset(self) -> None:
        self.active = None
        self.held = []

    def __len__(self) -> int:
        return self.num_dice_remaining

    def __repr__(self) -> str:
        return f"<Dice: {self.active}>"


class Deck:
    def __init__(self) -> None:
        self.cards: List[Card] = FULL_DECK.copy()
        self.discard: List[Card] = []
        self.current: Optional[Card] = None
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
        self.wins = 0
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
        self.state = "new_turn"

    def play(self) -> None:
        while not self.is_over:
            self.take_turn_for_current_player()

    @property
    def current_player(self) -> Player:
        return self.players[self._current_player_idx]

    def take_turn_for_current_player(self) -> None:
        self.turn_is_in_progress = True
        self.state = "new_turn"
        while self.turn_is_in_progress:
            actions = self.get_available_actions()
            if actions:
                action = self.current_player.choose_action(actions)
                self.perform_action(action)
            else:
                self.turn_is_in_progress = False
        self._end_turn()

    def get_available_actions(self) -> List[str]:
        actions = []
        if self.state == "new_turn":
            actions.append("draw")
        if self.state in ["drew_card", "took_pointers"]:
            actions.append("roll")
        if self.state == "took_pointers":
            actions.append("stop")
        if self.state == "rolled":
            actions.extend(self.dice.get_possible_actions(self.dice.active))  # type: ignore
        if self.state == "filled":
            actions.extend(["draw", "stop"])
        if self.state == "busted":
            actions = []
        return actions

    def perform_action(self, action: str) -> None:
        if action == "draw":
            self.draw()
        elif action == "roll":
            self.roll()
        elif action.startswith("take_"):
            self.take_pointers(action)
        elif action == "stop":
            self.stop()

    def draw(self) -> None:
        self.deck.current = self.deck.draw()
        if self.deck.current.name == "No Dice":
            self.state = "busted"
        else:
            self.state = "drew_card"

    def roll(self) -> None:
        self.dice.roll()
        if self.dice.compute_score(self.dice.active):  # type: ignore
            self.state = "rolled"
        else:
            self.state = "busted"

    def take_pointers(self, action: str) -> None:
        pointers = list(map(int, action.split("take_")[1].split(",")))
        self.dice.hold_dice(pointers)
        self.state = "took_pointers"

    def stop(self) -> None:
        self.turn_is_in_progress = False

    def _end_turn(self) -> None:
        self.compute_turn_score_for_current_player()
        self._increment_player_index()
        self.dice.reset()

    def compute_turn_score_for_current_player(self) -> None:
        score = 0
        if not self.state == "busted":
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

    for n in range(1000):
        game = FillOrBust(players=[john, anna])
        while not game.is_over:
            game.play()
        if n % 100 == 0:
            print(n)
        if john.score > anna.score:
            john.wins += 1
        else:
            anna.wins += 1
        john.score = 0
        anna.score = 0
    print(f"John wins: {john.wins}")
    print(f"Anna wins: {anna.wins}")


if __name__ == "__main__":
    main()
