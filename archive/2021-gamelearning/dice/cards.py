from random import shuffle
from typing import Literal


GardenVarietyBonusT = Literal[300, 400, 500]


class Card:

    def __init__(self, title: str, category: str, bonus: int, must_fill: bool) -> None:
        self.title = title
        self.category = category
        self.bonus = bonus
        self.must_fill = must_fill

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            raise NotImplementedError
        return self.title == other.title

    def __hash__(self) -> int:
        return hash(self.title)

    def __lt__(self, other: "Card") -> bool:
        return self.title < other.title

    def __gt__(self, other: "Card") -> bool:
        return self.title > other.title

    def __str__(self) -> str:
        string = self.category
        if self.category == "garden_variety":
            string += f"_{self.bonus}"
        if self.must_fill:
            string += "_must-fill"
        return string

    def __repr__(self) -> str:
        return f"<Card: {self.title}>"


def make_a_no_dice_card() -> Card:
    return Card(title="No Dice!", category="no_dice", bonus=0, must_fill=False)


def make_a_fill_1000() -> Card:
    return Card(title="Fill 1,000!", category="fill_1000", bonus=1000, must_fill=True)


def make_a_bonus_card(bonus: GardenVarietyBonusT) -> Card:
    return Card(
        title=f"Bonus {bonus}", category="garden_variety", bonus=bonus, must_fill=False
    )


def make_a_must_bust_card() -> Card:
    return Card(title="Must Bust!", category="must_bust", bonus=0, must_fill=False)


class EmptyDeckException(Exception):
    pass


class Deck:
    NUM_NO_DICE = 8
    NUM_FILL_1000 = 6
    NUM_BONUS_300 = 12
    NUM_BONUS_400 = 10
    NUM_BONUS_500 = 8
    NUM_MUST_BUST = 3

    def __init__(self) -> None:
        self._shuffle_new_deck()

    def _shuffle_new_deck(self) -> None:
        self.cards: list[Card] = []
        self.cards.extend(make_a_no_dice_card() for _ in range(self.NUM_NO_DICE))
        self.cards.extend(make_a_fill_1000() for _ in range(self.NUM_FILL_1000))
        self.cards.extend(make_a_bonus_card(300) for _ in range(self.NUM_BONUS_300))
        self.cards.extend(make_a_bonus_card(400) for _ in range(self.NUM_BONUS_400))
        self.cards.extend(make_a_bonus_card(500) for _ in range(self.NUM_BONUS_500))
        self.cards.extend(make_a_must_bust_card() for _ in range(self.NUM_MUST_BUST))
        self.shuffle()

    def shuffle(self) -> None:
        if not self.cards:
            raise EmptyDeckException("Can't shuffle an empty deck.")
        shuffle(self.cards)

    def draw(self) -> Card:
        if not self.cards:
            self._shuffle_new_deck()
        return self.cards.pop()

    def __len__(self) -> int:
        return len(self.cards)
