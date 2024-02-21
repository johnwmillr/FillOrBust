from random import shuffle


class Card:

    def __init__(
        self, title: str, category: str, bonus: int | None, must_fill: bool
    ) -> None:
        self.title = title
        self.category = category
        self.bonus = bonus
        self.must_fill = must_fill

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return f"<Card: {self.title}>"


def make_a_no_dice_card() -> Card:
    return Card(title="No Dice!", category="no_dice", bonus=None, must_fill=False)


def make_a_fill_1000() -> Card:
    return Card(title="Fill 1,000!", category="fill_1000", bonus=1000, must_fill=True)


def make_a_bonus_card(bonus: int) -> Card:
    assert bonus in [300, 400, 500]
    return Card(
        title=f"Bonus {bonus}", category="garden_variety", bonus=bonus, must_fill=False
    )


class EmptyDeckException(Exception):
    pass


class Deck:
    def __init__(self) -> None:
        self._shuffle_new_deck()

    def _shuffle_new_deck(self) -> None:
        self.cards: list[Card] = []
        self.cards.extend(make_a_no_dice_card() for _ in range(8))
        self.cards.extend(make_a_fill_1000() for _ in range(6))
        self.cards.extend(make_a_bonus_card(300) for _ in range(12))
        self.cards.extend(make_a_bonus_card(400) for _ in range(10))
        self.cards.extend(make_a_bonus_card(500) for _ in range(8))
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
