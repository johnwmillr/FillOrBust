from typing import Dict, Union


class Card:
    def __init__(self, name: str, bonus: int, must_fill: bool) -> None:
        self.name = name
        self.bonus = bonus
        self.must_fill = must_fill

    def to_dict(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "name": self.name,
            "bonus": self.bonus,
            "must_fill": self.must_fill,
        }

    def __str__(self) -> str:
        return f"<{self.name}>"

    def __repr__(self) -> str:
        return f"<card.Card: {self.name}>"


# Create the cards
NUM_GARDEN_VARIETY = 8
BONUS_300 = [Card("Bonus 300", 300, False) for _ in range(NUM_GARDEN_VARIETY)]
BONUS_400 = [Card("Bonus 400", 400, False) for _ in range(NUM_GARDEN_VARIETY)]
BONUS_500 = [Card("Bonus 500", 500, False) for _ in range(NUM_GARDEN_VARIETY)]

NUM_NO_DICE = 8
NO_DICE = [Card("No Dice", 0, False) for _ in range(NUM_NO_DICE)]

NUM_FILL_1000 = 4
FILL_1000 = [Card("Fill 1000", 1000, True) for _ in range(NUM_FILL_1000)]

NUM_MUST_BUST = 3
MUST_BUST = [Card("Must Bust", 0, False) for _ in range(NUM_MUST_BUST)]

NUM_DOUBLE_TROUBLE = 2
DOUBLE_TROUBLE = [Card("Double Trouble", 0, True) for _ in range(NUM_DOUBLE_TROUBLE)]

NUM_VENGEANCE = 3
VENGEANCE = [Card("Vengeance", 0, True) for _ in range(NUM_VENGEANCE)]

# Assemble the deck
FULL_DECK = BONUS_300 + BONUS_400 + BONUS_500 + NO_DICE + FILL_1000 + MUST_BUST + DOUBLE_TROUBLE + VENGEANCE
