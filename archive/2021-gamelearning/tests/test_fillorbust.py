import unittest

from fillorbust.game import Card, Deck, Dice, FillOrBust, Player


class TestDice(unittest.TestCase):
    NUM_SIDES = 6

    def test_new_dice_are_valid(self) -> None:
        num_dice = 6
        dice = Dice(num_dice=num_dice, num_sides=6)
        self.assertEqual(len(dice), num_dice)

        for die in dice.active:
            self.assertIn(die, range(1, num_dice + 1))

    def test_hold_raises_value_error_on_missing_number(self) -> None:
        num_dice = 6
        dice = Dice(num_dice=num_dice, num_sides=6)
        dice.active = [1]
        with self.assertRaises(ValueError):
            dice.hold_dice([5])

    def test_hold_removes_dice(self) -> None:
        num_dice = 6
        dice = Dice(num_dice=num_dice, num_sides=6)
        dice.active = [1, 2, 2, 2, 2, 2]
        dice.hold_dice([1])
        self.assertEqual(dice.held, [[1]])
        self.assertEqual(dice.active, [2, 2, 2, 2, 2])

    def test_straight_is_computed_correctly(self) -> None:
        num_dice = 6
        dice = Dice(num_dice=num_dice, num_sides=6)
        dice.active = []
        dice.held = [list(range(1, num_dice + 1))]
        self.assertEqual(dice.compute_score(dice.held[0]), 1500)

    def test_triple_is_computed_correctly(self) -> None:
        num_dice = 6
        dice = Dice(num_dice=num_dice, num_sides=6)
        dice.active = []
        dice.held = [[2, 2, 2]]
        self.assertEqual(dice.compute_score(dice.held[0]), 200)

    def test_singles_are_computed_correctly(self) -> None:
        num_dice = 6
        dice = Dice(num_dice=num_dice, num_sides=6)
        dice.active = []
        dice.held = [[1, 5]]
        self.assertEqual(dice.compute_score(dice.held[0]), 150)

    def test_triple_single_combos_are_computed_correctly(self) -> None:
        num_dice = 6
        dice = Dice(num_dice=num_dice, num_sides=6)
        dice.active = []

        holds = [[[1, 1, 1, 1]], [[2, 2, 2, 5]], [[2, 2, 2, 3, 3, 3]], [[1, 1, 1, 1, 1, 1]]]
        expected = [1100, 250]
        for held, score in zip(holds, expected):
            dice.held = held
            self.assertEqual(dice.compute_score(dice.held[0]), score)

    def test_compute_turn_score(self) -> None:
        test_cases = [
            {"held_dice": [[]], "expected": 0},
            {
                "held_dice": [
                    [5],
                    [5, 5],
                ],
                "expected": 150,
            },
            {
                "held_dice": [
                    [1],
                    [1, 1, 1],
                ],
                "expected": 1100,
            },
        ]

        dice = Dice()
        for case in test_cases:
            dice.held = case["held_dice"]
            score = dice.compute_score_from_held_dice()
            self.assertEqual(score, case["expected"])


class TestDeck(unittest.TestCase):
    def test_draw(self) -> None:
        deck = Deck()
        num_cards_in_deck_at_start = deck.num_cards_remaining
        self.assertIsInstance(deck.draw(), Card)
        self.assertEqual(deck.num_cards_remaining, num_cards_in_deck_at_start - 1)

    def test_shuffle(self) -> None:
        deck = Deck()
        num_cards_in_deck_at_start = deck.num_cards_remaining
        n = 0
        while deck.num_cards_remaining > 0:
            deck.draw()
            n += 1
            self.assertEqual(n, len(deck.discard))
        deck.shuffle()
        self.assertEqual(deck.num_cards_remaining, num_cards_in_deck_at_start)
        self.test_draw()


# class TestGame(unittest.TestCase):


