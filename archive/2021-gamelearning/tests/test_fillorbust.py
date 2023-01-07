import unittest

from fillorbust.game import Card, Deck, Dice


class TestDice(unittest.TestCase):
    NUM_SIDES = 6

    def test_new_dice_are_valid(self) -> None:
        num_dice, num_sides = 6, 6
        dice = Dice(num_dice=num_dice, num_sides=num_sides)
        self.assertIsNone(dice.active)
        self.assertEqual(len(dice), num_dice)

        dice.roll()
        for die in dice.active:  # type: ignore
            self.assertIn(die, range(1, num_dice + 1))

    def test_num_dice_remaining(self) -> None:
        dice = Dice()
        self.assertEqual(dice.num_dice_remaining, dice.num_dice)
        dice.active = [1, 2, 3, 4]
        dice.held = [[1], [5]]
        self.assertEqual(dice.num_dice_remaining, 4)

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
        dice = Dice()
        dice.active = []
        dice.held = [list(range(1, dice.num_dice + 1))]
        self.assertEqual(dice.compute_score(dice.held[0]), 1500)

    def test_triple_is_computed_correctly(self) -> None:
        dice = Dice()
        dice.active = []
        dice.held = [[2, 2, 2]]
        self.assertEqual(dice.compute_score(dice.held[0]), 200)

    def test_singles_are_computed_correctly(self) -> None:
        dice = Dice()
        dice.active = []
        holds = [[1], [1, 5], [1, 1], [5]]
        scores = [100, 150, 200, 50]
        for held, score in zip(holds, scores):
            dice.held = [held]
            self.assertEqual(dice.compute_score(dice.held[0]), score)

    def test_triple_single_combos_are_computed_correctly(self) -> None:
        dice = Dice()
        dice.active = []

        holds = [[[1, 1, 1, 1]], [[2, 2, 2, 5]], [[2, 2, 2, 3, 3, 3]], [[1, 1, 1, 1, 1, 1]]]
        scores = [1100, 250, 500, 2000]
        for held, score in zip(holds, scores):
            dice.held = held
            self.assertEqual(dice.compute_score(dice.held[0]), score)

    def test_compute_turn_score(self) -> None:
        cases = [
            {"held": [[]], "expected": 0},
            {
                "held": [
                    [5],
                    [5, 5],
                ],
                "expected": 150,
            },
            {
                "held": [
                    [1],
                    [1, 1, 1],
                ],
                "expected": 1100,
            },
            {
                "held": [
                    [1, 2, 3, 4, 5, 6],
                ],
                "expected": 1500,
            },
        ]

        dice = Dice()
        for case in cases:
            dice.held = case["held"]
            score = dice.compute_score_from_held_dice()
            self.assertEqual(score, case["expected"])

    def test_get_possible_actions_from_dice(self) -> None:
        dice = Dice()
        cases = [
            {
                "active": [1, 1, 1, 1, 1, 1],
                "expected": [
                    "take_1,1,1,1,1,1",
                ],
            },
            {
                "active": [1, 1, 6, 4, 3, 3],
                "expected": [
                    "take_1",
                    "take_1,1",
                ],
            },
            {
                "active": [1, 5, 6, 4, 3, 3],
                "expected": [
                    "take_1",
                    "take_1,5",
                    "take_5",
                ],
            },
            {
                "active": [1, 2, 2, 3, 5],
                "expected": [
                    "take_1",
                    "take_1,5",
                    "take_5",
                ],
            },
            {
                "active": [1, 2, 2, 2, 5],
                "expected": [
                    "take_1",
                    "take_1,2,2,2",
                    "take_1,2,2,2,5",
                    "take_1,5",
                    "take_2,2,2",
                    "take_2,2,2,5",
                    "take_5",
                ],
            },
            {
                "active": [1, 2, 3, 4, 5, 6],
                "expected": [
                    "take_1",
                    "take_1,2,3,4,5,6",
                    "take_1,5",
                    "take_5",
                ],
            },
            {
                "active": [1],
                "expected": [
                    "take_1",
                ],
            },
            {
                "active": [2, 2, 2, 3, 3, 3],
                "expected": [
                    "take_2,2,2",
                    "take_2,2,2,3,3,3",
                    "take_3,3,3",
                ],
            },
            {
                "active": [1, 1, 1, 1, 3, 3],
                "expected": [
                    "take_1",
                    "take_1,1,1",
                    "take_1,1,1,1",
                ],
            },
            {
                "active": [],
                "expected": [],
            },
        ]
        for case in cases:
            self.assertEqual(dice.get_possible_actions(case["active"]), sorted(case["expected"]))


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
