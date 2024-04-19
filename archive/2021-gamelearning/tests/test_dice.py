import unittest

from dice.agents import Agent
from dice.cards import make_a_bonus_card, make_a_fill_1000, make_a_no_dice_card
from dice.game import Game
from dice.q import QTable
from dice.types import ActionT


class TestDeterminingHolds(unittest.TestCase):
    agent = Agent("tester")

    def _test_dice_holding(self, dice: list[int], expected: tuple[str, ...]) -> None:
        self.assertEqual(
            self.agent.determine_hold_options_from_dice(dice), sorted(expected)
        )

    def test_rolled_a_straight(self) -> None:
        self._test_dice_holding(list(range(1, Game.NUM_DICE + 1)), ("hold_111111",))

    def test_rolled_a_one(self) -> None:
        self._test_dice_holding([1, 2, 2, 3, 3, 4], ("hold_100000",))

    def test_rolled_a_five(self) -> None:
        self._test_dice_holding([5], ("hold_000010",))

    def test_rolled_four_ones(self) -> None:
        self._test_dice_holding(
            [1, 1, 1, 1, 4],
            (
                "hold_100000",
                "hold_200000",
                "hold_300000",
                "hold_400000",
            ),
        )

    def test_rolled_a_one_and_two_fives(self) -> None:
        self._test_dice_holding(
            [1, 2, 2, 3, 5, 5],
            (
                "hold_100000",
                "hold_100010",
                "hold_100020",
                "hold_000020",
                "hold_000010",
            ),
        )

    def test_rolled_a_triple_two_and_a_five(self) -> None:
        self._test_dice_holding(
            [2, 2, 2, 3, 5],
            (
                "hold_030000",
                "hold_030010",
                "hold_000010",
            ),
        )

    def test_rolled_two_triple_threes(self) -> None:
        self._test_dice_holding(
            [3, 3, 3, 3, 3, 3],
            (
                "hold_003000",
                "hold_006000",
            ),
        )

    def test_rolled_a_bust(self) -> None:
        self._test_dice_holding(
            [2, 2, 3, 3],
            (),
        )

    def test_rolled_two_ones_and_a_triple_six(self) -> None:
        self._test_dice_holding(
            [1, 1, 2, 6, 6, 6],
            (
                "hold_100000",
                "hold_200000",
                "hold_000003",
                "hold_100003",
                "hold_200003",
            ),
        )

    def test_rolled_a_triple_double_bust(self) -> None:
        self._test_dice_holding([2, 2, 3, 3, 4, 4], ())

    def test_one_one_and_five_threes(self) -> None:
        self._test_dice_holding(
            [1, 3, 3, 3, 3, 3],
            (
                "hold_100000",
                "hold_003000",
                "hold_103000",
            ),
        )


class TestStateIsImmutable(unittest.TestCase):
    # TODO: Test that the state is immutable and the agent will learn
    def test_previous_state_does_not_change_after_action(self) -> None:
        game = Game()
        previous_state = game.state
        game.handle_action("draw")
        self.assertNotEqual(previous_state, game.state)

    def test_that_state_is_a_valid_immutable_dict_key(self) -> None:
        game = Game()
        Q = QTable()

        previous_state = game.state
        game.handle_action("draw")
        Q.update(
            previous_state,
            "draw",
            game.state,
            0.0,
        )
        self.assertTrue(Q.state_in_table(previous_state))
        game.dice = (1, 2, 3)
        self.assertTrue(Q.state_in_table(previous_state))
        # self.assertTrue(Q.state_in_table(game.state))


class TestDeterminingAvailableActions(unittest.TestCase):
    agent = Agent("tester")

    def _test_available_actions(self, game: Game, expected: list[ActionT]) -> None:
        self.assertEqual(
            self.agent.determine_available_actions(game.state), sorted(expected)
        )

    def test_fresh_start_enables_draw(self) -> None:
        game = Game()
        self._test_available_actions(game, ["draw"])

    def test_drawing_garden_variety_enables_roll(self) -> None:
        game = Game()
        game.card = make_a_bonus_card(300)
        self._test_available_actions(game, ["roll"])

    def test_rolling_pointers_enables_holds(self) -> None:
        game = Game()
        game.card = make_a_bonus_card(300)
        game.dice = [1, 1, 2, 2, 5]
        self._test_available_actions(
            game,
            [
                "hold_100000",
                "hold_200000",
                "hold_000010",
                "hold_100010",
                "hold_200010",
            ],  # type:ignore
        )

    def test_holding_dice_enables_roll_or_stop(self) -> None:
        game = Game()
        game.card = make_a_bonus_card(300)
        game.dice = [1, 1, 2, 2, 5]
        game.handle_action("hold_200000")  # type:ignore
        self._test_available_actions(game, ["roll", "stop"])
        self.assertEqual(game.turn_score, 200)

    def test_filling_enables_draw_or_stop(self) -> None:
        game = Game()
        game.card = make_a_bonus_card(300)
        game.handle_action("hold_111111")  # type:ignore
        self._test_available_actions(game, ["draw", "stop"])
        self.assertEqual(game.turn_score, 1800)
        self.assertEqual(game.banked_score, 0)

    def test_fill_1000_prevents_stopping_until_filled(self) -> None:
        game = Game()
        game.card = make_a_fill_1000()
        game.handle_action("hold_100000")  # type:ignore
        self._test_available_actions(game, ["roll"])
        game.handle_action("hold_000050")  # type:ignore
        self._test_available_actions(game, ["draw", "stop"])


class TestCardRequirements(unittest.TestCase):
    agent = Agent("tester")

    def test_garden_variety_bonus_gets_applied_to_fill(self) -> None:
        game = Game()
        game.card = make_a_bonus_card(300)
        game.dice = [1, 1, 2, 2, 2, 5]
        game.handle_action("hold_230010")  # type:ignore
        game.handle_action("stop")
        self.assertEqual(game.banked_score, 750)

    def test_garden_variety_allows_stopping(self) -> None:
        game = Game()
        game.card = make_a_bonus_card(300)
        game.dice = [1, 1, 2, 2, 3, 5]
        game.handle_action("hold_100000")  # type:ignore
        available_actions = sorted(self.agent.determine_available_actions(game.state))
        expected_actions = ["roll", "stop"]
        self.assertEqual(available_actions, expected_actions)
        game.handle_action("stop")
        self.assertEqual(game.banked_score, 100)

    def test_fill_1000_requires_fill(self) -> None:
        game = Game()
        game.card = make_a_fill_1000()
        game.dice = [1, 1, 2, 2, 3, 5]
        game.handle_action("hold_100000")  # type:ignore
        available_actions = self.agent.determine_available_actions(game.state)
        expected_actions = ["roll"]
        self.assertEqual(available_actions, expected_actions)

    def test_no_dice_ends_turn(self) -> None:
        game = Game()
        game.card = make_a_bonus_card(300)
        game.dice = [1, 1, 2, 2, 2, 5]
        game.handle_action("hold_230010")  # type:ignore
        game.deck.cards = [make_a_no_dice_card()]
        game.handle_action("draw")
        self.assertEqual(game.history[-2], "draw: no_dice")
        self.assertEqual(game.banked_score, 0)


class TestGameScore(unittest.TestCase):

    def test_same_roll_requirement_for_triples(self) -> None:
        game = Game()
        game.dice = [1, 1, 2, 2, 3, 3]
        game.handle_action("hold_200000")  # type:ignore
        game.dice = [1, 2, 2, 3]
        game.handle_action("hold_100000")  # type:ignore
        game.handle_action("stop")
        self.assertEqual(game.banked_score, 300)


class TestCalcScore(unittest.TestCase):
    calc_score = Game.calc_score

    def _test_dice_scoring(self, dice: list[int], expected: int) -> None:
        self.assertEqual(self.calc_score(dice), expected)

    def test_straight(self) -> None:
        self._test_dice_scoring(
            [1, 2, 3, 4, 5, 6],
            1500,
        )

    def test_triple_two(self) -> None:
        self._test_dice_scoring([2, 2, 2, 3, 4, 6], 200)

    def test_double_triple_two(self) -> None:
        self._test_dice_scoring([2, 2, 2, 2, 2, 2], 400)

    def test_triple_three_and_triple_six(self) -> None:
        self._test_dice_scoring([3, 3, 3, 6, 6, 6], 900)

    def test_single_one(self) -> None:
        self._test_dice_scoring([1, 2, 2, 3, 4, 4], 100)

    def test_double_ones(self) -> None:
        self._test_dice_scoring([1, 1, 2, 3, 4, 4], 200)

    def test_single_five(self) -> None:
        self._test_dice_scoring([5, 2, 2, 3, 4, 4], 50)

    def test_double_fives(self) -> None:
        self._test_dice_scoring([5, 5, 2, 3, 4, 4], 100)

    def test_four_fives(self) -> None:
        self._test_dice_scoring([5, 5, 5, 5, 4, 4], 550)

    def test_four_ones(self) -> None:
        self._test_dice_scoring([1, 1, 1, 1], 1100)

    def test_five_sixes(self) -> None:
        self._test_dice_scoring([6, 6, 6, 6, 6, 4], 600)

    def test_no_scoring(self) -> None:
        self._test_dice_scoring([2, 3, 4, 4, 6, 6], 0)

    def test_122445(self) -> None:
        self._test_dice_scoring([1, 2, 2, 4, 4, 5], 150)

    def test_11344(self) -> None:
        self._test_dice_scoring([1, 1, 3, 4, 4], 200)

    def test_empty_input(self) -> None:
        self.assertRaises(
            AssertionError,
            self.calc_score,
            [],
        )
