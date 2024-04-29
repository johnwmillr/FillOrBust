import pytest

from dice.game import (
    DiceCounts,
    calc_score,
    FillOrBustEnvironment,
    determine_available_actions,
    determine_hold_options_from_dice,
    is_fill,
)
from dice.types import ActionT, Observation


# ------------------------------ #
# Test the `calc_score` function
# ------------------------------ #
@pytest.mark.parametrize(
    ["dice", "expected_score"],
    [
        (
            (1, 1, 1, 1, 1, 1),
            1500,
        ),  # Straight
        ((0, 3, 0, 0, 0, 0), 200),  # Three 2's
        ((0, 6, 0, 0, 0, 0), 400),  # Six 2's
        ((0, 0, 3, 0, 0, 3), 900),  # Three 3's, Three 6's
        ((1, 0, 0, 0, 0, 0), 100),  # One 1
        ((2, 0, 0, 0, 0, 0), 200),  # Two 1's
        ((0, 0, 0, 0, 1, 0), 50),  # One 5
        ((0, 0, 0, 0, 2, 0), 100),  # Two 5's
        ((0, 0, 0, 0, 4, 0), 550),  # Four 5's
        ((4, 0, 0, 0, 0, 0), 1100),  # Four 1's
        ((0, 0, 0, 0, 0, 5), 600),  # Five 6's
        ((0, 1, 1, 2, 0, 2), 0),  # Bust
        ((1, 2, 0, 2, 1, 0), 150),  # One 1, one 5
        ((2, 0, 1, 2, 0, 0), 200),  # Two 1's
    ],
)
def test_calc_score(dice: DiceCounts, expected_score: int) -> None:
    assert calc_score(dice) == expected_score


@pytest.fixture()
def game() -> FillOrBustEnvironment:
    return FillOrBustEnvironment()


@pytest.mark.parametrize("expected_score", [300])
def test_same_roll_requirement_for_triples(
    expected_score: int, game: FillOrBustEnvironment
) -> None:
    game.dice = DiceCounts((2, 2, 2, 0, 0, 0))
    game.step(("hold", DiceCounts((2, 0, 0, 0, 0, 0))))
    game.dice = DiceCounts((1, 2, 1, 0, 0, 0))
    game.step(("hold", DiceCounts((1, 0, 0, 0, 0, 0))))
    game.step(("stop", None))
    assert game.game_score == expected_score


# ---------------------------------------------------- #
# Test the `determine_hold_options_from_dice` function
# ---------------------------------------------------- #
@pytest.mark.parametrize(
    ["dice", "expected_holds"],
    [
        ((1, 1, 1, 1, 1, 1), [(1, 1, 1, 1, 1, 1)]),  # Straight
        ((1, 2, 2, 1, 0, 0), [(1, 0, 0, 0, 0, 0)]),  # One 1
        ((0, 0, 0, 0, 1, 0), [(0, 0, 0, 0, 1, 0)]),  # One 5
        (
            (4, 0, 0, 0, 0, 0),
            [
                (1, 0, 0, 0, 0, 0),
                (2, 0, 0, 0, 0, 0),
                (3, 0, 0, 0, 0, 0),
                (4, 0, 0, 0, 0, 0),
            ],
        ),  # Four 1's
        (
            (1, 0, 0, 0, 2, 0),
            [
                (1, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 1, 0),
                (1, 0, 0, 0, 2, 0),
                (0, 0, 0, 0, 2, 0),
                (0, 0, 0, 0, 1, 0),
            ],
        ),  # One 1, two 5's
        (
            (0, 3, 1, 0, 1, 0),
            [
                (0, 3, 0, 0, 0, 0),
                (0, 3, 0, 0, 1, 0),
                (0, 0, 0, 0, 1, 0),
            ],
        ),  # One triple 2, one 5
        (
            (0, 0, 6, 0, 0, 0),
            [
                (0, 0, 3, 0, 0, 0),
                (0, 0, 6, 0, 0, 0),
            ],
        ),  # Two triple 3's
        (
            (2, 0, 0, 0, 0, 3),
            [
                (1, 0, 0, 0, 0, 0),
                (2, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 3),
                (1, 0, 0, 0, 0, 3),
                (2, 0, 0, 0, 0, 3),
            ],
        ),  # Two 1's and a triple 6
        (
            (1, 0, 5, 0, 0, 0),
            [
                (1, 0, 0, 0, 0, 0),
                (0, 0, 3, 0, 0, 0),
                (1, 0, 3, 0, 0, 0),
            ],
        ),  # One 1, five 3's
        ((0, 2, 2, 0, 0, 0), []),  # Bust
        ((0, 2, 2, 2, 0, 0), []),  # Triple double bust
    ],
)
def test_determine_hold_options_from_dice(
    dice: DiceCounts, expected_holds: list[DiceCounts]
) -> None:
    actual = determine_hold_options_from_dice(dice)
    assert actual == sorted(expected_holds)


# -------------------------------------------------------------------------- #
# Test the `is_fill` function
# -------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ["hold", "expected"],
    [
        (DiceCounts((1, 1, 1, 1, 1, 1)), True),
        (DiceCounts((1, 1, 1, 1, 1, 0)), False),
        (DiceCounts((6, 0, 0, 0, 0, 0)), True),
        (DiceCounts((5, 0, 0, 0, 0, 0)), False),
        (DiceCounts((1, 0, 0, 0, 0, 0)), False),
        (DiceCounts((1, 2, 0, 0, 0, 0)), False),
    ],
)
def test_is_fill(hold: DiceCounts, expected: bool) -> None:
    assert is_fill(hold) == expected


# ---------------------------------- #
# Test `determine_available_actions`
# ---------------------------------- #
# These parameterized tests seem pretty hard to read
@pytest.mark.parametrize(
    ["obs", "expected"],
    [
        (
            Observation(0, DiceCounts((0, 0, 0, 0, 0, 0))),
            [("roll", None)],
        ),
        (
            Observation(0, DiceCounts((1, 0, 0, 0, 0, 0))),
            [("hold", DiceCounts((1, 0, 0, 0, 0, 0)))],
        ),
        (
            Observation(100, DiceCounts((0, 0, 0, 0, 0, 0))),
            [("roll", None), ("stop", None)],
        ),
        (
            Observation(0, DiceCounts((0, 3, 0, 0, 1, 0))),
            [
                ("hold", DiceCounts((0, 3, 0, 0, 0, 0))),
                ("hold", DiceCounts((0, 3, 0, 0, 1, 0))),
                ("hold", DiceCounts((0, 0, 0, 0, 1, 0))),
            ],
        ),
        (
            Observation(350, DiceCounts((0, 0, 0, 0, 0, 0))),
            [("roll", None), ("stop", None)],
        ),
        (
            Observation(0, DiceCounts((0, 0, 0, 0, 0, 0))),
            [("roll", None)],
        ),
    ],
)
def test_determine_available_actions(obs: Observation, expected: list[ActionT]) -> None:
    assert determine_available_actions(obs) == sorted(expected)


# def test_drawing_garden_variety_enables_roll(self) -> None:
#     game = Game()
#     game.card = make_a_bonus_card(300)
#     self._test_available_actions(game, ["roll"])


# def test_rolling_pointers_enables_holds(self) -> None:
#     game = Game()
#     game.card = make_a_bonus_card(300)
#     game.dice = [1, 1, 2, 2, 5]
#     self._test_available_actions(
#         game,
#         [
#             "hold_100000",
#             "hold_200000",
#             "hold_000010",
#             "hold_100010",
#             "hold_200010",
#         ],  # type:ignore
#     )


# def test_holding_dice_enables_roll_or_stop(self) -> None:
#     game = Game()
#     game.card = make_a_bonus_card(300)
#     game.dice = [1, 1, 2, 2, 5]
#     game.handle_action("hold_200000")  # type:ignore
#     self._test_available_actions(game, ["roll", "stop"])
#     self.assertEqual(game.turn_score, 200)


# def test_filling_enables_draw_or_stop(self) -> None:
#     game = Game()
#     game.card = make_a_bonus_card(300)
#     game.handle_action("hold_111111")  # type:ignore
#     self._test_available_actions(game, ["draw", "stop"])
#     self.assertEqual(game.turn_score, 1800)
#     self.assertEqual(game.banked_score, 0)


# def test_fill_1000_prevents_stopping_until_filled(self) -> None:
#     game = Game()
#     game.card = make_a_fill_1000()
#     game.handle_action("hold_100000")  # type:ignore
#     self._test_available_actions(game, ["roll"])
#     game.handle_action("hold_000050")  # type:ignore
#     self._test_available_actions(game, ["draw", "stop"])


# ---------------------------- #
# Test that state is immutable
# ---------------------------- #
# TODO: Test that the state is immutable and the agent will learn
# def test_previous_state_does_not_change_after_action() -> None:
#     game = Game()
#     previous_state = game.state
#     game.handle_action("draw")
#     assert previous_state != game.state


# class TestCardRequirements(unittest.TestCase):
#     agent = Agent("tester")

#     def test_garden_variety_bonus_gets_applied_to_fill(self) -> None:
#         game = Game()
#         game.card = make_a_bonus_card(300)
#         game.dice = [1, 1, 2, 2, 2, 5]
#         game.handle_action("hold_230010")  # type:ignore
#         game.handle_action("stop")
#         self.assertEqual(game.banked_score, 750)

#     def test_garden_variety_allows_stopping(self) -> None:
#         game = Game()
#         game.card = make_a_bonus_card(300)
#         game.dice = [1, 1, 2, 2, 3, 5]
#         game.handle_action("hold_100000")  # type:ignore
#         available_actions = sorted(self.agent.determine_available_actions(game.state))
#         expected_actions = ["roll", "stop"]
#         self.assertEqual(available_actions, expected_actions)
#         game.handle_action("stop")
#         self.assertEqual(game.banked_score, 100)

#     def test_fill_1000_requires_fill(self) -> None:
#         game = Game()
#         game.card = make_a_fill_1000()
#         game.dice = [1, 1, 2, 2, 3, 5]
#         game.handle_action("hold_100000")  # type:ignore
#         available_actions = self.agent.determine_available_actions(game.state)
#         expected_actions = ["roll"]
#         self.assertEqual(available_actions, expected_actions)

#     def test_no_dice_ends_turn(self) -> None:
#         game = Game()
#         game.card = make_a_bonus_card(300)
#         game.dice = [1, 1, 2, 2, 2, 5]
#         game.handle_action("hold_230010")  # type:ignore
#         game.deck.cards = [make_a_no_dice_card()]
#         game.handle_action("draw")
#         self.assertEqual(game.history[-2], "draw: no_dice")
#         self.assertEqual(game.banked_score, 0)
