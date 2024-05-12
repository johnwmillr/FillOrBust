from unittest.mock import patch
import pytest

from dice.cards import GardenVarietyBonusT, make_a_bonus_card, make_a_fill_1000
from dice.game import (
    DiceCounts,
    calc_score,
    FillOrBustEnvironment,
    determine_available_actions,
    determine_hold_options_from_dice,
    is_fill,
    roll_dice,
)
from dice.types import ActionT, Observation


# ------------------------------ #
# Test the `roll_dice` function
# ------------------------------ #
@pytest.mark.parametrize(
    ["dice", "expected_counts"],
    [
        ([1, 2, 3, 4, 5, 6], (1, 1, 1, 1, 1, 1)),
        ([1, 1, 1, 1, 1, 1], (6, 0, 0, 0, 0, 0)),
        ([1, 1, 1], (3, 0, 0, 0, 0, 0)),
    ],
)
def test_roll_dice(dice: list[int], expected_counts: DiceCounts) -> None:
    with patch("dice.game.random.randint", side_effect=dice):
        assert roll_dice(len(dice)) == expected_counts


# --------------------------- #
# Test the `is_fill` function
# --------------------------- #
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


# ---------------------------------- #
# Test `determine_available_actions`
# ---------------------------------- #
# These parameterized tests are pretty hard to read
@pytest.mark.parametrize(
    ["obs", "expected"],
    [
        (
            Observation(0, 0, DiceCounts((0, 0, 0, 0, 0, 0)), make_a_bonus_card(300)),
            [("roll", None)],
        ),
        (
            Observation(0, 0, DiceCounts((1, 0, 0, 0, 0, 0)), make_a_bonus_card(300)),
            [("hold", DiceCounts((1, 0, 0, 0, 0, 0)))],
        ),
        (
            Observation(100, 1, DiceCounts((0, 0, 0, 0, 0, 0)), make_a_bonus_card(300)),
            [("roll", None), ("stop", None)],
        ),
        (
            Observation(100, 2, DiceCounts((0, 0, 0, 0, 0, 0)), make_a_fill_1000()),
            [("roll", None)],
        ),
        (
            Observation(0, 0, DiceCounts((0, 3, 0, 0, 1, 0)), make_a_bonus_card(300)),
            [
                ("hold", DiceCounts((0, 3, 0, 0, 0, 0))),
                ("hold", DiceCounts((0, 3, 0, 0, 1, 0))),
                ("hold", DiceCounts((0, 0, 0, 0, 1, 0))),
            ],
        ),
        (
            Observation(350, 4, DiceCounts((0, 0, 0, 0, 0, 0)), make_a_bonus_card(300)),
            [("roll", None), ("stop", None)],
        ),
        (
            Observation(0, 0, DiceCounts((0, 0, 0, 0, 0, 0)), make_a_bonus_card(300)),
            [("roll", None)],
        ),
        (
            Observation(0, 0, DiceCounts((0, 0, 0, 0, 0, 0)), None),
            [("draw", None)],
        ),
        (
            # Test the scenario where we have a must-fill card and we've filled and can either stop or draw a new card
            Observation(1500, 6, DiceCounts((0, 0, 0, 0, 0, 0)), make_a_fill_1000()),
            [("draw", None), ("stop", None)],
        ),
    ],
)
def test_determine_available_actions(obs: Observation, expected: list[ActionT]) -> None:
    assert determine_available_actions(obs) == sorted(expected)


@pytest.mark.parametrize(
    ["dice", "bonus", "expected_score"],
    [
        (DiceCounts((2, 3, 0, 0, 1, 0)), 300, 750),
        (DiceCounts((1, 2, 1, 1, 0, 1)), 300, 400),
        (DiceCounts((0, 3, 1, 1, 0, 1)), 500, 700),
        (DiceCounts((2, 2, 0, 0, 1, 0)), 300, 250),  # No fill
        (DiceCounts((1, 2, 0, 1, 0, 1)), 300, 100),  # No fill
        (DiceCounts((0, 3, 1, 1, 0, 0)), 500, 200),  # No fill
    ],
)
def test_garden_variety_bonus_only_gets_applied_with_fill(
    dice: DiceCounts,
    bonus: GardenVarietyBonusT,
    expected_score: int,
    game: FillOrBustEnvironment,
) -> None:
    game.card = make_a_bonus_card(bonus)

    game.dice = dice
    game.step(("hold", dice))
    game.step(("stop", None))
    assert game.game_score == expected_score


# ---------------------------- #
# Test that state is immutable
# ---------------------------- #
# TODO: Test that the state is immutable and the agent will learn
# def test_previous_state_does_not_change_after_action() -> None:
#     game = Game()
#     previous_state = game.state
#     game.handle_action("draw")
#     assert previous_state != game.state
