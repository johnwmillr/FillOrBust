from unittest.mock import patch

import pytest

from dice.agents import Agent, determine_available_actions
from dice.game import FillOrBust, calc_score, hold_dice, is_fill, roll_dice
from dice.types import Action, DiceCounts


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


# ------------------------------ #
# Test the `hold_dice` function
# ------------------------------ #
@pytest.mark.parametrize(
    ["current_holds", "new_holds", "expected_holds"],
    [
        pytest.param(
            DiceCounts(0, 0, 0, 0, 0, 0),
            DiceCounts(1, 1, 1, 1, 1, 1),
            DiceCounts(1, 1, 1, 1, 1, 1),
            id="Hold a straight",
        ),
        pytest.param(
            DiceCounts(0, 0, 0, 0, 1, 0),
            DiceCounts(1, 0, 0, 0, 0, 0),
            DiceCounts(1, 0, 0, 0, 1, 0),
            id="Hold a one",
        ),
    ],
)
def test_hold_dice(
    current_holds: DiceCounts,
    new_holds: DiceCounts,
    expected_holds: DiceCounts,
) -> None:
    updated_holds = hold_dice(current_holds, new_holds)
    assert updated_holds == expected_holds


# --------------------------- #
# Test the `is_fill` function
# --------------------------- #
@pytest.mark.parametrize(
    ["hold", "expected"],
    [
        (DiceCounts(1, 1, 1, 1, 1, 1), True),
        (DiceCounts(1, 1, 1, 1, 1, 0), False),
        (DiceCounts(6, 0, 0, 0, 0, 0), True),
        (DiceCounts(5, 0, 0, 0, 0, 0), False),
        (DiceCounts(1, 0, 0, 0, 0, 0), False),
        (DiceCounts(1, 2, 0, 0, 0, 0), False),
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
        pytest.param(DiceCounts(0, 6, 0, 0, 0, 0), 0, id="Six 2's"),
        pytest.param(DiceCounts(1, 0, 0, 0, 0, 0), 100, id="One 1"),
        pytest.param(DiceCounts(2, 0, 0, 0, 0, 0), 200, id="Two 1's"),
        pytest.param(DiceCounts(0, 0, 0, 0, 1, 0), 50, id="One 5"),
        pytest.param(DiceCounts(0, 0, 0, 0, 2, 0), 100, id="Two 5's"),
        pytest.param(DiceCounts(0, 1, 1, 2, 0, 2), 0, id="Bust"),
        pytest.param(DiceCounts(1, 2, 0, 2, 1, 0), 150, id="One 1, one 5"),
        pytest.param(DiceCounts(2, 0, 1, 2, 0, 0), 200, id="Two 1's"),
    ],
)
def test_calc_score(dice: DiceCounts, expected_score: int) -> None:
    assert calc_score(dice) == expected_score


@pytest.fixture()
def agent() -> Agent:
    return Agent("test_agent")


# ---------------------------------- #
# Test `determine_available_actions`
# ---------------------------------- #
@pytest.mark.parametrize(
    ["obs", "expected"],
    [
        pytest.param(
            fill_or_bust,
            [("roll", None)],
            id="New game. Must roll.",
        ),
        pytest.param(
            fill_or_bust(DiceCounts(ones=2, twos=2, fives=1), DiceCounts(ones=1)),
            [
                ("hold", DiceCounts(ones=1)),
                ("hold", DiceCounts(ones=2)),
                ("hold", DiceCounts(ones=1, fives=1)),
                ("hold", DiceCounts(ones=2, fives=1)),
                ("hold", DiceCounts(fives=1)),
            ],
            id="Rolled some pointers. Must hold.",
        ),
    ],
)
def test_determine_available_actions(env: FillOrBust, expected: list[Action]) -> None:
    assert determine_available_actions(env) == sorted(expected)
