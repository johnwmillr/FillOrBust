from unittest.mock import patch

import pytest

from dice.agents import Agent, determine_available_actions
from dice.game import calc_score, is_fill, roll_dice
from dice.types import Action, DiceCounts, Observation


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
            Observation(num_dice_remaining=0, held_score=0),
            ["roll"],
            id="New game. Must roll.",
        ),
        pytest.param(
            Observation(num_dice_remaining=3, held_score=100),
            ["roll", "stop"],
            id="Rolled some pointers. Can roll or stop.",
        ),
    ],
)
def test_determine_available_actions(obs: Observation, expected: list[Action]) -> None:
    assert determine_available_actions(obs) == sorted(expected)
