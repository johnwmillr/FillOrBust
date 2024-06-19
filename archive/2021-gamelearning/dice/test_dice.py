from unittest.mock import patch

import pytest

from dice.agents import (
    Agent,
    determine_available_actions,
    determine_hold_options_from_dice,
)
from dice.cards import (
    Deck,
    GardenVarietyBonusT,
    make_a_bonus_card,
    make_a_fill_1000,
    make_a_must_bust_card,
)
from dice.constants import NUM_DICE_IN_GAME
from dice.game import DiceCounts, FillOrBust, calc_score, hold_dice, is_fill, roll_dice
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


# ------------------------------ #
# Test the `hold_dice` function
# ------------------------------ #
@pytest.mark.parametrize(
    ["dice", "current_holds", "new_holds", "expected_dice", "expected_holds"],
    [
        pytest.param(
            DiceCounts(1, 1, 1, 1, 1, 1),
            DiceCounts(0, 0, 0, 0, 0, 0),
            DiceCounts(1, 1, 1, 1, 1, 1),
            DiceCounts(0, 0, 0, 0, 0, 0),
            DiceCounts(1, 1, 1, 1, 1, 1),
            id="Hold a straight",
        ),
        pytest.param(
            DiceCounts(1, 0, 2, 1, 0, 1),
            DiceCounts(0, 0, 0, 0, 1, 0),
            DiceCounts(1, 0, 0, 0, 0, 0),
            DiceCounts(0, 0, 2, 1, 0, 1),
            DiceCounts(1, 0, 0, 0, 1, 0),
            id="Hold a one",
        ),
    ],
)
def test_hold_dice(
    dice: DiceCounts,
    current_holds: DiceCounts,
    new_holds: DiceCounts,
    expected_dice: DiceCounts,
    expected_holds: DiceCounts,
) -> None:
    updated_dice, updated_holds = hold_dice(dice, current_holds, new_holds)
    assert updated_dice == expected_dice
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
        pytest.param(
            (1, 1, 1, 1, 1, 1),
            1500,
            id="Straight",
        ),
        pytest.param((0, 3, 0, 0, 0, 0), 200, id="Three 2's"),
        pytest.param((0, 6, 0, 0, 0, 0), 400, id="Six 2's"),
        pytest.param((0, 0, 3, 0, 0, 3), 900, id="Three 3's, Three 6's"),
        pytest.param((1, 0, 0, 0, 0, 0), 100, id="One 1"),
        pytest.param((2, 0, 0, 0, 0, 0), 200, id="Two 1's"),
        pytest.param((0, 0, 0, 0, 1, 0), 50, id="One 5"),
        pytest.param((0, 0, 0, 0, 2, 0), 100, id="Two 5's"),
        pytest.param((0, 0, 0, 0, 4, 0), 550, id="Four 5's"),
        pytest.param((4, 0, 0, 0, 0, 0), 1100, id="Four 1's"),
        pytest.param((0, 0, 0, 0, 0, 5), 600, id="Five 6's"),
        pytest.param((0, 1, 1, 2, 0, 2), 0, id="Bust"),
        pytest.param((1, 2, 0, 2, 1, 0), 150, id="One 1, one 5"),
        pytest.param((2, 0, 1, 2, 0, 0), 200, id="Two 1's"),
    ],
)
def test_calc_score(dice: DiceCounts, expected_score: int) -> None:
    assert calc_score(dice) == expected_score


@pytest.fixture()
def agent() -> Agent:
    return Agent("test_agent")


@pytest.fixture()
def game(agent: Agent) -> FillOrBust:
    return FillOrBust([agent])


@pytest.mark.parametrize("expected_score", [300])
def test_same_roll_requirement_for_triples(
    expected_score: int, agent: Agent, game: FillOrBust
) -> None:
    game.dice = DiceCounts(2, 2, 2, 0, 0, 0)
    game.step(("hold", DiceCounts(2, 0, 0, 0, 0, 0)))
    game.dice = DiceCounts(1, 2, 1, 0, 0, 0)
    game.step(("hold", DiceCounts(1, 0, 0, 0, 0, 0)))
    game.step(("stop", None))
    assert agent.game_score == expected_score


# ---------------------------------- #
# Test `determine_available_actions`
# ---------------------------------- #
@pytest.mark.parametrize(
    ["obs", "expected"],
    [
        # pytest.param(
        #     Observation(0, 6, DiceCounts(), make_a_bonus_card(300)),
        #     [("roll", None)],
        #     id="Drew a card, haven't rolled.",
        # ),
        # pytest.param(
        #     Observation(0, 6, DiceCounts(1, 0, 0, 0, 0, 0), make_a_bonus_card(300)),
        #     [("hold", DiceCounts(1, 0, 0, 0, 0, 0))],
        #     id="Rolled 1 one.",
        # ),
        # pytest.param(
        #     Observation(100, 5, DiceCounts(), make_a_bonus_card(300)),
        #     [("roll", None), ("stop", None)],
        #     id="Held dice, can either roll or stop.",
        # ),
        # pytest.param(
        #     Observation(100, 3, DiceCounts(), make_a_fill_1000()),
        #     [("roll", None)],
        #     id="Drew a Fill 1000, must roll.",
        # ),
        # pytest.param(
        #     Observation(0, 6, DiceCounts(2, 2, 2, 0, 0, 0), make_a_bonus_card(300)),
        #     [
        #         ("hold", DiceCounts(1, 0, 0, 0, 0, 0)),
        #         ("hold", DiceCounts(2, 0, 0, 0, 0, 0)),
        #     ],
        #     id="Rolled 2 ones.",
        # ),
        # pytest.param(
        #     Observation(150, 4, DiceCounts(0, 3, 0, 0, 1, 0), make_a_bonus_card(300)),
        #     [
        #         ("hold", DiceCounts(0, 3, 0, 0, 0, 0)),
        #         ("hold", DiceCounts(0, 3, 0, 0, 1, 0)),
        #         ("hold", DiceCounts(0, 0, 0, 0, 1, 0)),
        #     ],
        #     id="Rolled a triple two and 1 five.",
        # ),
        # pytest.param(
        #     Observation(350, 2, DiceCounts(), make_a_bonus_card(300)),
        #     [("roll", None), ("stop", None)],
        #     id="Holding dice, could roll or stop.",
        # ),
        # pytest.param(
        #     Observation(0, 6, DiceCounts(), None),
        #     [("draw", None)],
        #     id="No card, must draw.",
        # ),
        # pytest.param(
        #     Observation(250, 1, DiceCounts(), make_a_must_bust_card()),
        #     [("roll", None)],
        #     id="Must bust, must roll.",
        # ),
        pytest.param(
            Observation(None, 0, DiceCounts()),
            [("roll", NUM_DICE_IN_GAME)],
            id="New turn, must roll.",
        ),
        pytest.param(
            Observation("hold", 1500, DiceCounts()),
            [("roll", NUM_DICE_IN_GAME), ("stop", None)],
            id="Just held a straight, can roll or stop.",
        ),
        pytest.param(
            Observation("roll", 150, DiceCounts(2, 0, 0, 0, 0, 1)),
            [
                ("hold", DiceCounts(1, 0, 0, 0, 0, 0)),
                ("hold", DiceCounts(2, 0, 0, 0, 0, 0)),
            ],
            id="Rolled two ones.",
        ),
        pytest.param(
            Observation("stop", 150, DiceCounts(2, 0, 0, 0, 0, 1)),
            [("roll", NUM_DICE_IN_GAME)],
            id="Just stopped, must roll.",
        ),
    ],
)
def test_determine_available_actions(obs: Observation, expected: list[ActionT]) -> None:
    assert determine_available_actions(obs) == sorted(expected)


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


# ------------------- #
# Test out some cards
# ------------------- #
# @pytest.mark.parametrize(
#     ["dice", "bonus", "expected_score"],
#     [
#         pytest.param(DiceCounts(2, 3, 0, 0, 1, 0), 300, 750, id="Three 2's, one 5"),
#         pytest.param(DiceCounts(1, 2, 1, 1, 0, 1), 300, 400, id="One 1"),
#         pytest.param(DiceCounts(0, 3, 1, 1, 0, 1), 500, 700, id="Three 2's"),
#         pytest.param(
#             DiceCounts(2, 2, 0, 0, 1, 0), 300, 250, id="Two 1's, one 5, no fill"
#         ),
#         pytest.param(DiceCounts(1, 2, 0, 1, 0, 1), 300, 100, id="One 1, no fill"),
#         pytest.param(DiceCounts(0, 3, 1, 1, 0, 0), 500, 200, id="Three 2's, no fill"),
#     ],
# )
# def test_garden_variety_bonus_only_gets_applied_with_fill(
#     dice: DiceCounts,
#     bonus: GardenVarietyBonusT,
#     expected_score: int,
#     agent: Agent,
#     game: FillOrBust,
# ) -> None:
#     game.card = make_a_bonus_card(bonus)

#     game.dice = dice
#     game.step(agent, ("hold", dice))
#     game.step(agent, ("stop", None))

#     assert game.player is not None
#     assert game.player.game_score == expected_score


# def test_that_must_bust_preserves_score_after_busting(
#     agent: Agent,
#     game: FillOrBust,
# ) -> None:
#     game.card = make_a_must_bust_card()
#     game.hold_score = 200
#     with patch("dice.game.roll_dice", return_value=DiceCounts(0, 1, 1, 1, 0, 0)):
#         game.step(agent, ("roll", None))
#         assert game.player is not None
#         assert game.player.game_score == 200


# Write unit tests from the `Deck` class
NUM_CARDS_IN_DECK = sum(
    [
        Deck.NUM_NO_DICE,
        Deck.NUM_FILL_1000,
        Deck.NUM_BONUS_300,
        Deck.NUM_BONUS_400,
        Deck.NUM_BONUS_500,
        Deck.NUM_MUST_BUST,
    ]
)


def test_deck_initialization() -> None:
    deck = Deck()
    assert len(deck.cards) == NUM_CARDS_IN_DECK


def test_deck_draw() -> None:
    deck = Deck()
    card = deck.draw()
    assert len(deck.cards) == NUM_CARDS_IN_DECK - 1
    assert card is not None


def test_deck_empty_draw() -> None:
    deck = Deck()
    for _ in range(NUM_CARDS_IN_DECK):
        deck.draw()
    assert len(deck.cards) == 0

    # Check that the deck is shuffled and can be drawn from again
    deck.draw()
    assert len(deck.cards) == NUM_CARDS_IN_DECK - 1
