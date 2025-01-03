import random

from dice.agents import determine_hold_options_from_dice
from dice.constants import NUM_DICE_IN_GAME, NUM_SIDES_ON_DIE
from dice.types import (
    Action,
    ActionName,
    DiceCounts,
    Observation,
    Reward,
    Terminated,
    TrainingStats,
    Truncated,
)


def calc_score(dice: DiceCounts) -> int:
    assert len(dice), "Can't score empty dice"
    TRIPLE_VALUES = {1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}

    # Check for a straight
    if dice == (1, 1, 1, 1, 1, 1):
        return 1500

    # Check for triples and singles
    score = 0
    for die, count in enumerate(dice, start=1):
        if count == 6:
            return 2 * TRIPLE_VALUES[die]
        elif count >= 3:
            score += TRIPLE_VALUES[die]

        # Check for any single pointers
        if die == 1:
            score += 100 * (count % 3)
        elif die == 5:
            score += 50 * (count % 3)
    return score


def roll_dice(num_dice: int) -> DiceCounts:
    dice = list(random.randint(1, NUM_SIDES_ON_DIE) for _ in range(num_dice))
    return DiceCounts(*(dice.count(die) for die in range(1, NUM_SIDES_ON_DIE + 1)))


def hold_dice(current_holds: DiceCounts, new_holds: DiceCounts) -> DiceCounts:
    updated_holds = DiceCounts(
        *(old_hold + new_hold for old_hold, new_hold in zip(current_holds, new_holds))
    )
    return updated_holds


def is_bust(dice: DiceCounts) -> bool:
    return dice.ones == 0 and dice.fives == 0 and max(dice) < 3


def is_fill(held_dice: DiceCounts) -> bool:
    return sum(held_dice) == NUM_DICE_IN_GAME


def convert_num_turns_to_reward(num_turns: int) -> float:
    return 1e3 / num_turns


class FillOrBust:
    MAX_SCORE = 10_000
    last_action: ActionName | None

    def __init__(self) -> None:
        self.dice = DiceCounts()
        self.held_dice = DiceCounts()
        self.held_score = 0
        self.game_score = 0
        self.last_action = None

        # Training stats
        self.turn_count = 0

    def reset_for_new_game(self) -> Observation:
        self.dice = DiceCounts()
        self.held_dice = DiceCounts()
        self.held_score = 0
        self.game_score = 0
        self.last_action = None
        self.turn_count = 0
        return self._get_obs()

    @property
    def num_dice(self) -> int:
        return NUM_DICE_IN_GAME - sum(self.held_dice)

    def step(
        self, action: Action
    ) -> tuple[Observation, Reward, Terminated, Truncated, TrainingStats]:
        reward, terminated, truncated = 0.0, False, False

        match action[0]:
            case "roll":
                self.dice = roll_dice(self.num_dice)
                if is_bust(self.dice):
                    self.dice = DiceCounts()
                    self.held_dice = DiceCounts()
                    self.held_score = 0
                    self.turn_count += 1
            case "hold":
                assert isinstance(action[1], DiceCounts), action
                assert action[1] in determine_hold_options_from_dice(self.dice), action
                self.held_score += calc_score(action[1])
                self.held_dice = hold_dice(self.held_dice, action[1])
                self.dice = DiceCounts()
                if is_fill(self.held_dice):
                    self.held_dice = DiceCounts()
            case "stop":
                self.game_score += self.held_score
                self.dice = DiceCounts()
                self.held_dice = DiceCounts()
                self.held_score = 0
                self.turn_count += 1
        self.last_action = action[0]

        if self.game_score >= self.MAX_SCORE:
            reward = convert_num_turns_to_reward(self.turn_count)
            terminated = True

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            TrainingStats(self.game_score, self.turn_count),
        )

    def _get_obs(self) -> Observation:
        return Observation(self.dice, self.num_dice, self.held_score, self.last_action)
