import random

from dice.types import (
    Action,
    DiceCounts,
    Observation,
    Reward,
    Terminated,
    TrainingStats,
    Truncated,
)

NUM_DICE_IN_GAME = 6
NUM_SIDES_ON_DIE = 6


def calc_score(dice: DiceCounts) -> int:
    return 100 * dice.ones + 50 * dice.fives


def roll_dice(num_dice: int) -> DiceCounts:
    dice = list(random.randint(1, NUM_SIDES_ON_DIE) for _ in range(num_dice))
    return DiceCounts(*(dice.count(die) for die in range(1, NUM_SIDES_ON_DIE + 1)))


def is_bust(dice: DiceCounts) -> bool:
    return dice.ones == 0  # and dice.fives == 0


def is_fill(held_dice: DiceCounts) -> bool:
    return sum(held_dice) == NUM_DICE_IN_GAME


def convert_num_turns_to_reward(num_turns: int) -> float:
    return 1e3 / num_turns


class FillOrBust:
    MAX_SCORE = 10_000

    def __init__(self) -> None:
        self.num_dice = NUM_DICE_IN_GAME
        self.held_score = 0
        self.game_score = 0

        # Training stats
        self.turn_count = 0

    def reset_for_new_game(self) -> Observation:
        self.num_dice = NUM_DICE_IN_GAME
        self.held_score = 0
        self.game_score = 0
        self.turn_count = 0
        return Observation(self.num_dice, self.held_score)

    def step(
        self, action: Action
    ) -> tuple[Observation, Reward, Terminated, Truncated, TrainingStats]:
        reward, terminated, truncated = 0.0, False, False

        match action:
            case "roll":
                dice = roll_dice(self.num_dice)
                score = calc_score(dice)
                if score == 0:  # Busted
                    self.held_score = 0
                    self.turn_count += 1
                else:
                    self.held_score += score
                    self.num_dice -= dice.ones + dice.fives
            case "stop":
                self.game_score += self.held_score
                self.turn_count += 1

        if self.held_score == 0 or self.num_dice <= 0:
            self.num_dice = NUM_DICE_IN_GAME

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
        return Observation(self.num_dice, self.held_score)
