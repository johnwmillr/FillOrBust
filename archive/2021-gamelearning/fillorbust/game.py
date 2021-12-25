from collections import Counter
from random import choice, randint
from typing import List, Tuple

import numpy as np

from .agents import DumbAgent


NUM_DICE = 6
TRIPLE_VALUES = [1000, 200, 300, 400, 500, 600]
SINGLE_VALUES = {1: 100, 5: 50}
MAX_SCORE = 1_000


class FillOrBust:
    def __init__(self, players: Tuple[DumbAgent, ...], max_score: int = MAX_SCORE) -> None:
        self.players = players
        self.num_players = len(self.players)
        self.current_player = choice(["X", "O"])
        self.max_score = max_score
        self.dice = NUM_DICE * [None]
        self.keep_rolling = True
        self.score = 0
        self.states = []

    def advance_to_next_player(self) -> None:
        self.current_player = "O" if self.current_player == "X" else "X"
        self.keep_rolling = True
        self.reset_dice()

    @property
    def state(self) -> str:
        return f"{self.score},{self.current_player},{self.num_dice_remaining}"

    def record_state(self, symbol, turn_score, score, num_dice) -> None:
        state = f"{symbol},{turn_score},{score},{num_dice}"
        self.states.append(state)

    @property
    def num_dice_remaining(self) -> int:
        return len(self.dice)

    def roll(self) -> List[int]:
        self.dice = np.sort(np.random.randint(1, 7, self.num_dice_remaining))
        print(self.dice)

    def reset_dice(self, num_dice_to_roll: int = NUM_DICE) -> None:
        self.dice = num_dice_to_roll * [None]

    def take_action_on_dice(self, action: Tuple[int], keep_rolling: bool) -> Tuple[int, int]:
        score = self.calc_score_on_dice(action)
        self.reset_dice(len(self.dice) - len(action))
        self.keep_rolling = keep_rolling
        return score

    def take_turn(self, player: DumbAgent) -> int:
        print(f"Player: {player.symbol}, {player.score}")
        turn_score = 0
        while not self.turn_is_over:
            self.roll()
            if self.dice_is_bust():
                turn_score = 0
                break

            # Get action from player (if we didn't bust)
            action, keep_rolling = player.select_action_from_dice(self.dice)
            turn_score += self.take_action_on_dice(action, keep_rolling)

            # Check for a fill
            if self.num_dice_remaining == 0:
                turn_score += 500

        player.score += turn_score
        self.reset_dice()
        self.advance_to_next_player()
        return

    def calc_score_on_dice(self, dice: List[int]) -> int:
        # First step: just count 1's and 5's
        score = 0
        counts = Counter(dice)
        for pointer, value in SINGLE_VALUES.items():
            score += counts.get(pointer, 0) * value
        return score

    def dice_is_bust(self) -> bool:
        if not any(self.dice):
            return False
        return not any(p in self.dice for p in [1, 5])

    @property
    def turn_is_over(self) -> bool:
        if self.dice_is_bust():
            return True
        if not self.keep_rolling:
            return True
        if len(self.dice) == 0:
            return True
        return False

    @property
    def game_is_over(self) -> bool:
        return any(player.score >= self.max_score for player in self.players)

    @property
    def winner(self) -> bool:
        highest_score = 0
        player_with_highest_score = None
        for player in self.players:
            if player.score > highest_score:
                highest_score = player.score
                player_with_highest_score = player.symbol
        return player_with_highest_score

    def _dice_to_display_str(self, dice: List[str]) -> str:
        pass

    def __str__(self):
        return self._dice_to_display_str(self.dice)

    def __repr__(self) -> str:
        return self.__str__()
