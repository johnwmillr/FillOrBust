from collections import Counter
from random import shuffle
from typing import List, Union

import numpy as np

from .agents import POINTERS, DumbAgent, SmartAgent

NUM_DICE = 6
TRIPLE_VALUES = [1000, 200, 300, 400, 500, 600]
SINGLE_VALUES = {1: 100, 5: 50}
MAX_SCORE = 10_000


class FillOrBust:
    def __init__(self, players: List[Union[DumbAgent, SmartAgent]], max_score: int = MAX_SCORE) -> None:
        self.players = players
        for player in self.players:
            player.score = 0
        self.max_score = max_score
        self.turn_score = 0
        self.dice = []
        self._randomize_player_order()
        self.start_new_turn()

    def _randomize_player_order(self) -> None:
        self.name_to_player = {player.name: player for player in self.players}
        self.player_names = [player.name for player in self.players]
        shuffle(self.player_names)
        self.current_player = self.name_to_player[self.player_names[0]]

    def start_new_turn(self) -> None:
        self.current_player = self.get_next_player()
        self.turn_score = 0
        self.dice = np.asarray(NUM_DICE * [None])
        self.roll()

    def finish_turn(self) -> None:
        self.current_player.process_reward(self.turn_score)
        self.start_new_turn()

    def get_next_player(self) -> Union[DumbAgent, SmartAgent]:
        i = self.player_names.index(self.current_player.name)
        name = self.player_names[(i + 1) % len(self.players)]
        return self.name_to_player[name]

    def remove_pointers_update_score_and_dice(self, pointers: List[int]) -> None:
        self.turn_score += self.calc_score_on_dice(pointers)
        self.dice = [None] * (len(self.dice) - len(pointers))
        if len(self.dice) == 0:
            # self.turn_score += 1000
            self.dice = NUM_DICE * [None]

    def perform_action(self, action: str) -> None:
        if action == "stop":
            self.stop()
        elif action == "roll":
            self.roll()

    def stop(self) -> None:
        self.current_player.score += self.turn_score
        self.finish_turn()

    def roll(self) -> List[int]:
        self.dice = np.sort(np.random.randint(1, 7, self.num_dice_remaining))
        busted = self.calc_score_on_dice(self.dice) == 0
        if busted:
            self.turn_score = 0
            self.finish_turn()
            return

        # Always take all pointers (for now)
        pointers = [die for die in self.dice if die in POINTERS]
        self.remove_pointers_update_score_and_dice(pointers)

    def calc_score_on_dice(self, dice: List[int]) -> int:
        score = 0
        counts = Counter(dice)
        for pointer, value in SINGLE_VALUES.items():
            score += counts.get(pointer, 0) * value
        return score

    @property
    def state(self) -> str:
        # name = self.current_player.name
        # score = self.current_player.score
        turn_score = self.turn_score
        num_dice = len(self.dice)
        # num_pointers = len([d for d in self.dice if d in POINTERS])
        state = f"{num_dice}|{turn_score:03d}"
        return state

    @property
    def game_is_over(self) -> bool:
        over_max_score = any(player.score >= self.max_score for player in self.players)
        no_tie = len(set(player.score for player in self.players)) > 1
        is_over = over_max_score and no_tie
        # if is_over:
        #     best_score = max(player.score for player in self.players)
        #     for player in self.players:
        #         reward = 1 if player.score == best_score else 0
        #         player.process_reward(reward)
        return is_over

    @property
    def num_dice_remaining(self) -> int:
        return len(self.dice)

    def _dice_to_display_str(self, dice: List[str]) -> str:
        return str(dice)

    def __str__(self):
        return self._dice_to_display_str(self.dice)

    def __repr__(self) -> str:
        return f"FillOrBust: {self.__str__()}"
