from itertools import combinations
from random import choice, random
from typing import List, Tuple, Union
import pandas as pd

# from fillorbust.game import NUM_DICE


# from .game import SINGLE_VALUES

# POINTERS = list(SINGLE_VALUES.keys())
POINTERS = [1, 5]
NUM_DICE = 6


class Agent:
    def __init__(self, name: str):
        self.name = name
        self.score = 0
        self.states = []

    def combine_state_and_action(self, state: str, action: str) -> str:
        return f"{state}|{action}"

    def record_state(self, state: str, action: str) -> None:
        self.states.append(self.combine_state_and_action(state, action))

    def determine_possible_actions(self, state: str) -> List[str]:
        num_dice, turn_score = self.parse_state_str(state)
        if num_dice:
            action = ["roll"]
            if num_dice < NUM_DICE or turn_score > 0:
                action.append("stop")
            return action

    def get_pointers_from_dice(self, dice) -> List[int]:
        return [die for die in dice if die in POINTERS]

    def parse_state_str(self, state: str) -> Tuple[int, int, int, int]:
        splits = state.split("|")
        # name = splits[0]
        # score = int(splits[0])
        turn_score = int(splits[1])
        num_dice = int(splits[0])
        # num_pointers = int(splits[2])
        return num_dice, turn_score

    def process_reward(self, reward: int) -> None:
        pass

    def state_str_to_dice(self, state: str) -> List[Union[str, None]]:
        str_to_num = lambda s: None if "None" in s else int(s)
        dice = list(map(str_to_num, state.rsplit("|", 1)[-1][1:-1].split(" ")))
        return dice

    def save(self, filename: str = None) -> None:
        """Save the trained agent to a csv file"""
        if filename is None:
            filename = f"fob-agent_{self.name}_{self.num_games}"
        if not filename.endswith(".csv"):
            filename += ".csv"
        with open(filename, "w") as fout:
            fout.write(self._to_csv())
        print(f"Saved agent: '{filename}'")

    def _to_csv(self) -> str:
        counts = pd.DataFrame.from_dict(self.states, orient="index", columns=["count"])
        wins = pd.DataFrame.from_dict(self.prior, orient="index", columns=["win_percentage"])
        states = (
            counts.join(wins, how="outer")
            .fillna(0)
            .reset_index()
            .rename(columns={"index": "state"})
            .assign(trained_name=self.name)
            .sort_values(by="count", ascending=False)
        )
        return states.to_csv(index=False)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Agent: '{self.name}' {self.score}"


class DumbAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)

    def select_action(self, state: str) -> Tuple[Tuple[int], bool]:
        possible_actions = self.determine_possible_actions(state)
        action = choice(possible_actions)
        self.record_state(state, action)
        return action

    def __repr__(self):
        return super().__repr__() + " (Dumb)"


class SmartAgent(Agent):
    def __init__(self, name: str, epsilon: float = 0.1, alpha: float = 1.0, verbose=False):
        super().__init__(name)
        self.prior = {}
        self.epsilon = epsilon
        self.alpha = alpha

    def get_prior(self, state: str, action: str) -> float:
        return self.prior.get(self.combine_state_and_action(state, action), 100)

    def select_action(self, state: str) -> Tuple[int]:
        possible_actions = self.determine_possible_actions(state)
        if random() < self.epsilon:
            # Explore
            action = choice(possible_actions)
        else:
            # Exploit
            history = {action: self.get_prior(state, action) for action in possible_actions}
            max_value = max(history.values())
            action = choice([key for key, value in history.items() if value == max_value])
        self.record_state(state, action)
        return action

    def process_reward(self, reward: int) -> None:
        while self.states:
            state = self.states.pop()
            v = self.get_prior(*state.rsplit("|", 1))
            v += self.alpha * (reward - v)
            self.prior[state] = v

    def print_prior(self, by_keys: bool = True) -> None:
        if by_keys:
            sorted_prior = sorted(self.prior.items(), key=lambda item: item[0], reverse=False)
        else:
            sorted_prior = sorted(self.prior.items(), key=lambda item: item[1], reverse=True)
        for key, val in sorted_prior:
            print(f"{key}: {val}")

    def __repr__(self):
        return super().__repr__() + " (Smart)"
