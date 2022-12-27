from itertools import combinations
from random import choice, random
from typing import List, Tuple, Union

POINTERS = [1, 5]
NUM_DICE = 6


class Agent:
    def __init__(self, name: str):
        self.name = name
        self.score = 0
        self.states = []
        self.state_counts = {}

    def combine_state_and_action(self, state: str, action: str) -> str:
        return f"{state}|{action}"

    def record_state(self, state: str, action: str) -> None:
        self.states.append(self.combine_state_and_action(state, action))
        self.state_counts[state] = self.state_counts.setdefault(state, 0) + 1

    def determine_possible_actions(self, state: str) -> List[str]:
        num_dice, pointers, turn_score = self.parse_state_str(state)
        if len(pointers) == 0:
            actions = ["roll"]
            if num_dice < NUM_DICE or turn_score:
                actions.append("stop")
        else:
            actions = []
            for combo in self.get_pointer_combos(pointers):
                actions.append("".join(str(combo.count(p)) for p in POINTERS))
        return actions

    def parse_state_str(self, state: str) -> Tuple[int, int, int]:
        num_dice, pointers, turn_score = state.split("|")
        pointers = [int(count) * [pointer] for count, pointer in zip(pointers, POINTERS)]
        pointers = [x for y in pointers for x in y]
        return int(num_dice), pointers, int(turn_score)

    def get_pointer_combos(self, pointers: List[int]) -> List[int]:
        combos = []
        for r in range(len(pointers), 0, -1):
            combos.extend(set(combinations(pointers, r)))
        return sorted(combos, key=len)

    def process_reward(self, reward: int) -> None:
        pass

    def state_str_to_dice(self, state: str) -> List[Union[str, None]]:
        str_to_num = lambda s: None if "None" in s else int(s)
        dice = list(map(str_to_num, state.rsplit("|", 1)[-1][1:-1].split(" ")))
        return dice

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Agent: '{self.name}' {self.score}"


class SmartAgent(Agent):
    def __init__(self, name: str, epsilon: float = 0.1, alpha: float = 1.0, verbose=False):
        super().__init__(name)
        self.prior = {}
        self.epsilon = epsilon
        self.alpha = alpha

    def get_prior(self, state: str, action: str, find_closest_state: bool = False) -> float:
        if find_closest_state:
            state = self.find_most_similar_state(state)
        return self.prior.get(self.combine_state_and_action(state, action), 0)

    def select_action(self, state: str) -> Tuple[int]:
        possible_actions = self.determine_possible_actions(state)
        if random() < self.epsilon:
            action = choice(possible_actions)
        else:
            # history = {action: self.    (state, action, True) for action in possible_actions}
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
            reward = self.prior[state]

    def find_most_similar_state(self, state: str) -> str:
        num_dice, pointers, turn_score = state.split("|", 3)[:3]
        tree = self.convert_prior_to_tree()
        if num_dice not in tree or pointers not in tree[num_dice]:
            return f"{num_dice}|{pointers}|{turn_score}"

        scores = tree[num_dice][pointers]
        min_diff = int(1e9)
        closest_score = None
        for score in scores:
            diff = abs(int(turn_score) - int(score))
            if diff < min_diff:
                min_diff = diff
                closest_score = score
        return f"{num_dice}|{pointers}|{closest_score}"

    def convert_prior_to_tree(self) -> dict:
        tree = dict()
        for key, val in self.prior.items():
            num_dice, pointers, turn_score = key.split("|", 3)[:3]
            d = tree.setdefault(num_dice, {})
            p = d.setdefault(pointers, {})
            p[turn_score] = val
        return tree

    def print_prior(self, by_keys: bool = True) -> None:
        if by_keys:
            sorted_prior = sorted(self.prior.items(), key=lambda item: item[0], reverse=False)
        else:
            sorted_prior = sorted(self.prior.items(), key=lambda item: item[1], reverse=True)
        for key, val in sorted_prior:
            print(f"{key}: {val}")

    def __repr__(self):
        return super().__repr__() + " (Smart)"


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


class HumanAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)

    def select_action(self, state: str) -> Tuple[Tuple[int], bool]:
        possible_actions = self.determine_possible_actions(state)
        print(possible_actions)
        if len(possible_actions) == 1:
            action = possible_actions[0]
        else:
            action = input()
        self.record_state(state, action)
        return action

    def __repr__(self):
        return super().__repr__() + " (Human)"
