from itertools import combinations
from random import choice
from typing import List, Tuple


class Agent:
    def __init__(self, symbol: str):
        assert symbol in ["X", "O"], symbol
        self.symbol = symbol
        self.opponent = "O" if self.symbol == "X" else "X"
        self.score = 0
        self.turn_score = 0
        self.states = {}
        self.wins = {}
        self.prior = {}
        self.num_games = 0
        self.num_wins = 0
        self.num_ties = 0
        self.banked_pointers = []

    def determine_possible_actions(self, dice: List[int]) -> List[int]:
        actions = []
        singles = [d for d in dice if d in [1, 5]]
        for n in range(1, len(singles) + 1):
            actions.extend(sorted(combinations(singles, n)))
        return sorted(set(actions), key=lambda x: (len(x), min(x)))

    def produce_state_from_action(self, dice: List[int], action: List[int]) -> int:
        if action == 0:
            state = "S"
        else:
            state = len(dice) - len(action)
        return state

    def determine_possible_states(self, dice: List[int]) -> List[int]:
        possible_actions = self.determine_possible_actions(dice)
        possible_states = set()
        for n, action in enumerate(possible_actions):
            possible_states.add(self.produce_state_from_action(dice, action))
        return possible_states

    def _dice_to_state(self, dice) -> str:
        return "".join(map(str, dice))


class DumbAgent(Agent):
    def __init__(self, symbol: str):
        super().__init__(symbol)

    def select_action_from_dice(self, dice: List[str]) -> Tuple[int]:
        actions = self.determine_possible_actions(dice)
        if len(actions) == 0:
            return None, False
        action = choice(actions)
        roll_again = choice([True, False])
        if not roll_again:
            print("STOPPING")
        return action, roll_again

    def __repr__(self):
        return super().__repr__() + " (Dumb)"
