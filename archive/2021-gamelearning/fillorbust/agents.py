from itertools import combinations
from random import choice, random
from typing import List, Tuple
import pandas as pd


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

    def update_prior(self, *args, **kwargs) -> None:
        pass

    def determine_possible_actions(self, dice: List[int]) -> List[Tuple[Tuple[int, ...], bool]]:
        actions = []
        singles = [d for d in dice if d in [1, 5]]
        for n in range(1, len(singles) + 1):
            options = sorted(combinations(singles, n))
            options = [(action, True) for action in options] + [(action, False) for action in options]
            actions.extend(options)
        return sorted(set(actions), key=lambda x: (len(x[0]), min(x[0])))

    def produce_possible_state(self, dice: List[int], action: List[int], keep_rolling: bool) -> str:
        num_dice_remaining = len(dice) - len(action)
        action = "".join(map(lambda x: str(x), action)) + f",{keep_rolling}"
        state = f"{self.symbol},{action},{num_dice_remaining}"
        return state

    def determine_possible_next_states(self, dice: List[int]) -> List[int]:
        possible_actions = self.determine_possible_actions(dice)
        possible_states = set()
        for n, action in enumerate(possible_actions):
            possible_states.add(self.produce_possible_state(dice, *action))
        return list(possible_states)

    def get_state_value(self, state: str) -> float:
        return self.prior.get(state, 0.5)

    @property
    def agent_win_percentage(self) -> float:
        try:
            return self.num_wins / self.num_games
        except ZeroDivisionError:
            return 0

    def log_outcome(self, winner: str) -> None:
        self.num_games += 1
        if winner == self.symbol:
            self.num_wins += 1
        if winner == "T":
            self.num_ties += 1

    def save(self, filename: str = None) -> None:
        """Save the trained agent to a csv file"""
        if filename is None:
            filename = f"fob-agent_{self.symbol}_{self.num_games}"
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
            .assign(trained_symbol=self.symbol)
            .sort_values(by="count", ascending=False)
        )
        return states.to_csv(index=False)

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return (
            f"Agent '{self.symbol}' Record: {self.num_wins}/{self.num_games} ({100 * self.agent_win_percentage:.1f}%)"
        )


class DumbAgent(Agent):
    def __init__(self, symbol: str):
        super().__init__(symbol)

    def select_action(self, dice: List[str]) -> Tuple[Tuple[int], bool]:
        actions = self.determine_possible_actions(dice)
        if len(actions) == 0:
            return None, False
        action = choice(actions)
        return action

    def __repr__(self):
        return super().__repr__() + " (Dumb)"


class SmartAgent(Agent):
    def __init__(self, symbol: str, epsilon: float = 0.1, alpha: float = 0.9, verbose=False):
        super().__init__(symbol)

        self.epsilon = epsilon
        self.alpha = alpha
        self.verbose = verbose

    def select_action(self, dice: List[str]) -> Tuple[int]:
        possible_actions = self.determine_possible_actions(dice)
        if random() < self.epsilon:
            return choice(possible_actions)

        # Exploit
        possible_states = self.determine_possible_next_states(dice)
        probs = [self.get_state_value(s) for s in possible_states]
        best_index = choice([i for i, prob in enumerate(probs) if prob == max(probs)])
        return possible_actions[best_index]

    def update_prior(self, states: List[str], winner: str) -> None:
        states = states[::-1]
        for n, state in enumerate(states):
            self.states[state] = self.states.get(state, 0) + 1

            # Recalculate the current state's win probability
            v = self.get_state_value(state)
            if n == 0:
                if winner == self.symbol:
                    reward = self.score
                else:
                    reward = 0
                v = v + (reward - v) / self.states[state]
            else:
                # Back-propagate win probabilities to earlier states
                v_next_state = self.get_state_value(states[n - 1])
                v += self.alpha * (v_next_state - v)
            self.prior[state] = v

        # Decrease alpha over time
        if self.alpha > 0.2 and self.num_games % 100_000 == 0:
            self.alpha *= 0.99

    def save(self, filename: str = None) -> None:
        """Save the trained agent to a csv file"""
        if filename is None:
            filename = f"fob-agent_{self.symbol}_{self.num_games}_ep_{self.epsilon}_alpha_{self.alpha}"
        if not filename.endswith(".csv"):
            filename += ".csv"
        with open(filename, "w") as fout:
            fout.write(self._to_csv())
        print(f"Saved agent: '{filename}'")

    def load_trained_agent(self, filename: str) -> None:
        training = pd.read_csv(filename, dtype={"state": str})

        # Flip the trained agent symbol if necessary
        trained_symbol = training["trained_symbol"].iloc[0]
        own_symbol = self.symbol
        if trained_symbol != own_symbol:
            training = training.assign(
                state_copy=lambda df: df["state"]
                .str.replace(trained_symbol, "Y")
                .str.replace(own_symbol, trained_symbol)
                .str.replace("Y", own_symbol),
                state=lambda df: df["state_copy"],
                trained_symbol=own_symbol,
            ).drop(columns=["state_copy"])

        # Populate the agent with the trained data
        training = training.set_index("state")
        self.states = training["count"].to_dict()
        self.prior = training["win_percentage"].to_dict()
        self.reset_game_history()

    def _to_csv(self) -> str:
        from io import StringIO

        states = pd.read_table(StringIO(super()._to_csv()), sep=",", dtype={"state": str})
        states = states.assign(epsilon=self.epsilon, alpha=self.alpha)
        return states.to_csv(index=False)

    def __repr__(self):
        return (
            super().__repr__()
            + f" (Smart) (Epsilon: {self.epsilon}) (Alpha: {self.alpha: 0.3f}) (States: {len(self.prior)})"
        )
