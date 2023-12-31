from collections import Counter
from copy import deepcopy
from itertools import product
import pickle
import random
import time
from typing import cast

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from dice.game import Game
from dice.q import QTable
from dice.types import ActionT, StateT


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.scores: list[int] = []
        self.total_score = 0
        self.best_score = -1
        self.num_games_played = 0
        self.actions = []
        self.num_actions_in_game: list[int] = []

    def determine_hold_options_from_dice(
        self,
        dice: Counter[int],
    ) -> list[ActionT]:
        if len(dice) == 0:
            return []

        # Check for a straight
        if dice == Game.STRAIGHT_DICE:
            return [
                "hold_111111",  # type: ignore
            ]

        # Keep track of how many times a particular dice can be held
        options: dict[int, tuple[str, ...]] = {
            n: ("0",) for n in range(1, Game.NUM_DICE + 1)
        }

        # Check or singles
        for single in [1, 5]:
            options[single] = tuple(map(str, range(dice[single] + 1)))

        # Check for triples
        for triple in [2, 3, 4, 6]:
            count = dice[triple]
            if count >= 3:
                options[triple] += ("3",)
                if count == 6:
                    options[triple] += ("6",)

        # Translate dice options to "hold_" strings
        holds = map(
            lambda x: cast(ActionT, "hold_" + "".join(x)), product(*options.values())
        )
        holds = list(
            cast(ActionT, item)
            for item in holds
            if not str(item).endswith("0" * Game.NUM_DICE)
        )

        # Check that all holds are valid
        for hold in holds:
            assert sum(map(int, str(hold).split("_")[-1])) <= Game.NUM_DICE, hold

        return sorted(holds)  # type: ignore

    def determine_available_actions(self, state: StateT) -> list[ActionT]:
        options: list[ActionT] = []
        if state["held"].total() == Game.NUM_DICE:  # Fill
            return ["stop", "roll"]
        if len(state["rolled"]) == 0:  # There are unrolled dice
            options = ["roll"]
            if state["held"].total() > 0:
                options.append("stop")
            return options
        else:
            return self.determine_hold_options_from_dice(state["rolled"])

    def choose_action(self, state: StateT) -> ActionT:
        return random.choice(self.determine_available_actions(state))

    def play(self, verbose: bool = False) -> None:
        game = Game()  # Empty state for each new episode (i.e. new game)
        if verbose:
            print("Starting Game!")
        while not game.turn_is_over:
            action = self.choose_action(game.state)
            if verbose:
                print(f"Game State: {game.state}")
                print(f"Possible Actions: {self.determine_available_actions(game.state)}")
                print(f"Selected Action:  {action}")
            game.handle_action(action)
        self.scores.append(game.state["turn_score"])
        self.total_score += game.state["turn_score"]
        if game.state["turn_score"] > self.best_score:
            self.best_score = game.state["turn_score"]
        self.num_actions_in_game.append(len(game.history))
        self.num_games_played += 1

    def print_performance(self) -> None:
        scores = sorted(self.scores)
        median = scores[len(scores) // 2]
        print(
            f"{self.name.title()} Agent average score: {self.total_score // self.num_games_played:7.0f} | Median: {median:4.0f} | Best Score: {self.best_score} | ({len(self.scores)} Games) ({sum(num for num in self.num_actions_in_game) / self.num_games_played:0.1f} actions/game)"
        )

    def __repr__(self) -> str:
        return f"<Agent: '{self.name}', score: {self.total_score / self.num_games_played:.1f}>"


class RiskyAgent(Agent):
    def choose_action(self, state: StateT) -> ActionT:
        possible_actions = self.determine_available_actions(state)
        if "roll" in possible_actions:
            if (Game.NUM_DICE - state["held"].total()) > 1:
                return "roll"
        return super().choose_action(state)


class StopAtNDice(Agent):
    def __init__(self, name: str, num_dice_to_stop_at: int) -> None:
        super().__init__(name)
        self.num_dice_to_stop_at = num_dice_to_stop_at

    def choose_action(self, state: StateT) -> ActionT:
        possible_actions = self.determine_available_actions(state)
        if (
            "stop" in possible_actions
            and (Game.NUM_DICE - state["held"].total()) <= self.num_dice_to_stop_at
        ):
            return "stop"

        possible_actions = [action for action in possible_actions if action != "stop"]
        return random.choice(possible_actions)  # type: ignore


class Smart(Agent):
    def __init__(self, name: str, Q: QTable | None = None) -> None:
        super().__init__(name)
        self.Q = QTable() if Q is None else Q
        self.evaluations: list[tuple[int, float, int]] = []
        self.print_actions = False

    def choose_action(self, state: StateT) -> ActionT:
        assert self.Q is not None, f"{self.name}'s Q Table is empty."

        if Q_values := self.Q.get_Q_values_for_state(state):
            max_reward = -1.0
            best_action: ActionT | None = None
            for action, reward in Q_values.items():
                if reward > max_reward:
                    max_reward = reward
                    best_action = action
            if best_action and best_action in self.determine_available_actions(state):
                return best_action
        elif self.print_actions:
            print(state)
        return super().choose_action(state)

    def train(self, num_episodes: int = 1_000, evaluate: bool = False, verbose: bool = True) -> None:
        t_start = time.time()
        for num_episode in range(num_episodes):
            game = Game()
            while not game.turn_is_over:
                # Determine which action to take based on current state
                action: ActionT
                if random.random() < self.Q.epsilon or not self.Q.state_in_table(
                    game.state
                ):
                    action = super().choose_action(game.state)  # Explore
                else:
                    action = self.choose_action(game.state)  # Exploit

                # Perform the selected action
                previous_state = deepcopy(game.state)
                game.handle_action(action)
                reward = game.turn_score if game.turn_is_over else 0

                # Update the Q table based on the last action
                self.Q.update(previous_state, action, deepcopy(game.state), reward)
            self.actions.append(game.history)

            # Perform incremental evaluations during training
            if (num_episode % (num_episodes / 100)) == 0:
                if verbose:
                    print(f"{num_episode}/{num_episodes}")
                if evaluate:
                    self.evaluations.append(
                        (num_episode, *self.evaluate(num_episode=num_episode))
                    )
        elapsed = (time.time() - t_start) / 60
        print(
            f"{self.name.title()} Agent is done training. ({num_episodes} episodes.) ({elapsed:0.1f} minutes)"
        )

    def evaluate(
        self,
        num_games: int = 50,
        num_episode: int | None = None,
        optimize_for_score: bool = True,
    ) -> tuple[float, int]:
        agent = Smart(f"eval_{num_episode}" if num_episode else "eval", Q=self.Q)
        for _ in range(num_games):
            agent.play()

        if optimize_for_score:
            mean = agent.total_score / agent.num_games_played
            median = sorted(agent.scores)[agent.num_games_played // 2]
        else:
            mean = (
                sum(num for num in agent.num_actions_in_game) / agent.num_games_played
            )
            median = sorted(agent.num_actions_in_game)[num_games // 2]
        return mean, median

    def plot_evaluations(self) -> None:
        if not len(self.evaluations):
            print("No evaluations to plot.")
            return
        _, ax = plt.subplots(figsize=[5, 3])
        assert isinstance(ax, Axes)

        x, mean, median = zip(*self.evaluations)
        params = (
            str(self.Q)
            .split(": ", 1)[1]
            .strip(">")
            .replace(", ", "\n")
            .replace("self.", "")
        )
        ax.plot(x, mean, "k.-", label=params, alpha=0.6)
        ax.plot(x, median, "r.-", label="median", alpha=0.4)
        ax.grid(True, alpha=0.4)
        # ax.set_ylim((0, 750))
        ax.legend()
        plt.show()

    def save(self, fn: str) -> None:
        with open(fn, "wb") as fout:
            pickle.dump(self, fout)
            print(f"Saved trained agent to '{fn}'")
