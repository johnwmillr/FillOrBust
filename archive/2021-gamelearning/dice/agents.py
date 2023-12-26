from collections import Counter
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
        self.num_actions_in_game = []

    def determine_hold_options_from_dice(
        self,
        dice: Counter[int],
    ) -> tuple[ActionT, ...]:
        # TODO: Explain what `options` is
        options: dict[int, tuple[str, ...]] = {
            n: ("0",) for n in range(1, Game.NUM_DICE + 1)
        }

        # TODO: Make sure we're covering all possible hold options
        for pointer in [1, 5]:
            options[pointer] = tuple(map(str, range(dice[pointer] + 1)))

        # for triple in range(1, Game.NUM_DICE + 1):
        for triple in [2, 3, 4, 6]:
            count = dice[triple]
            if count >= 3:
                options[triple] = ("3",)
                if count == 6:
                    options[triple] += ("6",)

        holds = map(
            lambda x: cast(ActionT, "hold_" + "".join(x)), product(*options.values())
        )
        holds = tuple(
            cast(ActionT, item)
            for item in holds
            if not str(item).endswith("0" * Game.NUM_DICE)
        )
        # Check for a straight
        # TODO: this should be done above
        if dice == Game.STRAIGHT_DICE:
            holds += ("hold_111111",)

        # Check that all holds are valid
        for hold in holds:
            assert sum(map(int, str(hold).split("_")[-1])) <= Game.NUM_DICE, hold

        return holds  # type: ignore

    def make_rolled_dice_from_state(self, state: StateT) -> Counter[int]:
        return state["rolled"]

    def determine_available_actions(self, state: StateT) -> tuple[ActionT, ...]:
        # TODO: validate all of this logic
        options: tuple[ActionT, ...] = ()
        if state["num_dice_remaining"] == 0 and state["held"].total() > 0:  # Fill
            return ("stop", "roll")
        if (
            state["rolled"].total() > 0 and Game.calc_score(state["rolled"]) == 0
        ):  # Bust
            return (None,)
        if state["rolled"].total() == 0:  # Unrolled dice
            options = ("roll",)
            if state["held"].total() > 0:
                options += ("stop",)
            return options

        # Check whether it's time to hold dice
        if state["rolled"].total() > 0:
            return self.determine_hold_options_from_dice(state["rolled"])
        raise Exception(f"Unhandled state: {state=}")

    def choose_action(self, state: StateT) -> ActionT:
        return random.choice(self.determine_available_actions(state))

    def play(self) -> None:
        game = Game()  # Empty state for each new episode (i.e. new game)
        while not game.game_is_over:
            action = self.choose_action(game.state)
            game.handle_action(action)
        self.scores.append(game.state["game_score"])
        self.total_score += game.state["game_score"]
        if game.state["game_score"] > self.best_score:
            self.best_score = game.state["game_score"]
        self.num_actions_in_game.append(len(game.history))
        self.num_games_played += 1

    def print_performance(self) -> None:
        scores = sorted(self.scores)
        median = scores[len(scores) // 2]
        print(
            f"{self.name.title()} Agent average score: {self.total_score // self.num_games_played:7.0f} | Median: {median:4.0f} | Best Score: {self.best_score} | ({len(self.scores)} Games) ({sum(num for num in self.num_actions_in_game) / self.num_games_played:0.1f} actions/game)"
        )

    def __repr__(self) -> str:
        return f"<Agent: '{self.name}', score: {self.total_score}>"


class RiskyAgent(Agent):
    def choose_action(self, state: StateT) -> ActionT:
        possible_actions = self.determine_available_actions(state)
        if "roll" in possible_actions:
            if state["num_dice_remaining"] > 1:
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
            and state["num_dice_remaining"] <= self.num_dice_to_stop_at
        ):
            return "stop"

        possible_actions = [action for action in possible_actions if action != "stop"]
        return random.choice(possible_actions)  # type: ignore


class Smart(Agent):
    def __init__(self, name: str, Q: QTable | None = None) -> None:
        super().__init__(name)
        self.Q = QTable() if Q is None else Q
        self.evaluations: list[tuple[int, float, int]] = []

    def choose_action(self, state: StateT) -> ActionT:
        assert self.Q is not None, f"{self.name}'s Q Table is empty."

        if Q_values := self.Q.get_Q_values_for_state(state):
            # TODO: Optimize for shortest number of turns till 10k
            max_reward = -1.0
            best_action: ActionT = None
            for action, reward in Q_values.items():
                if reward > max_reward:
                    max_reward = reward
                    best_action = action
            if best_action and best_action in self.determine_available_actions(state):
                return best_action
        return super().choose_action(state)

    def train(self, num_episodes: int = 1_000, evaluate: bool = False) -> None:
        t_start = time.time()
        for num_episode in range(num_episodes):
            game = Game()
            while not game.game_is_over:
                # Determine which action to take based on current state
                action: ActionT
                if random.random() < self.Q.epsilon or not self.Q.state_in_table(
                    game.state
                ):
                    action = super().choose_action(game.state)  # Explore
                else:
                    action = self.choose_action(game.state)  # Exploit

                # Perform the selected action
                previous_state = game.state
                game.handle_action(action)
                if game.game_is_over:
                    # Optimize for winning quickly
                    reward = 1000 / len(game.history)
                else:
                    reward = 0

                # Update the Q table based on the last action
                self.Q.update(previous_state, action, game.state, reward)
            self.actions.append(game.history)

            # Perform incremental evaluations during training
            if (num_episode % (num_episodes / 100)) == 0:
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
        num_games: int = 100,
        num_episode: int | None = None,
    ) -> tuple[float, int]:
        agent = Smart(f"eval_{num_episode}" if num_episode else "eval", Q=self.Q)
        for _ in range(num_games):
            agent.play()

        mean = sum(num for num in agent.num_actions_in_game) / agent.num_games_played
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
