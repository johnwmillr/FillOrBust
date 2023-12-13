from itertools import product
import pickle
import random
import time
from typing import cast

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from dice.game import Game
from dice.q import QTable, UnknownActionError
from dice.types import ActionT, StateT


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.scores = []
        self.total_score = 0
        self.best_score = -1
        self.num_games_played = 0

    def determine_hold_options_from_dice(
        self, dice: list[int], allow_triples: bool = False
    ) -> tuple[ActionT, ...]:
        options: dict[int, tuple[str, ...]] = {
            n: ("0",) for n in range(1, Game.NUM_DICE + 1)
        }
        for pointer in [1, 5]:
            options[pointer] = tuple(map(str, range(dice.count(pointer) + 1)))

        if allow_triples:
            for triple in [2, 3, 4, 6]:
                count = dice.count(triple)
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
        return holds

    def make_rolled_dice_from_state(self, state: StateT) -> list[int]:
        dice: list[int] = []
        for die, value in state["rolled"].items():
            dice.extend([die for _ in range(value)])
        return dice

    def determine_available_actions(self, state: StateT) -> tuple[ActionT, ...]:
        options: tuple[ActionT, ...] = ()
        if state["num_dice"] == 0:  # Fill
            return ("stop",)
        if (
            sum(state["rolled"].values()) > 0
            and Game.calc_score(self.make_rolled_dice_from_state(state)) == 0
        ):  # Bust
            return (None,)
        if state["num_dice"] > sum(state["rolled"].values()):  # Unrolled dice
            options = ("roll",)
            if state["hold_score"]:
                options += ("stop",)
            return options

        # Check whether it's time to hold dice
        if (
            sum(state["rolled"].values()) == state["num_dice"]
        ):  # TODO: Is this always True?
            hold_options = self.determine_hold_options_from_dice(
                self.make_rolled_dice_from_state(state)
            )
            return hold_options
        raise Exception(f"Unhandled state: {state=}")

    def choose_action(self, state: StateT) -> ActionT:
        return random.choice(self.determine_available_actions(state))

    def play(self) -> None:
        game = Game()  # Empty state for each new episode (i.e. new game)
        while not game.turn_is_over:
            action = self.choose_action(game.state)
            if action is None:
                game.handle_bust()
            elif action == "roll":
                game.handle_roll()
            elif action.startswith("hold"):
                game.handle_hold(action)
            elif action == "stop":
                game.handle_stop()
            else:
                raise UnknownActionError(action)
        self.scores.append(game.state["final_score"])
        self.total_score += game.state["final_score"]
        if game.state["final_score"] > self.best_score:
            self.best_score = game.state["final_score"]
        self.num_games_played += 1

    def print_performance(self) -> None:
        scores = sorted(self.scores)
        median = scores[len(scores) // 2]
        print(
            f"{self.name.title()} Agent average score: {self.total_score // self.num_games_played:7.0f} | Median: {median:4.0f} | Best Score: {self.best_score} | ({len(self.scores)} Games)"
        )

    def __repr__(self) -> str:
        return f"<Agent: '{self.name}', score: {self.total_score}>"


class RiskyAgent(Agent):
    def choose_action(self, state: StateT) -> ActionT:
        possible_actions = self.determine_available_actions(state)
        if "roll" in possible_actions:
            if state["num_dice"] > 1:
                return "roll"
        return super().choose_action(state)


class StopAtNDice(Agent):
    def __init__(self, name: str, num_dice_to_stop_at: int) -> None:
        super().__init__(name)
        self.num_dice_to_stop_at = num_dice_to_stop_at

    def choose_action(self, state: StateT) -> ActionT:
        possible_actions = self.determine_available_actions(state)
        if "stop" in possible_actions and state["num_dice"] <= self.num_dice_to_stop_at:
            return "stop"

        possible_actions = [action for action in possible_actions if action != "stop"]
        return random.choice(possible_actions)  # type: ignore


class Smart(Agent):
    def __init__(self, name: str, Q: QTable | None = None) -> None:
        super().__init__(name)
        self.Q = QTable() if Q is None else Q
        self.evaluations: list[tuple[int, float]] = []

    def choose_action(self, state: StateT) -> ActionT:
        assert self.Q is not None, f"{self.name}'s Q Table is empty."

        if Q_values := self.Q.get_Q_values_for_state(state):
            max_reward = -1.0
            best_action: ActionT = None
            for action, reward in Q_values.items():
                if reward > max_reward:
                    max_reward = reward
                    best_action = action
            if best_action and best_action in self.determine_available_actions(state):
                return best_action
        return super().choose_action(state)

    def train(
        self, num_episodes: int = 1_000, evaluate: bool = False, metric: str = "mean"
    ) -> None:
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
                reward = 0  # TODO: why doesn't manipulating the reward have more of an affect on performance?
                previous_state = game.state
                if action is None:
                    game.handle_bust()
                elif action == "roll":
                    game.handle_roll()
                elif action.startswith("hold"):
                    game.handle_hold(action)
                elif action == "stop":
                    game.handle_stop()
                    reward = game.state["final_score"]
                else:
                    raise UnknownActionError(action)

                # Update the Q table based on the last action
                self.Q.update(previous_state, action, game.state, reward)

            # Perform incremental evaluations during training
            if evaluate and ((num_episode % (num_episodes / 100)) == 0):
                self.evaluations.append(
                    (num_episode, self.evaluate(num_episode=num_episode, metric=metric))
                )
        elapsed = (time.time() - t_start) / 60
        print(
            f"{self.name.title()} Agent is done training. ({num_episodes} episodes.) ({elapsed:0.1f} minutes)"
        )

    def evaluate(
        self, num_games: int = 100, num_episode: int | None = None, metric: str = "mean"
    ) -> float:
        agent = Smart(f"eval_{num_episode}" if num_episode else "eval", Q=self.Q)
        for _ in range(num_games):
            agent.play()
        if metric == "mean":
            return agent.total_score / agent.num_games_played
        elif metric == "median":
            return sorted(agent.scores)[num_games // 2]
        else:
            raise Exception(f"Unknown evaluation metric: '{metric}'")

    def plot_evaluations(self) -> None:
        if not len(self.evaluations):
            print("No evaluations to plot.")
            return
        _, ax = plt.subplots(figsize=[5, 3])
        assert isinstance(ax, Axes)
        x, y = zip(*self.evaluations)
        ax.plot(x, y, "k.-", label=self.Q, alpha=0.6)
        ax.set_ylim((0, 350))
        ax.legend()
        plt.show()

    def save(self, fn: str) -> None:
        with open(fn, "wb") as fout:
            pickle.dump(self, fout)
            print(f"Saved trained agent to '{fn}'")
