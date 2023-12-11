from itertools import product
import pickle
import random
import time
from typing import cast

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from dice_types import ActionT, StateT, QTableT


class UnknownActionError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


# state = (
#     "score",
#     "num_dice",
#     "num_ones_rolled",
#     "num_fives_rolled",
#     "num_ones_held",
#     "num_fives_held",
# )
# state = (0, 6, -1, -1, 0, 0)  # -1 means you haven't rolled yet?
# # State at start of game
# # Actions: ROLL, STOP (because of the -1's)
# # ROLL

# state = (0, 6, 2, 1, 0, 0)
# # State after ROLL
# # Actions: HOLD (because sum of held dice is zero?)
# # HOLD_ACTION = (2, 1)  # (num_ones, num_fives)

# state = (0, 3, -1, -1, 2, 1)
# # State after HOLD
# # Actions: ROLL, STOP (because of the ones)
# # STOP

# state = (250, 6, -1, -1, 0, 0)  # New turn?

# %%
# ----------------
state = {
    "score": 0,
    "num_dice": 6,
    "rolled": {
        1: 0,
        2: 3,
        3: 0,
        4: 1,
        5: 1,
        6: 0,
    },
    "held": {
        1: 1,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
    },
}
# %%

# Q Table
# | States  |     Actions     |
# | State   |   Roll |   Stop |
# |---------|--------|--------|
# | (0, []) |   ?    |    ?   |
# | (50, [5]) |  ?   |    ?   |
# |


class QTable:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        """Stores Q-values that map states and actions to rewards.

        Args:
            learning_rate: The learning rate (alpha) for updating Q-values.
                Higher values lead to faster convergence but may result in
                unstable learning. Defaults to 0.1.
            discount_factor: The discount factor (gamma) for future rewards.
                Determines the importance of future rewards. Values closer to 1
                prioritize long-term rewards; values closer to 0 focus on
                short-term gains. Defaults to 0.9.
            epsilon: The exploration-exploitation trade-off factor for
                epsilon-greedy strategy. Higher values prioritize exploration
                over exploitation. Defaults to 0.1.

        Notes:
            - The QTable class manages a Q-table for Q-learning in reinforcement learning.
            - Learning Rate (alpha):
                - Controls the impact of new information on existing Q-values.
                - Increasing the learning rate tends to make the agent adapt to new experiences quickly,
                    but it may result in unstable learning or overshooting the optimal policy.
                - Lower values provide more stable but slower learning.
            - Discount Factor (gamma):
                - Influences the importance of future rewards in Q-value updates.
                - Higher values prioritize long-term rewards, emphasizing future returns in decision-making.
                - Lower values focus more on immediate rewards, leading to short-term gains.
            - Epsilon (Exploration-Exploitation Trade-off):
                - Affects the balance between exploration and exploitation in the agent's behavior.
                - Higher values prioritize exploration, encouraging the agent to take more random actions.
                - Lower values prioritize exploitation, causing the agent to choose known high-reward actions more frequently.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.table: QTableT = {}

    def __repr__(self) -> str:
        return (
            f"<QTable: {self.learning_rate=}, {self.discount_factor}, {self.epsilon=}>"
        )

    @property
    def empty_actions(self) -> dict[ActionT, float]:
        # return {action: 0 for action in VALID_HOLDS + ("roll", "stop")}
        return {"roll": 0, "stop": 0}

    def state_to_Q_table_keys(self, state: StateT) -> tuple[str, ...]:
        NUM_DICE = 6

        score_key = "score_" + str(state["hold_score"])
        roll_key = "roll_count_" + "".join(
            str(state["rolled"][n]) for n in range(1, NUM_DICE + 1) if n in [1, 5]
        )
        hold_key = "hold_count_" + "".join(
            str(state["held"][n]) for n in range(1, NUM_DICE + 1) if n in [1, 5]
        )
        return (score_key, roll_key, hold_key)

    def state_in_table(self, state: StateT) -> bool:
        score_key, roll_key, hold_key = self.state_to_Q_table_keys(state)
        if roll_key in self.table:
            if hold_key in self.table[roll_key]:
                return True
        return False

    def get_Q_values_for_state(self, state: StateT) -> dict[ActionT, float] | None:
        if not self.state_in_table(state):
            return None
        score_key, roll_key, hold_key = self.state_to_Q_table_keys(state)
        return self.table[roll_key][hold_key]

    def update(
        self, state: StateT, action: ActionT, new_state: StateT, reward: float
    ) -> None:
        """
        Update the Q-table based on the observed state, action, new state, and reward.

        Args:
            state: The current state.
            action: The action taken in the current state.
            new_state: The resulting new state after taking the action.
            reward: The immediate reward obtained after taking the action.

        Raises:
            AssertionError: If the action is not recognized.

        Notes:
            This method updates the Q-value in the table for the given
            state-action pair based on the observed reward and the maximum
            Q-value among available actions in the new state. It uses the
            Q-learning update equation to adjust the Q-value toward the optimal
            value.
        """
        # assert action in self.empty_actions, UnknownActionError(action)

        # TODO: need to add "score if stopping now" to state
        # The top-level key only is updated with the final game score.
        # We aren't recording mid-game states that would allow us to
        # decide to stop.

        # assert action is not None
        # if action.startswith("hold_"):
        #     holds = list(action.split("_")[1])
        #     action = "hold_" + str(holds[0]) + str(holds[4])

        # Check if we have reward values for `state` and `new_state`
        for representation in [state, new_state]:
            score_key, roll_key, hold_key = self.state_to_Q_table_keys(representation)

            # if score_key not in self.table:
            #     self.table[score_key] = {}
            if roll_key not in self.table:
                self.table[roll_key] = {}
            if hold_key not in self.table[roll_key]:
                self.table[roll_key][hold_key] = {}
            if action not in self.table[roll_key][hold_key]:
                self.table[roll_key][hold_key][action] = 0.0
                for a, v in self.empty_actions.items():
                    self.table[roll_key][hold_key][a] = v

        # Update the reward base on the action
        score_state, roll_state, hold_state = self.state_to_Q_table_keys(state)
        score_new_state, roll_new_state, hold_new_state = self.state_to_Q_table_keys(
            new_state
        )
        self.table[roll_state][hold_state][action] += self.learning_rate * (
            reward
            + self.discount_factor
            * max(self.table[roll_new_state][hold_new_state].values())
            - self.table[roll_state][hold_state][action]
        )


class Game:
    NUM_DICE = 6

    def __init__(self) -> None:
        self.score = 0
        self.dice: list[int] = self._get_fresh_dice(Game.NUM_DICE)
        self.holds: dict[int, int] = self._get_fresh_holds()
        self.actions: list[ActionT] = []

    def _get_fresh_dice(self, num_dice: int) -> list[int]:
        return [-1 for _ in range(num_dice)]

    def _get_fresh_holds(self) -> dict[int, int]:
        return {n: 0 for n in range(1, Game.NUM_DICE + 1)}

    @property
    def num_dice(self) -> int:
        return len(self.dice)

    @property
    def state(self) -> StateT:
        return self.to_state()

    @property
    def stopped(self) -> bool:
        return self.score > 0 and sum(self.holds.values()) == 0

    @property
    def busted(self) -> bool:
        # We rolled and got no pointers
        return (
            sum(self.state["rolled"].values()) > 0 and self.calc_score(self.dice) == 0
        )

    @property
    def filled(self) -> bool:
        return self.num_dice == 0

    @property
    def num_ones_rolled(self) -> int:
        return self.dice.count(1)

    @property
    def num_ones_held(self) -> int:
        return self.holds[1]

    @property
    def num_twos_rolled(self) -> int:
        return self.dice.count(2)

    @property
    def num_twos_held(self) -> int:
        return self.holds[2]

    @property
    def num_threes_rolled(self) -> int:
        return self.dice.count(3)

    @property
    def num_threes_held(self) -> int:
        return self.holds[3]

    @property
    def num_fours_rolled(self) -> int:
        return self.dice.count(4)

    @property
    def num_fours_held(self) -> int:
        return self.holds[4]

    @property
    def num_fives_rolled(self) -> int:
        return self.dice.count(5)

    @property
    def num_fives_held(self) -> int:
        return self.holds[5]

    @property
    def num_sixes_rolled(self) -> int:
        return self.dice.count(6)

    @property
    def num_sixes_held(self) -> int:
        return self.holds[6]

    def to_state(self) -> StateT:
        # Calculate score if you stopped now
        hold_score = 0
        dice_from_holds: list[int] = []
        for die, hold_count in self.holds.items():
            dice_from_holds.extend([die for _ in range(hold_count)])
        hold_score += self.calc_score(dice_from_holds)

        return {
            "hold_score": hold_score,
            "final_score": self.score,
            "num_dice": self.num_dice,
            "rolled": {
                1: self.num_ones_rolled,
                2: self.num_twos_rolled,
                3: self.num_threes_rolled,
                4: self.num_fours_rolled,
                5: self.num_fives_rolled,
                6: self.num_sixes_rolled,
            },
            "held": {
                1: self.num_ones_held,
                2: self.num_twos_held,
                3: self.num_threes_held,
                4: self.num_fours_held,
                5: self.num_fives_held,
                6: self.num_sixes_held,
            },
        }

    def roll_dice(self, num_dice: int) -> None:
        self.dice = sorted(random.randint(1, 6) for _ in range(num_dice))

    @classmethod
    def calc_score(cls, dice: list[int]) -> int:
        return 100 * dice.count(1) + 50 * dice.count(5)

    def handle_roll(self) -> None:
        self.actions.append("roll")
        self.roll_dice(self.num_dice)
        if self.busted:
            self.handle_bust()

    def handle_hold(self, hold_action: ActionT) -> None:
        self.actions.append(hold_action)
        assert hold_action is not None
        new_holds: dict[int, int] = {}
        for die, hold_count in enumerate(hold_action.split("_")[-1], start=1):
            new_holds[die] = int(hold_count)

        assert new_holds[1] <= self.num_ones_rolled, (
            new_holds[1],
            self.num_ones_rolled,
        )
        assert new_holds[5] <= self.num_fives_rolled, (
            new_holds[5],
            self.num_fives_rolled,
        )

        # Add the new held dice from this roll to the turn's held dice
        # Also remove the held dice from `self.dice`
        for die, hold_count in new_holds.items():
            if hold_count:
                self.holds[die] += hold_count
                for _ in range(hold_count):
                    self.dice.remove(die)

        # Reset dice to indicate that they're ready to be rolled
        self.dice = self._get_fresh_dice(len(self.dice))

    def handle_bust(self) -> None:
        self.score = 0

    def handle_stop(self) -> None:
        self.actions.append("stop")
        self.score += self.state["hold_score"]
        self.holds = self._get_fresh_holds()

    @property
    def turn_is_over(self) -> bool:
        return self.busted or self.stopped or self.filled


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
            f"{self.name.title()} Agent average score: {self.total_score // self.num_games_played:7.0f} | Median: {median:4.0f} | Best Score: {self.best_score}"
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
            best_action: ActionT | None = None
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
                    # reward = -100000000
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
            if evaluate and ((num_episode % (num_episodes / 50)) == 0):
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


if False and __name__ == "__main__":
    num_dice = 6
    num_games = 1000
    scores = []
    for n in range(num_games):
        game = Game()
        game.roll_dice(num_dice)
        scores.append(game.calc_score(game.dice))

    print(f"Average score: {sum(scores) / num_games}")


if __name__ == "__main__":
    print("Start!")

    # Train the smart agent
    smart = Smart(
        "smart",
        Q=QTable(
            learning_rate=0.1,
            discount_factor=0.8,
            epsilon=0.7,
        ),
    )

    agents: list[Agent] = [
        # smart,
        # Agent("dumb "),
        # RiskyAgent("risky"),
        # StopAtNDice("stop1", num_dice_to_stop_at=1),
        # StopAtNDice("stop2", num_dice_to_stop_at=2),
        # StopAtNDice("stop3", num_dice_to_stop_at=3),
        # StopAtNDice("stop4", num_dice_to_stop_at=4),
        # StopAtNDice("stop5", num_dice_to_stop_at=5),
        StopAtNDice("stop6", num_dice_to_stop_at=6),
    ]

    NUM_TRAIN_GAMES = 1_000
    if smart in agents:
        smart.train(num_episodes=NUM_TRAIN_GAMES, evaluate=False, metric="mean")

    # Evaluate the agents
    NUM_TEST_GAMES = 1_000
    for agent in agents:
        for n in range(NUM_TEST_GAMES):
            agent.play()

    for agent in agents:
        agent.print_performance()

    if smart.evaluations:
        smart.plot_evaluations()

    if NUM_TRAIN_GAMES >= 1_000_000:
        fn = f"./dice/trained_agent_{smart.num_games_played}.pkl"
        with open(fn, "wb") as fout:
            print(f"Saved trained agent to '{fn}'")
            pickle.dump(smart, fout)
