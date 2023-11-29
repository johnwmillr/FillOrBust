from copy import deepcopy
import random

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from dice_types import ActionT, StateT, QTableT, VALID_HOLDS


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
        return {action: 0 for action in VALID_HOLDS + ("roll", "stop")}

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
        assert action in self.empty_actions, UnknownActionError(action)

        if state not in self.table:
            self.table[state] = self.empty_actions
        if new_state not in self.table:
            self.table[new_state] = self.empty_actions
        self.table[state][action] += self.learning_rate * (
            reward
            + self.discount_factor * max(self.table[new_state].values())
            - self.table[state][action]
        )


class Game:
    def __init__(self) -> None:
        self.score = 0
        self.num_dice = 6
        self.num_ones_rolled = -1
        self.num_fives_rolled = -1
        self.num_ones_held = 0
        self.num_fives_held = 0
        self.stopped = False
        self.actions: list[ActionT] = []

    @property
    def state(self) -> StateT:
        return self.to_state()

    @property
    def busted(self) -> bool:
        # We rolled and got no pointers
        return self.num_ones_rolled == 0 and self.num_fives_rolled == 0

    # @property
    # def stopped(self) -> bool:
    #     # We held dice and chose STOP?
    #     return (self.num_ones_held > 0 or self.num_fives_held > 0)

    def to_state(self) -> StateT:
        return (
            self.score,
            self.num_dice,
            self.num_ones_rolled,
            self.num_fives_rolled,
            self.num_ones_held,
            self.num_fives_held,
        )

    def roll_dice(self, num_dice: int) -> tuple[int, ...]:
        dice = tuple(random.randint(1, 6) for _ in range(num_dice))
        return dice

    def calc_score(self, dice: tuple[int, ...]) -> int:
        return 100 * dice.count(1) + 50 * dice.count(5)

    def handle_bust(self) -> None:
        # self.busted = True
        self.score = 0
        return

    def handle_roll(self) -> None:
        self.actions.append("roll")
        dice = self.roll_dice(self.num_dice)
        self.num_ones_rolled = dice.count(1)
        self.num_fives_rolled = dice.count(5)

        roll_score = self.calc_score(dice)
        if roll_score == 0:
            self.handle_bust()

    def handle_stop(self) -> None:
        # assert self.num_ones_held > 0 or self.num_fives_held > 0, (
        #     self.num_ones_held,
        #     self.num_fives_held,
        # )
        self.actions.append("stop")
        self.score += self.calc_score(
            self.num_ones_held * (1,) + self.num_fives_held * (5,)
        )
        self.num_ones_held = 0
        self.num_fives_held = 0
        self.stopped = True

    def handle_hold(self, hold_action: ActionT) -> None:
        holds: tuple[int, int] = tuple(map(int, hold_action.split("_")[-1]))  # type: ignore
        assert holds[0] <= self.num_ones_rolled, (holds[0], self.num_ones_rolled)
        assert holds[1] <= self.num_fives_rolled, (holds[1], self.num_fives_rolled)
        self.actions.append(hold_action)

        self.num_dice -= sum(holds)
        self.num_ones_held += holds[0]
        self.num_fives_held += holds[1]
        self.num_ones_rolled = -1
        self.num_fives_rolled = -1

    @property
    def game_is_over(self) -> bool:
        return self.busted or self.stopped


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.score = 0
        self.best_score = -1
        self.num_games_played = 0

    def determine_available_actions(self, state: StateT) -> tuple[ActionT, ...]:
        options = ()
        if state[2] == 0 and state[3] == 0:  # Bust
            return (None,)
        if state[2] == -1 and state[3] == -1:  # Unrolled dice
            options = ("roll",)
            if state[4] or state[5]:
                options += ("stop",)
            return options
        if state[2] > 0 or state[3] > 0:  # Rolled dice with pointers, ready to hold
            hold_options: tuple[ActionT, ...] = ()
            for ones in range(state[2] + 1):
                for fives in range(state[3] + 1):
                    if ones or fives:
                        hold_options += (f"hold_{ones}{fives}",)  # type: ignore
            return hold_options
        raise Exception(f"Unhandled state: {state=}")

    def choose_action(self, state: StateT) -> ActionT:
        return random.choice(self.determine_available_actions(state))

    def play(self) -> None:
        game = Game()  # Empty state for each new episode (i.e. new game)
        while not game.game_is_over:
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
        self.score += game.state[0]
        if game.state[0] > self.best_score:
            self.best_score = game.state[0]
        self.num_games_played += 1

    def print_performance(self) -> None:
        print(
            f"{self.name.title()} Agent average score: {self.score / self.num_games_played:7.2f} | Best Score: {self.best_score}"
        )

    def __repr__(self) -> str:
        return f"<Agent: '{self.name}', score: {self.score}>"


class RiskyAgent(Agent):
    def choose_action(self, state: StateT) -> ActionT:
        possible_actions = self.determine_available_actions(state)
        if "roll" in possible_actions:
            if state[0] < 300:
                return "roll"
        return super().choose_action(state)


class Smart(Agent):
    def __init__(self, name: str, Q: QTable | None = None) -> None:
        super().__init__(name)
        self.Q = QTable() if Q is None else Q
        self.evaluations: list[tuple[int, float]] = []

    def choose_action(self, state: StateT) -> ActionT:
        assert self.Q is not None, f"{self.name}'s Q Table is empty."
        if state in self.Q.table:
            max_reward = -1.0
            best_action: ActionT | None = None
            for action, reward in self.Q.table[state].items():
                if reward > max_reward:
                    max_reward = reward
                    best_action = action
            if best_action and best_action in self.determine_available_actions(state):
                return best_action
        return super().choose_action(state)

    def train(self, num_episodes: int = 1_000, evaluate: bool = False) -> None:
        for num_episode in range(num_episodes):
            game = Game()
            while not game.game_is_over:
                action: ActionT
                if random.random() < self.Q.epsilon or game.state not in self.Q.table:
                    action = super().choose_action(game.state)  # Explore
                else:
                    action = self.choose_action(game.state)  # Exploit

                game.state
                self.determine_available_actions(game.state)

                reward = 0
                previous_state = game.state
                if action is None:
                    game.handle_bust()
                elif action == "roll":
                    game.handle_roll()
                elif action.startswith("hold"):
                    game.handle_hold(action)
                elif action == "stop":
                    game.handle_stop()
                    reward = game.state[0]
                else:
                    raise UnknownActionError(action)
                self.Q.update(previous_state, action, game.state, reward)
            if evaluate and ((num_episode % 1000) == 0):
                self.evaluations.append(
                    (num_episode, self.evaluate(num_episode=num_episode))
                )
        print(f"{self.name.title()} Agent is done training.")

    def evaluate(self, num_games: int = 1_000, num_episode: int | None = None) -> float:
        agent = Smart(
            f"eval_{num_episode}" if num_episode else "eval", Q=deepcopy(self.Q)
        )
        for _ in range(num_games):
            agent.play()
        return agent.score / agent.num_games_played

    def plot_evaluations(self) -> None:
        if not len(self.evaluations):
            print("No evaluations to plot.")
            return
        _, ax = plt.subplots(figsize=[5, 3])
        assert isinstance(ax, Axes)
        x, y = zip(*self.evaluations)
        ax.plot(x, y, "ko-", label=self.Q, alpha=0.6)
        ax.legend()
        plt.show()


if __name__ == "__main__":
    dumb = Agent("dumb ")
    risky = RiskyAgent("risky")

    # Train the smart agent
    smart = Smart(
        "smart",
        Q=QTable(
            learning_rate=0.2,
            discount_factor=1.0,
            epsilon=0.5,
        ),
    )
    NUM_TRAIN_GAMES = 10_000
    smart.train(num_episodes=NUM_TRAIN_GAMES, evaluate=False)

    # Evaluate the agents
    agents: list[Agent] = [smart, dumb, risky]
    NUM_TEST_GAMES = 10_000
    for n in range(NUM_TEST_GAMES):
        for agent in agents:
            agent.play()

    for agent in agents:
        agent.print_performance()

    smart.plot_evaluations()
    print(f"{len(smart.Q.table)=}")
