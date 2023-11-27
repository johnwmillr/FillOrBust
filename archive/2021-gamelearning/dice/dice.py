import random
from typing import Literal


ActionT = Literal["roll", "stop"]
StateT = int
QTableT = dict[StateT, dict[ActionT, float]]


"""
Q Table

| States  |     Actions     |
| State   |   Roll |   Stop |
|---------|--------|--------|
| (0, []) |   ?    |    ?   |
| (50, [5]) |  ?   |    ?   |
|
"""


class QTable:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.table: QTableT = {}

    def update(
        self, state: StateT, action: ActionT, new_state: StateT, reward: float
    ) -> None:
        if state not in self.table:
            self.table[state] = {"roll": 0, "stop": 0}
        if new_state not in self.table:
            self.table[new_state] = {"roll": 0, "stop": 0}
        self.table[state][action] += self.learning_rate * (
            reward
            + self.discount_factor * max(self.table[new_state].values())
            - self.table[state][action]
        )


def roll() -> int:
    return random.randint(1, 6)


def calc_score(dice: list[int]) -> int:
    return 100 * dice.count(1) + 50 * dice.count(5)


class Game:
    def __init__(self) -> None:
        self.score = 0
        self.stopped = False
        self.busted = False

    @property
    def _initial_state(self) -> StateT:
        return 0

    @property
    def state(self) -> StateT:
        return self.to_state()

    def to_state(self) -> StateT:
        return self.score

    def handle_roll(self, die: int) -> None:
        roll_score = calc_score([die])
        if roll_score == 0:
            self.busted = True
            self.score = 0
            return

        self.score += roll_score

    def handle_stop(self) -> None:
        self.stopped = True

    @property
    def game_is_over(self) -> bool:
        return self.busted or self.stopped


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.score = 0

    def choose_action(self, state: StateT) -> ActionT:
        return random.choice(["roll", "stop"])

    def play(self) -> None:
        game = Game()  # Starting state for each new episode (i.e. new game)
        while not game.game_is_over:
            action = self.choose_action(game.state)
            if action == "roll":
                game.handle_roll(roll())
            else:
                game.handle_stop()
        self.score += game.state

    def __repr__(self) -> str:
        return f"<Agent: '{self.name}', score: {self.score}>"


class Smart(Agent):
    def __init__(self, name: str, Q: QTable | None = None) -> None:
        super().__init__(name)
        self.Q = QTable() if Q is None else Q

    def choose_action(self, state: StateT) -> ActionT:
        assert self.Q is not None, f"{self.name}'s Q Table is empty."
        if state in self.Q.table:
            max_reward = -1.0
            best_action: ActionT | None = None
            for action, reward in self.Q.table[state].items():
                if reward > max_reward:
                    max_reward = reward
                    best_action = action
            if best_action is not None:
                return best_action
        return random.choice(["roll", "stop"])

    def train(self, num_episodes: int = 1_000) -> None:
        for _ in range(num_episodes):
            game = Game()
            while not game.game_is_over:
                action: ActionT
                if random.random() < self.Q.epsilon or game.state not in self.Q.table:
                    action = random.choice(["roll", "stop"])  # Explore
                else:
                    action = self.choose_action(game.state)  # Exploit

                if action == "roll":
                    game.handle_roll(roll())
                    new_state = game.state
                    reward = 0
                else:
                    new_state = game.state
                    reward = new_state
                self.Q.update(game.state, action, new_state, reward)


class RiskyAgent(Agent):
    def choose_action(self, state: StateT) -> ActionT:
        if state < 500:
            return "roll"
        return "stop"


if __name__ == "__main__":
    smart = Smart(
        "smart",
        Q=QTable(
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.8,
        ),
    )
    smart.train(num_episodes=10_000)
    dumb = Agent("dumb")
    risky = RiskyAgent("risky")

    # Evaluate the agents
    NUM_TEST_GAMES = 100_000
    for n in range(NUM_TEST_GAMES):
        smart.play()
        dumb.play()
        risky.play()
    print(f"{smart.name.title()} Agent average score: {smart.score / NUM_TEST_GAMES}")
    print(f"{dumb.name.title()} Agent average score: {dumb.score / NUM_TEST_GAMES}")
    print(f"{risky.name.title()} Agent average score: {risky.score / NUM_TEST_GAMES}")
