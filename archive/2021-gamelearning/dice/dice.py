import random
from typing import Literal


ActionT = Literal["roll", "stop"]
StateT = tuple[int, tuple[int, ...]]
QTableT = dict[StateT, dict[ActionT, float]]


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
        self.table[state][action] += learning_rate * (
            reward
            + discount_factor * max(self.table[new_state].values())
            - self.table[state][action]
        )


def _initial_state() -> StateT:
    return (0, ())


def roll() -> int:
    return random.randint(1, 6)


def calc_score(dice: list[int]) -> int:
    return 100 * dice.count(1) + 50 * dice.count(5)


def update_state(current_state: StateT, die: int) -> StateT:
    current_score, rolls = current_state
    roll_score = calc_score([die])
    if roll_score == 0:
        return _initial_state()  # Bust

    current_score += roll_score
    rolls += (die,)
    if len(rolls) > 3:  # Only track three most recent rolls
        rolls = rolls[1:]
    return (current_score, rolls)


class Agent:
    def __init__(self, name: str, Q: QTable | None) -> None:
        self.name = name
        self.Q = Q
        self.score = 0

    def choose_action(self, state: StateT) -> str:
        return random.choice(["roll", "stop"])

    def play(self) -> None:
        state = _initial_state()  # Starting state for each new episode (i.e. new game)
        keep_rolling = True

        while keep_rolling:
            action = self.choose_action(state)
            if action == "roll":
                state = update_state(state, roll())
            else:
                keep_rolling = False
        self.score += state[0]

    def __repr__(self) -> str:
        return f"<Agent: '{self.name}', score: {self.score}>"


class Smart(Agent):
    def choose_action(self, state: StateT) -> str:
        assert self.Q is not None, f"{self.name}'s Q Table is empty."
        if state in self.Q.table:
            max_reward = -1.0
            best_action = ""
            for action, reward in self.Q.table[state].items():
                if reward > max_reward:
                    max_reward = reward
                    best_action = action
            return best_action
        return random.choice(["roll", "stop"])


"""
Q Table

| States  |     Actions     |
| State   |   Roll |   Stop |
|---------|--------|--------|
| (0, []) |   ?    |    ?   |
| (50, [5]) |  ?   |    ?   |
|
"""

if __name__ == "__main__":
    # Q-learning parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.1
    num_episodes: int = 1_000_000

    def update_Q(
        Q: QTableT, state: StateT, action: ActionT, new_state: StateT, reward: float
    ):
        if state not in Q:
            Q[state] = {"roll": 0, "stop": 0}
        if new_state not in Q:
            Q[new_state] = {"roll": 0, "stop": 0}
        Q[state][action] += learning_rate * (
            reward + discount_factor * max(Q[new_state].values()) - Q[state][action]
        )
        return Q

    # Initialize Q-table
    Q: QTableT = {}

    # Q-learning loop
    for episode in range(num_episodes):
        state = _initial_state()  # Starting state for each episode
        keep_rolling = True

        while keep_rolling:
            action: ActionT
            if random.random() < epsilon or state not in Q:
                action = random.choice(["roll", "stop"])  # Explore
            else:
                action = (
                    "roll" if Q[state]["roll"] > Q[state]["stop"] else "stop"
                )  # Exploit

            if action == "roll":
                new_state = update_state(state, roll())
                Q = update_Q(Q, state, action, new_state, 0)
                if new_state == _initial_state():
                    keep_rolling = False  # Bust
            else:
                new_state = state
                reward = new_state[0]
                Q = update_Q(Q, state, action, new_state, reward)
                keep_rolling = False

            state = new_state

    # The trained Q-table will contain learned values for state-action pairs
    print("Done training.")

    NUM_GAMES = 1000
    dumb = Agent("dumb")
    smart = Smart("smart")
    smart.train(num_episodes=num_episodes)

    for n in range(NUM_GAMES):
        dumb.play()
        smart.play()
    print(f"{dumb.name.title()} Agent average score: {dumb.score / NUM_GAMES}")
    print(f"{smart.name.title()} Agent average score: {smart.score / NUM_GAMES}")
