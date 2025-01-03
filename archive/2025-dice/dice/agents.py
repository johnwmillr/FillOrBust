from collections import defaultdict
import random

import pandas as pd

from dice.types import Action, Observation, Terminated, TrainingStats


def determine_available_actions(obs: Observation) -> list[Action]:
    """Determines the available actions based on the current state of the game."""
    if obs.num_dice_remaining == 0 or obs.num_dice_remaining == 6:  # New game or busted
        return ["roll"]
    return ["roll", "stop"]  # Rolled some pointers


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self._training_stats: list[TrainingStats] = []
        self.training_stats: pd.DataFrame | None = None

    def __repr__(self) -> str:
        return f"<Agent: {self.name}>"

    def get_action(self, obs: Observation) -> Action:
        """
        Returns a random action from the list of available actions.
        """
        # Explore (choose a random action)
        return random.choice(determine_available_actions(obs))

    def update(
        self,
        obs: Observation,
        action: Action,
        reward: float,
        terminated: Terminated,
        next_obs: Observation,
        stats: TrainingStats | None = None,
    ):
        if stats is None:
            return

        # Update the agent's training stats
        self._training_stats.append(stats)


class SmartAgent(Agent):
    def __init__(
        self,
        name: str,
        learning_rate: float | None,
        epsilon: float | None,
        discount_factor: float | None,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            epsilon: The epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        super().__init__(name)
        self.q_values: dict[Observation, dict[Action, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.lr = learning_rate if learning_rate is not None else 0.001
        self.epsilon = epsilon if epsilon is not None else 1.0
        self.discount_factor = discount_factor if discount_factor is not None else 0.9
        self.training_error: list[float] = []

    def __repr__(self) -> str:
        return f"<SmartAgent: {self.name}>"

    def get_action(self, obs: Observation) -> Action:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if obs in self.q_values and random.random() > self.epsilon:
            # Exploit (choose the action with the best reward)
            q_values = self.q_values[obs]
            max_value = 0.0
            best_action = None
            for action, value in q_values.items():
                if value > max_value:
                    max_value = value
                    best_action = action

            if best_action:
                return best_action

        # Explore (choose a random action)
        return super().get_action(obs)

    def update(
        self,
        obs: Observation,
        action: Action,
        reward: float,
        terminated: bool,
        next_obs: Observation,
        stats: TrainingStats | None = None,
    ):
        """Updates an action's Q-value."""
        super().update(obs, action, reward, terminated, next_obs, stats)

        current_q_value = self.q_values[obs][action]
        future_q_value = (not terminated) * max(
            self.q_values[next_obs].values(), default=0.0
        )
        temporal_difference = (
            reward + self.discount_factor * future_q_value - current_q_value
        )

        # Update the Q-value for this action
        self.q_values[obs][action] = current_q_value + self.lr * temporal_difference
        if reward:
            self.training_error.append(temporal_difference)
