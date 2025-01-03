from __future__ import annotations

import random
from collections import defaultdict
from itertools import product
from typing import TYPE_CHECKING

import pandas as pd

from dice.constants import NUM_DICE_IN_GAME
from dice.types import Action, DiceCounts, Observation, Terminated, TrainingStats

if TYPE_CHECKING:
    from dice.game import FillOrBust  # Only imported for type hints


def determine_available_actions(env: FillOrBust) -> list[Action]:
    """Determines the available actions based on the current state of the game."""

    # TODO: this should be a (class?) method of FillOrBust

    if sum(env.dice) == 0:
        if not any(env.held_dice):  # New game or busted
            return [("roll", None)]
        return [("roll", None), ("stop", None)]  # Have some held dice, can roll or stop
    if any(env.dice):  # Just rolled, must hold
        return [("hold", hold) for hold in determine_hold_options_from_dice(env.dice)]
    raise ValueError(f"Can't determine action for game state: {env._get_obs()}")


def determine_hold_options_from_dice(dice: DiceCounts) -> list[DiceCounts]:
    if sum(dice) == 0:
        return [DiceCounts()]

    # Check for a straight
    if dice == DiceCounts(1, 1, 1, 1, 1, 1):
        return [
            DiceCounts(fives=1),
            DiceCounts(1),
            DiceCounts(ones=1, fives=1),
            DiceCounts(1, 1, 1, 1, 1, 1),
        ]

    # Keep track of how many times a particular dice can be held
    options: defaultdict[int, tuple[int, ...]] = defaultdict(tuple[int])
    for n in range(1, NUM_DICE_IN_GAME + 1):
        options[n] = (0,)

    # Check for singles
    for die, count in enumerate(dice, start=1):
        # Check for triples
        if count >= 3:
            options[die] += (3,)
        if count == 6:
            options[die] += (6,)

        # Check for singles
        if count and die in [1, 5]:
            options[die] += tuple(range(count + 1))

    # Translate dice options to holds
    combos = product(*options.values())
    holds = set(
        DiceCounts(*item) for item in combos if item != ((0,) * NUM_DICE_IN_GAME)
    )
    return sorted(holds)


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self._training_stats: list[TrainingStats] = []
        self.training_stats: pd.DataFrame | None = None

    def __repr__(self) -> str:
        return f"<Agent: {self.name}>"

    def get_action(self, env: FillOrBust, obs: Observation) -> Action:
        """
        Returns a random action from the list of available actions.
        """
        # Explore (choose a random action)
        return random.choice(determine_available_actions(env))

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

    def get_action(self, env: FillOrBust, obs: Observation) -> Action:
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
        return super().get_action(env, obs)

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
