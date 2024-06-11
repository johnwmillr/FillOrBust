# https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
import random
from collections import defaultdict
from itertools import product

from dice.constants import NUM_DICE_IN_GAME
from dice.types import ActionT, DiceCounts, Observation, StepInfo


def determine_available_actions(obs: Observation) -> list[ActionT]:
    """Determine what actions are available based on the current state

    Returns:
        list[ActionT]: A list of possible actions.
    """
    options: list[ActionT] = []

    if obs.inheriting and obs.card is not None and sum(obs.dice) > 0:
        return [("inherit", False), ("inherit", True)]

    if obs.card is None:
        # No card, must draw
        return [("draw", None)]

    if obs.num_dice_remaining == 0:
        # Filled, must either draw or stop
        return [("draw", None), ("stop", None)]

    # Either we haven't rolled at all, or we've rolled and just held dice
    if sum(obs.dice) == 0 or sum(obs.dice) > obs.num_dice_remaining:
        # No just-rolled dice waiting to be held, could roll
        options.append(("roll", None))
        if (
            obs.hold_score > 0
            and obs.card.category != "must_bust"
            and not obs.card.must_fill
            and not obs.inheriting
        ):
            # Have held some dice, could stop
            options.append(("stop", None))
        return options

    # We just rolled, need to hold some dice
    possible_holds = determine_hold_options_from_dice(obs.dice)
    options.extend(("hold", hold) for hold in possible_holds)
    return options


def determine_hold_options_from_dice(dice: DiceCounts) -> list[DiceCounts]:
    if sum(dice) == 0:
        return [DiceCounts()]

    # Check for a straight
    if dice == DiceCounts(1, 1, 1, 1, 1, 1):
        return [DiceCounts(1, 1, 1, 1, 1, 1)]

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
    holds = product(*options.values())
    holds = set(
        DiceCounts(*item) for item in holds if item != ((0,) * NUM_DICE_IN_GAME)
    )
    return sorted(holds)


class Agent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.game_score = 0
        self.turn_count = 0
        self.num_turns: list[int] = []
        self.rewards: list[float] = []
        self.game_scores: list[int] = []
        self.wins = 0

    def get_action(self, obs: Observation) -> ActionT:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # Explore (choose a random action)
        return random.choice(determine_available_actions(obs))

    def update(
        self,
        obs: Observation,
        action: ActionT,
        reward: float,
        terminated: bool,
        turn_is_over: bool,
        next_obs: Observation,
        info: StepInfo | None = None,
    ):
        if info is None:
            return

        # Update the agent's training info
        self.num_turns.append(info.num_turns)
        self.rewards.append(info.reward)
        self.game_scores.append(info.game_score)


class StopAtNDiceAgent(Agent):
    def __init__(self, name: str, num_dice_to_stop_at: int) -> None:
        """
        Initialize an agent that will stop rolling at a certain number of dice.
        """
        super().__init__(name)
        self.num_dice_to_stop_at = num_dice_to_stop_at

    def get_action(self, obs: Observation) -> ActionT:
        if obs.num_dice_remaining <= self.num_dice_to_stop_at:
            return ("stop", None)
        return super().get_action(obs)


class SmartAgent(Agent):
    def __init__(
        self,
        name: str,
        initial_learning_rate: float,
        learning_rate_decay: float,
        final_learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        super().__init__(name)
        self.q_values: dict[Observation, dict[ActionT, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.lr = initial_learning_rate
        self.lr_decay = learning_rate_decay
        self.final_lr = final_learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error: list[float] = []

    def get_action(self, obs: Observation) -> ActionT:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if obs in self.q_values and random.random() > self.epsilon:
            # Exploit (choose the action with the best reward)
            qvalues = self.q_values[obs]
            max_value = 0.0
            best_action = None
            for action, value in qvalues.items():
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
        action: ActionT,
        reward: float,
        terminated: bool,
        turn_is_over: bool,
        next_obs: Observation,
        info: StepInfo | None = None,
    ):
        """Updates an action's Q-value."""
        super().update(obs, action, reward, terminated, turn_is_over, next_obs, info)

        current_q_value = self.q_values[obs][action]
        future_q_value = (not terminated) * max(
            self.q_values[next_obs].values(), default=0.0
        )
        temporal_difference = (
            reward + self.discount_factor * future_q_value - current_q_value
        )

        # Update the Q-value for this action
        self.q_values[obs][action] = current_q_value + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def decay_learning_rate(self) -> None:
        self.lr = max(self.final_lr, self.lr - self.lr_decay)
