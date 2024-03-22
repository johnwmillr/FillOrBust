from dice.types import ActionT, QTableKeyT, State, QTableT


class UnknownActionError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def convert_state_to_Q_table_key(state: State) -> QTableKeyT:
    if isinstance(state, str):
        return state
    card = "card_" + str(state.card)
    discard = "discard_" + str(sorted(state.discard.elements()))
    banked_score = "banked_score_" + str(state.banked_score)
    turn_score = "turn_score_" + str(state.turn_score)
    dice = "roll_count_" + str(sorted(state.dice.elements()))
    holds = "hold_count_" + str(sorted(state.holds.elements()))
    return "+".join((card, discard, banked_score, turn_score, dice, holds))


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
            f"<QTable: {self.learning_rate=}, {self.discount_factor=}, {self.epsilon=}>"
        )

    def state_to_Q_table_key(self, state: State) -> QTableKeyT:
        return convert_state_to_Q_table_key(state)
        card = "card_" + str(state.card)
        # discard = "discard_" + str(list(map(str, sorted(state.discard))))
        # banked_score = "banked_score_" + str(state.banked_score)
        turn_score = "turn_score_" + str(state.turn_score)
        roll = "roll_count_" + str(sorted(state.dice.elements()))
        hold = "hold_count_" + str(sorted(state.holds.elements()))
        return (card, turn_score, roll, hold)

    def state_in_table(self, state: State) -> bool:
        return self.state_to_Q_table_key(state) in self.table

    def get_Q_values_for_state(self, state: State) -> dict[ActionT, float]:
        return self.table.get(self.state_to_Q_table_key(state), {})

    def assign_value_to_state_and_action(
        self, state: State, action: ActionT, value: float
    ) -> None:
        key = self.state_to_Q_table_key(state)
        if key not in self.table:
            self.table[key] = {}
        self.table[key][action] = value

    def update(
        self, current_state: State, action: ActionT, new_state: State, reward: float
    ) -> float:
        """
        Update the Q-table based on the observed state, action, new state, and reward.

        Args:
            current_state: The current state.
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
        current_q_value = self.get_Q_values_for_state(current_state).get(action, 0.0)
        max_next_q_value = max(
            self.get_Q_values_for_state(new_state).values(), default=0
        )
        updated_q_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * max_next_q_value - current_q_value
        )
        self.assign_value_to_state_and_action(current_state, action, updated_q_value)
        return updated_q_value
