from dice.types import ActionT, StateT, QTableT


class UnknownActionError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


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

    @property
    def empty_actions(self) -> dict[ActionT, float]:
        return {"roll": 0, "stop": 0}

    def state_to_Q_table_keys(self, state: StateT) -> tuple[str, ...]:
        score_key = "score_" + str(state["game_score"])
        roll_key = "roll_count_" + str(state["rolled"])
        hold_key = "hold_count_" + str(state["held"])
        return (score_key, roll_key, hold_key)

    def state_in_table(self, state: StateT) -> bool:
        score_key, roll_key, hold_key = self.state_to_Q_table_keys(state)
        # if score_key in self.table:
        if roll_key in self.table:
            if hold_key in self.table[roll_key]:
                return True
        return False

    def get_Q_values_for_state(self, state: StateT) -> dict[ActionT, float]:
        if not self.state_in_table(state):
            return {}
        score_key, roll_key, hold_key = self.state_to_Q_table_keys(state)
        return self.table[roll_key][hold_key]

    def assign_value_to_state_and_action(
        self, state: StateT, action: ActionT, value: float = 0.0, overwrite: bool = True
    ) -> None:
        score_key, roll_key, hold_key = self.state_to_Q_table_keys(state)

        # if score_key not in self.table:
        #     self.table[score_key] = {}
        if roll_key not in self.table:
            self.table[roll_key] = {}
        if hold_key not in self.table[roll_key]:
            self.table[roll_key][hold_key] = {}
        if overwrite or (action not in self.table[roll_key][hold_key]):
            self.table[roll_key][hold_key][action] = value

    def update(
        self, current_state: StateT, action: ActionT, new_state: StateT, reward: float
    ) -> None:
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
        # assert action in self.empty_actions, UnknownActionError(action)

        # Check if we have reward values for `state` and `new_state`
        self.assign_value_to_state_and_action(current_state, action, overwrite=False)


        # Calculate the new Q-value for this action/state combo
        current_q_value = self.get_Q_values_for_state(current_state)[action]
        max_next_q_value = max(
            self.get_Q_values_for_state(new_state).values(), default=0
        )
        updated_q_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * max_next_q_value - current_q_value
        )
        self.assign_value_to_state_and_action(current_state, action, updated_q_value)
