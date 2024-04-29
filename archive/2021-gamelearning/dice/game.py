from collections import defaultdict
from itertools import product
import random

from dice.types import ActionT, DiceCounts, Observation


NUM_DICE_IN_GAME = 6
TRIPLE_VALUES = {1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}


def calc_score(dice: DiceCounts) -> int:
    assert len(dice), "Can't score empty dice"

    # Check for a straight
    if dice == (1, 1, 1, 1, 1, 1):
        return 1500

    # Check for triples and singles
    score = 0
    for die, count in enumerate(dice, start=1):
        if count == 6:
            return 2 * TRIPLE_VALUES[die]
        elif count >= 3:
            score += TRIPLE_VALUES[die]

        # Check for any single pointers
        if die == 1:
            score += 100 * (count % 3)
        elif die == 5:
            score += 50 * (count % 3)
    return score


def determine_available_actions(obs: Observation) -> list[ActionT]:
    """Determine what actions are available based on the current state

    Returns:
        list[ActionT]: A list of possible actions.
    """
    options: list[ActionT] = []

    # No rolled dice to act on
    if sum(obs.dice) == 0:
        # No just-rolled dice, could roll
        options.append(("roll", None))
        if obs.hold_score > 0:
            # Have held some dice, could stop
            options.append(("stop", None))
        return options

    # We just rolled, need to hold some dice
    possible_holds = determine_hold_options_from_dice(obs.dice)
    options.extend(("hold", hold) for hold in possible_holds)
    return options


def determine_hold_options_from_dice(dice: DiceCounts) -> list[DiceCounts]:
    if sum(dice) == 0:
        return [DiceCounts((0, 0, 0, 0, 0, 0))]

    # Check for a straight
    if dice == DiceCounts((1, 1, 1, 1, 1, 1)):
        return [DiceCounts((1, 1, 1, 1, 1, 1))]

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

    # Translate dice options to "hold_" strings
    holds = product(*options.values())
    holds = set(DiceCounts(item) for item in holds if item != ((0,) * NUM_DICE_IN_GAME))
    return sorted(holds)


def roll_dice(num_dice: int) -> DiceCounts:
    dice = list(random.randint(1, 6) for _ in range(num_dice))
    return DiceCounts(dice.count(die) for die in range(1, 7))  # type: ignore


def hold_dice(current_holds, new_holds: DiceCounts) -> DiceCounts:
    return DiceCounts(
        old_hold + new_hold for old_hold, new_hold in zip(current_holds, new_holds)
    )


def is_bust(dice: DiceCounts) -> bool:
    return dice[0] == 0 and dice[4] == 0 and max(dice) < 3


def is_fill(held_dice: DiceCounts) -> bool:
    return sum(held_dice) == NUM_DICE_IN_GAME


# https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
class FillOrBustEnvironment:
    def __init__(self, score_needed_to_win: int = 10_000) -> None:
        self.score_needed_to_win = score_needed_to_win
        self._steps = []
        self.num_turns_to_win = []
        self.rewards = []
        self.turn_scores = []
        self.turn_lengths = []
        self._reset_for_new_game()

    def step(self, action: ActionT) -> tuple[Observation, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.

        Args:
            action (ActType): an action provided by the agent

        Returns:
            observation: this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward: The amount of reward returned as a result of taking the action.
            terminated: whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated: whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a time limit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info: `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
        """
        self._steps.append((self._get_obs(), action))
        reward = 0
        terminated, truncated = False, False
        if action[0] == "roll":
            self.dice = roll_dice(NUM_DICE_IN_GAME - sum(self.holds))
            if is_bust(self.dice):
                self.reset_for_new_turn()
        elif action[0] == "hold":
            assert isinstance(action[1], DiceCounts)
            assert all(hold <= die for hold, die in zip(action[1], self.dice))
            self.dice = DiceCounts((0, 0, 0, 0, 0, 0))

            # Score holds as we go to keep track of triples
            self.hold_score += calc_score(action[1])
            self.holds = hold_dice(self.holds, action[1])
            if is_fill(self.holds):
                self.reset_dice_and_holds()
        elif action[0] == "stop":
            self.game_score += self.hold_score
            self.reset_for_new_turn()

        # Exit early if we've taken too many turns
        if self.num_turns > 60:
            truncated = True
            # reward = -10
            self.num_turns_to_win.append(None)

        # Give a reward if we've reached the max score
        if self.game_score >= self.score_needed_to_win:
            reward = 10000 / self.num_turns**2
            terminated = True
            self.num_turns_to_win.append(self.num_turns)

        self.rewards.append(reward)
        self.turn_length += 1
        return self._get_obs(), reward, terminated, truncated, {}

    def reset_dice_and_holds(self) -> None:
        self.dice = DiceCounts((0, 0, 0, 0, 0, 0))
        self.holds = DiceCounts((0, 0, 0, 0, 0, 0))
        self.hold_score = 0
        self.turn_length = 0

    def reset_for_new_turn(self) -> None:
        self.num_turns += 1
        self.turn_scores.append(self.hold_score)
        self.turn_lengths.append(self.turn_length)
        self.reset_dice_and_holds()

    def _reset_for_new_game(self) -> None:
        self.hold_score = 0
        self.turn_length = 0
        self.game_score = 0
        self.num_turns = 0
        self.reset_for_new_turn()
        self._steps.append("NEW GAME")

    def reset_for_new_game(self) -> tuple[Observation, dict]:
        self._reset_for_new_game()
        return self._get_obs(), {}

    def _get_obs(self) -> Observation:
        return Observation(
            self.hold_score,
            self.dice,
        )
