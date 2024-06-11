import random

from dice.agents import Agent
from dice.cards import Card, Deck
from dice.constants import NUM_DICE_IN_GAME
from dice.types import ActionT, DiceCounts, Observation, StepInfo


deck = Deck()


def draw_card() -> Card:
    return deck.draw()


def calc_score(dice: DiceCounts) -> int:
    assert len(dice), "Can't score empty dice"
    TRIPLE_VALUES = {1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}

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


def roll_dice(num_dice: int) -> DiceCounts:
    dice = list(random.randint(1, 6) for _ in range(num_dice))
    return DiceCounts(*(dice.count(die) for die in range(1, 7)))


def hold_dice(current_holds: DiceCounts, new_holds: DiceCounts) -> DiceCounts:
    return DiceCounts(
        *(old_hold + new_hold for old_hold, new_hold in zip(current_holds, new_holds))
    )


def is_bust(dice: DiceCounts) -> bool:
    return dice[0] == 0 and dice[4] == 0 and max(dice) < 3


def is_fill(held_dice: DiceCounts) -> bool:
    return sum(held_dice) == NUM_DICE_IN_GAME


def sigmoid(x: float) -> float:
    return 1 / (1 + 2.71828**-x)


def convert_num_turns_to_reward(num_turns: int) -> float:
    return 1000 / num_turns
    # return -sigmoid((num_turns - 50) / 10)

    if num_turns > 50:
        return 0
    elif num_turns > 45:
        return 0.5
    else:
        return 1

    reward = 1000 / num_turns - 18
    return min(max(reward, -3), 15)
    max_value = 85
    min_value = 55
    scaled_num_turns = -1 * (
        ((num_turns - min_value) / (max_value - min_value)) * 2 - 1
    )
    return min(max(-1.5, scaled_num_turns), 2)


# https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
class FillOrBustEnvironment:
    def __init__(self, score_needed_to_win: int = 10_000) -> None:
        self.score_needed_to_win = score_needed_to_win
        self.turn_is_over = False
        self.turn_scores = []
        self.player: Agent | None = None
        self.reset_for_new_game()

    def step(
        self, agent: Agent, action: ActionT
    ) -> tuple[Observation, float, bool, bool, bool, StepInfo]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.

        Args:
            action (ActionT): an action provided by the agent

        Returns:
            observation: this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward: The amount of reward returned as a result of taking the action.
            terminated: whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated: whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a time limit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            turn_is_over: whether the turn is over
            info: `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
        """
        # Update environment variables for this new step
        terminated, truncated, self.turn_is_over = False, False, False
        self.player = agent
        reward = 0

        # Respond to the action taken
        if action[0] == "draw" or action[0] == "inherit":
            if action[0] == "inherit":
                if action[1] is False:  # Decline the inheritance
                    self.card = None
                    self.hold_score = 0
                    self.dice = DiceCounts()
                    self.holds = DiceCounts()
                    self.inheritance_available = False

            self.card = draw_card()
            if self.card.category == "no_dice":
                self.reset_for_new_turn()  # XXX: This hides the no dice card from ever being observed—is that okay?
            self.dice = DiceCounts()  # Kludge for inheritance
        elif action[0] == "roll":
            self.dice = roll_dice(NUM_DICE_IN_GAME - sum(self.holds))
            if is_bust(self.dice):
                if self.card is not None and self.card.category == "must_bust":
                    self.player.game_score += self.hold_score
                else:
                    self.hold_score = 0
                self.reset_for_new_turn()
            self.inheritance_available = False
        elif action[0] == "hold":
            assert isinstance(action[1], DiceCounts)

            # Score holds as we go to keep track of triples
            self.hold_score += calc_score(action[1])
            self.holds = hold_dice(self.holds, action[1])
            if is_fill(self.holds):
                if self.card is not None:
                    self.hold_score += self.card.bonus
                self.reset_dice_and_holds_and_cards()
        elif action[0] == "stop":
            self.player.game_score += self.hold_score
            self.reset_for_new_turn(
                inheritance=False
            )  # Set to True for MussBuss inheritance

        # Give a reward if we've reached the max score
        if self.player.game_score >= self.score_needed_to_win:
            reward = convert_num_turns_to_reward(self.player.turn_count)
            # reward = 1
            terminated = True

        # Save some training info
        info = StepInfo(
            num_turns=self.player.turn_count,
            reward=reward,
            game_score=self.player.game_score,
        )
        return self._get_obs(), reward, terminated, truncated, self.turn_is_over, info

    def reset_dice_and_holds_and_cards(self) -> None:
        self.dice = DiceCounts()
        self.holds = DiceCounts()
        self.card = None

    def reset_for_new_turn(self, inheritance: bool = False) -> None:
        # Set `inheritance` to `True` for Muss Buss inheritance
        self.turn_is_over = True
        if self.player is not None:
            self.player.turn_count += 1
        self.turn_scores.append(self.hold_score)
        if inheritance and self.hold_score > 0:
            self.inheritance_available = True
        else:
            self.inheritance_available = False
            self.hold_score = 0
            self.reset_dice_and_holds_and_cards()

    def reset_for_new_game(self) -> tuple[Observation, None]:
        self.hold_score = 0
        self.reset_for_new_turn()
        self.player = None
        return self._get_obs(), None

    def _get_obs(self) -> Observation:
        return Observation(
            self.hold_score,
            NUM_DICE_IN_GAME - sum(self.holds),
            self.dice,
            self.card,
            self.inheritance_available,
        )
