import random

from dice.agents import Agent
from dice.cards import Card, Deck
from dice.constants import NUM_DICE_IN_GAME
from dice.types import ActionStr, ActionT, DiceCounts, Observation, TrainingInfo

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


def hold_dice(
    dice: DiceCounts, current_holds: DiceCounts, new_holds: DiceCounts
) -> tuple[DiceCounts, DiceCounts]:
    updated_dice = DiceCounts(
        *(die - new_hold for die, new_hold in zip(dice, new_holds))
    )
    updated_holds = DiceCounts(
        *(old_hold + new_hold for old_hold, new_hold in zip(current_holds, new_holds))
    )
    return updated_dice, updated_holds


def is_bust(dice: DiceCounts) -> bool:
    return dice[0] == 0 and dice[4] == 0 and max(dice) < 3


def is_fill(held_dice: DiceCounts) -> bool:
    return sum(held_dice) == NUM_DICE_IN_GAME


def convert_num_turns_to_reward(num_turns: int) -> float:
    # return 1 / num_turns
    if num_turns > 70:
        return -0.5
    elif num_turns > 60:
        return -0.25
    elif num_turns > 50:
        return 0
    elif num_turns > 45:
        return 0.25
    elif num_turns > 40:
        return 0.5
    elif num_turns > 35:
        return 0.75
    elif num_turns > 30:
        return 1
    return 1.5


class FillOrBust:

    SCORE_NEEDED_TO_WIN = 10_000
    current_player: Agent
    card: Card | None

    def __init__(self, players: list[Agent]) -> None:
        self.players = players
        self.reset_for_new_game()

    def step(
        self, action: ActionT
    ) -> tuple[Observation, float, bool, bool, bool, TrainingInfo]:
        turn_is_over, terminated, truncated = False, False, False
        reward = 0

        self.last_action: ActionStr | None = action[0]
        match action[0]:
            case "inherit":
                if action[1] is False:
                    self.inheritance = 0
                    self.dice = DiceCounts()
            case "draw":
                self.card = draw_card()
                if self.card.category == "no_dice":
                    self.current_player.hold_score = 0
                    self.inheritance = 0
                    self.card = None
                    turn_is_over = True
                    # self.reset_for_new_turn()  # XXX: This hides the no dice card from ever being observed—is that okay?
            case "roll":
                self.dice = roll_dice(NUM_DICE_IN_GAME - sum(self.held_dice))
                if is_bust(self.dice):
                    self.current_player.hold_score = 0
                    self.inheritance = 0
                    turn_is_over = True

                # Add the inheritance to the hold score (if there is any)
                self.current_player.hold_score += self.inheritance
                self.inheritance = 0
            case "hold":
                assert isinstance(action[1], DiceCounts)
                self.dice, self.held_dice = hold_dice(
                    self.dice, self.held_dice, action[1]
                )
                self.current_player.hold_score += calc_score(action[1])
                if is_fill(self.held_dice):
                    if self.card is not None:
                        self.current_player.hold_score += self.card.bonus
                    self.dice = DiceCounts()
                    self.held_dice = DiceCounts()
            case "stop":
                self.inheritance = self.current_player.hold_score
                self.current_player.game_score += self.current_player.hold_score
                self.current_player.hold_score = 0
                turn_is_over = True
            case _:
                raise ValueError(f"Invalid action: {action[0]}")

        # Check whether the game is over
        if self.current_player.game_score >= self.SCORE_NEEDED_TO_WIN:
            self.current_player.wins += 1
            reward = convert_num_turns_to_reward(self.current_player.turn_count)
            self.current_player.episode_rewards.append(reward)
            terminated = True

        info = TrainingInfo(
            num_turns=self.current_player.turn_count,
            reward=reward,
            game_score=self.current_player.game_score,
        )
        return self._get_obs(), reward, turn_is_over, terminated, truncated, info

    def _get_obs(self) -> Observation:
        return Observation(
            last_action=self.last_action,
            hold_score=self.current_player.hold_score,
            inheritance=self.inheritance,
            dice=self.dice,
            card=self.card,
        )

    def reset_for_new_turn(self, advance_to_next_player: bool = True) -> Observation:
        if advance_to_next_player:
            self.current_player.hold_score = 0
            self.current_player.turn_count += 1
            self.current_player = self.players[
                (self.players.index(self.current_player) + 1) % len(self.players)
            ]

        # Reset the turn state
        # self.last_action = None
        self.card = None
        self.held_dice = DiceCounts()
        if self.inheritance == 0:
            self.dice = DiceCounts()  # Keep dice around if there's an inheritance
        return self._get_obs()

    def reset_for_new_game(self) -> Observation:
        for player in self.players:
            if player.turn_count:
                player.num_turns_in_game.append(player.turn_count)
                player.game_count += 1
            player.turn_count = 0
            player.hold_score = 0
            player.game_score = 0
        self.last_action = None
        self.inheritance = 0
        self.current_player = random.choice(self.players)
        self.reset_for_new_turn(advance_to_next_player=False)
        return self._get_obs()
