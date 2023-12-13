import random

from dice.types import ActionT, StateT


class Game:
    NUM_DICE = 6

    def __init__(self) -> None:
        self.score = 0
        self.dice: list[int] = self._get_fresh_dice(Game.NUM_DICE)
        self.holds: dict[int, int] = self._get_fresh_holds()
        self.actions: list[ActionT] = []

    def _get_fresh_dice(self, num_dice: int) -> list[int]:
        return [-1 for _ in range(num_dice)]

    def _get_fresh_holds(self) -> dict[int, int]:
        return {n: 0 for n in range(1, Game.NUM_DICE + 1)}

    @property
    def num_dice(self) -> int:
        return len(self.dice)

    @property
    def state(self) -> StateT:
        return self.to_state()

    @property
    def stopped(self) -> bool:
        return self.score > 0 and sum(self.holds.values()) == 0

    @property
    def busted(self) -> bool:
        # We rolled and got no pointers
        return (
            sum(self.state["rolled"].values()) > 0 and self.calc_score(self.dice) == 0
        )

    @property
    def filled(self) -> bool:
        return self.num_dice == 0

    @property
    def num_ones_rolled(self) -> int:
        return self.dice.count(1)

    @property
    def num_ones_held(self) -> int:
        return self.holds[1]

    @property
    def num_twos_rolled(self) -> int:
        return self.dice.count(2)

    @property
    def num_twos_held(self) -> int:
        return self.holds[2]

    @property
    def num_threes_rolled(self) -> int:
        return self.dice.count(3)

    @property
    def num_threes_held(self) -> int:
        return self.holds[3]

    @property
    def num_fours_rolled(self) -> int:
        return self.dice.count(4)

    @property
    def num_fours_held(self) -> int:
        return self.holds[4]

    @property
    def num_fives_rolled(self) -> int:
        return self.dice.count(5)

    @property
    def num_fives_held(self) -> int:
        return self.holds[5]

    @property
    def num_sixes_rolled(self) -> int:
        return self.dice.count(6)

    @property
    def num_sixes_held(self) -> int:
        return self.holds[6]

    def to_state(self) -> StateT:
        return {
            "final_score": self.score,
            "num_dice": self.num_dice,
            "rolled": {
                1: self.num_ones_rolled,
                2: self.num_twos_rolled,
                3: self.num_threes_rolled,
                4: self.num_fours_rolled,
                5: self.num_fives_rolled,
                6: self.num_sixes_rolled,
            },
            "held": {
                1: self.num_ones_held,
                2: self.num_twos_held,
                3: self.num_threes_held,
                4: self.num_fours_held,
                5: self.num_fives_held,
                6: self.num_sixes_held,
            },
        }

    def roll_dice(self, num_dice: int) -> None:
        self.dice = sorted(random.randint(1, 6) for _ in range(num_dice))

    @classmethod
    def calc_score(cls, dice: list[int]) -> int:
        return 100 * dice.count(1) + 50 * dice.count(5)

    def handle_roll(self) -> None:
        self.actions.append("roll")
        self.roll_dice(self.num_dice)
        if self.busted:
            self.handle_bust()

    def handle_hold(self, hold_action: ActionT) -> None:
        self.actions.append(hold_action)
        assert hold_action is not None
        new_holds: dict[int, int] = {}
        for die, hold_count in enumerate(hold_action.split("_")[-1], start=1):
            new_holds[die] = int(hold_count)

        assert new_holds[1] <= self.num_ones_rolled, (
            new_holds[1],
            self.num_ones_rolled,
        )
        assert new_holds[5] <= self.num_fives_rolled, (
            new_holds[5],
            self.num_fives_rolled,
        )

        # Add the new held dice from this roll to the turn's held dice
        # Also remove the held dice from `self.dice`
        for die, hold_count in new_holds.items():
            if hold_count:
                self.holds[die] += hold_count
                for _ in range(hold_count):
                    self.dice.remove(die)

        # Reset dice to indicate that they're ready to be rolled
        self.dice = self._get_fresh_dice(len(self.dice))

    def handle_bust(self) -> None:
        self.score = 0

    def handle_stop(self) -> None:
        self.actions.append("stop")
        hold_score = 0
        dice_from_holds: list[int] = []
        for die, hold_count in self.holds.items():
            dice_from_holds.extend([die for _ in range(hold_count)])
        hold_score += self.calc_score(dice_from_holds)

        self.score += hold_score
        self.holds = self._get_fresh_holds()

    @property
    def turn_is_over(self) -> bool:
        return self.busted or self.stopped or self.filled
