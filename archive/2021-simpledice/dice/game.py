"""
Dice
"""

import random


def roll_dice(verbose: bool = False) -> int:
    """Rolls a dice, returns the corresponding score"""
    roll = random.choice(range(1, 7))
    pointers = {1: 100, 2: 50, 3: 50, 5: 50}
    if verbose:
        print(f"Roll: {roll}")
    return pointers.get(roll, 0)


def stop_turn() -> str:
    return "stop"


def determine_action(state: int, stopping_score: int):
    """Returns a function based on `state`"""
    # Determine the stopping score based on the specified strategy
    # if strategy == "careful":
    #     stopping_score = 0
    # elif strategy == "greedy":
    #     stopping_score = 0

    # Choose an action according to play strategy
    if state > stopping_score:
        return stop_turn
    return roll_dice


def play(strategy) -> int:
    state, total_reward, num_turns = 0, 0, 0
    while True:
        action = determine_action(state, strategy)
        reward = action()
        num_turns += 1
        if reward == 0: # Didn't roll a pointer
            return 0
        elif reward == "stop": # Chose to stop
            return total_reward

        # Otherwise, keep playing
        total_reward += reward
        state = total_reward


def play_many_games(
        strategy: str,
        num_games: int = 10_000,
        verbose: bool = False) -> None:
    return [play(strategy) for _ in range(num_games)]

    if verbose:
        summary = f"""
            After playing {num_games / 1e3:.1f}k games with the '{strategy}' strategy,
            the average score was {average_score}."""
        print(summary)
    return average_score


class Player():

    def __init__(self, name: str):
        self.name = name
        self.score = 0




class Game():

    def __init__(self, num_players: int):
        self.num_players = num_players
        self.players = [Player(n) for n in range(num_players)]

    def play(self):
        pass