from typing import Tuple

from .agents import DumbAgent
from .game import FillOrBust


def play_a_game_of_agents(agents: Tuple[DumbAgent, ...], verbose: bool = True) -> str:
    game = FillOrBust(agents)
    while not game.game_is_over:
        agent = [a for a in agents if a == game.current_player][0]
        game.take_turn(agent)

    if verbose:
        print(f"\n{game.winner} is the winner!")
    for agent in agents:
        if verbose:
            print(agent.name, agent.score)
    return game.winner


def play_many_games(players, num_games: int = 1000, verbose=True):
    for n in range(num_games):
        play_a_game_of_agents(players, verbose=False)
        if verbose and (n == 0 or n % int(0.1 * num_games) == 0):
            print(f"({n}/{num_games})")
            [print(player.__repr__()) for player in players]
    return players
