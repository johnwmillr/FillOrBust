from typing import Tuple

from .agents import DumbAgent
from .game import FillOrBust


def play_a_game_of_agents(agents: Tuple[DumbAgent, ...], verbose: bool = True) -> str:
    game = FillOrBust(agents)
    while not game.game_is_over:
        agent = [a for a in agents if a.symbol == game.current_player][0]

        # Current player take their turn
        game.take_turn(agent)

    print(f"\n{game.winner} is the winner!")
    for agent in agents:
        print(agent.symbol, agent.score)
    return game.winner
