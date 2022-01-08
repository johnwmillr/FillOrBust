import sys

from fillorbust.agents import DumbAgent, SmartAgent
from fillorbust.training import play_many_games


def main():
    agents = [SmartAgent("X"), DumbAgent("O")]
    num_games = 100
    agents = play_many_games(agents, num_games=num_games, verbose=True)
    for agent in agents:
        print(agent.__repr__())
        agent.save()


if __name__ == "__main__":
    main()
    sys.exit()
