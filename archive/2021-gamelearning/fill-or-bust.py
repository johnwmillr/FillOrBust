import sys

from fillorbust.agents import DumbAgent, SmartAgent
from fillorbust.training import play_a_game_of_agents


def main():
    agents = [SmartAgent("X"), DumbAgent("O")]
    play_a_game_of_agents(agents)


if __name__ == "__main__":
    main()
    sys.exit()
