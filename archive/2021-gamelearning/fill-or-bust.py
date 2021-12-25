import sys

from fillorbust.agents import DumbAgent
from fillorbust.training import play_a_game_of_agents


def main():
    agents = [DumbAgent("X"), DumbAgent("O")]
    play_a_game_of_agents(agents)
    print()


if __name__ == "__main__":
    main()
    sys.exit()
