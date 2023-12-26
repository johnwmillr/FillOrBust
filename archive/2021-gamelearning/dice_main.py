import cProfile
from dice.agents import Agent, Smart, StopAtNDice
from dice.q import QTable


def main() -> None:
    print("Start!")

    # Train the smart agent
    smart = Smart(
        "smart",
        Q=QTable(
            learning_rate=0.8,
            discount_factor=0.9,
            epsilon=0.9,
        ),
    )

    agents: list[Agent] = [
        smart,
        Agent("dumb "),
        # RiskyAgent("risky"),
        StopAtNDice("stop2", num_dice_to_stop_at=2),
        # StopAtNDice("stop3", num_dice_to_stop_at=3),
    ]

    NUM_TRAIN_GAMES = 1_000
    if smart in agents:
        smart.train(num_episodes=NUM_TRAIN_GAMES, evaluate=False)

    # Evaluate the agents
    NUM_TEST_GAMES = 1000
    for agent in agents:
        for _ in range(NUM_TEST_GAMES):
            agent.play()

    for agent in agents:
        agent.print_performance()

    if smart.evaluations:
        smart.plot_evaluations()

    # if NUM_TRAIN_GAMES >= 1_000_000:
    #     smart.save(f"./dice/trained_agent_{smart.num_games_played}.pkl")


if __name__ == "__main__":
    main()


# cProfile.run("main()", filename="profiling_results.txt")
