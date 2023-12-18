import cProfile
from dice.agents import Agent, Smart, StopAtNDice
from dice.q import QTable


def main() -> None:
    print("Start!")

    # Train the smart agent
    smart = Smart(
        "smart",
        Q=QTable(
            learning_rate=0.6,
            discount_factor=0.7,
            epsilon=0.8,
        ),
    )

    agents: list[Agent] = [
        smart,
        # Agent("dumb "),
        # RiskyAgent("risky"),
        StopAtNDice("stop2", num_dice_to_stop_at=2),
        StopAtNDice("stop3", num_dice_to_stop_at=3),
    ]

    NUM_TRAIN_GAMES = 10_000
    if smart in agents:
        smart.train(num_episodes=NUM_TRAIN_GAMES, evaluate=True)

    # Evaluate the agents
    NUM_TEST_GAMES = 5_000
    for agent in agents:
        for _ in range(NUM_TEST_GAMES):
            agent.play()

    for agent in agents:
        agent.print_performance()

    if smart.evaluations:
        smart.plot_evaluations()

    if NUM_TRAIN_GAMES >= 1_000_000:
        smart.save(f"./dice/trained_agent_{smart.num_games_played}.pkl")


if __name__ == "__main__":
    main()


# cProfile.run("main()", filename="profiling_results.txt")
