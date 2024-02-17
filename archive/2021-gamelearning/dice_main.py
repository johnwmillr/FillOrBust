import time
from dice.agents import Agent, Smart, StopAtNDice
from dice.q import QTable


def main() -> None:
    print("Start!")

    # Train the smart agent
    smart = Smart(
        "smart",
        Q=QTable(
            learning_rate=0.2,
            discount_factor=0.65,
            epsilon=0.8,
        ),
        learning_rate_decay=0.9,
        discount_factor_decay=1.05,
    )

    agents: list[Agent] = [
        smart,
        # Agent("dumb "),
        # RiskyAgent("risky"),
        StopAtNDice("stop2", num_dice_to_stop_at=2),
        StopAtNDice("stop3", num_dice_to_stop_at=3),
    ]

    NUM_TRAIN_GAMES = 10000
    if smart in agents:
        smart.train(
            num_episodes=NUM_TRAIN_GAMES, max_score=10_000, evaluate=True, verbose=True
        )

    # Evaluate the agents
    NUM_TEST_GAMES = 1_000
    for agent in agents:
        if isinstance(agent, Smart):
            agent = Smart("Smart", Q=smart.Q)
        for _ in range(NUM_TEST_GAMES):
            agent.play(max_score=10_000)
        agent.print_performance()

    if smart.evaluations:
        smart.plot_evaluations()

    if NUM_TRAIN_GAMES >= 25_000:
        smart.save(f"./dice/trained_agent_{int(time.time())}_{NUM_TRAIN_GAMES}.pkl")


if __name__ == "__main__":
    main()


# cProfile.run("main()", filename="profiling_results.txt")
