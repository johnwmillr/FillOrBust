import pickle

from agents import Agent, Smart, StopAtNDice
from q import QTable


if __name__ == "__main__":
    print("Start!")

    # Train the smart agent
    smart = Smart(
        "smart",
        Q=QTable(
            learning_rate=0.1,
            discount_factor=0.8,
            epsilon=0.7,
        ),
    )

    agents: list[Agent] = [
        # smart,
        # Agent("dumb "),
        # RiskyAgent("risky"),
        # StopAtNDice("stop1", num_dice_to_stop_at=1),
        # StopAtNDice("stop2", num_dice_to_stop_at=2),
        # StopAtNDice("stop3", num_dice_to_stop_at=3),
        # StopAtNDice("stop4", num_dice_to_stop_at=4),
        # StopAtNDice("stop5", num_dice_to_stop_at=5),
        StopAtNDice("stop6", num_dice_to_stop_at=6),
    ]

    NUM_TRAIN_GAMES = 1_000
    if smart in agents:
        smart.train(num_episodes=NUM_TRAIN_GAMES, evaluate=False, metric="mean")

    # Evaluate the agents
    NUM_TEST_GAMES = 1_000
    for agent in agents:
        for n in range(NUM_TEST_GAMES):
            agent.play()

    for agent in agents:
        agent.print_performance()

    if smart.evaluations:
        smart.plot_evaluations()

    if NUM_TRAIN_GAMES >= 1_000_000:
        fn = f"./dice/trained_agent_{smart.num_games_played}.pkl"
        with open(fn, "wb") as fout:
            print(f"Saved trained agent to '{fn}'")
            pickle.dump(smart, fout)
