import pickle
import time

from dice.agents import Agent, Smart, StopAtNDice
from dice.q import QTable
from dice.types import Estimator


def main() -> None:
    print("Start!")

    # Train the smart agent
    # smart = Smart(
    #     "smart",
    #     Q=QTable(
    #         learning_rate=0.3,
    #         discount_factor=0.5,
    #         epsilon=0.95,
    #     ),
    #     learning_rate_decay=0.6,
    #     discount_factor_decay=1.05,
    # )

    with open("./dice/trained_agents/trained_agent_1710054236_25000.pkl", "rb") as fin:
        smart: Smart = pickle.load(fin)

    fp = "./dice/dice_trained_estimator_RandomForestRegressor_1710095106.pkl"
    with open(fp, "rb") as fin:
        estimator: Estimator = pickle.load(fin)
        smart.estimator = estimator
        smart.estimator.verbose = 0
        print("Loaded the estimator.")

    agents: list[Agent] = [
        smart,
        # agent_with_estimator,
        # Agent("dumb "),
        # RiskyAgent("risky"),
        # StopAtNDice("stop2", num_dice_to_stop_at=2),
        # StopAtNDice("stop3", num_dice_to_stop_at=3),
    ]

    # NUM_TRAIN_GAMES = 25_000
    # if smart in agents:
    #     smart.train(
    #         num_episodes=NUM_TRAIN_GAMES, max_score=10_000, evaluate=True, verbose=True
    #     )

    # Evaluate the agents
    NUM_TEST_GAMES = 5
    for agent in agents:
        print(f"{agent.name=}")
        # if agent.name == "smart":
        #     agent = Smart("smart", Q=smart.Q)
        for _ in range(NUM_TEST_GAMES):
            agent.play(max_score=10_000, verbose=False)
        agent.print_performance()

    # if smart.evaluations:
    #     smart.plot_evaluations()

    # if NUM_TRAIN_GAMES >= 25_000:
    # smart.save(
    #     f"./dice/trained_agents/trained_agent_{int(time.time())}_{NUM_TRAIN_GAMES}.pkl"
    # )


if __name__ == "__main__":
    main()
