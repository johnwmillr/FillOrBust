import altair as alt
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dice.agents import Agent, SmartAgent
from dice.game import FillOrBustEnvironment, determine_available_actions

alt.data_transformers.enable("vegafusion")


def main() -> None:
    # hyperparameters
    n_episodes = 10_000
    learning_rate = 0.5
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (
        n_episodes // 1.5
    )  # reduce the exploration over time
    final_epsilon = 0.7
    discount_factor = 1.0

    # agent = Agent("dumb")
    agent = SmartAgent(
        "smart",
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    env = FillOrBustEnvironment()
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset_for_new_game()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    print("Done training!")


if __name__ == "__main__":
    main()
