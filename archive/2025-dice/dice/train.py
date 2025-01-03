import argparse

import altair as alt
import pandas as pd
from tqdm import tqdm

from dice.agents import Agent, SmartAgent
from dice.game import FillOrBust

# alt.renderers.set_embed_options(theme="dark")
# alt.renderers.enable("mimetype")


def train(
    agent: SmartAgent,
    *,
    n_episodes: int = 1000,
) -> None:
    """Train agents to play Fill or Bust.

    Args:
        agents: The agents to train.
        n_episodes: The number of episodes to train the agents. Defaults to 1000.
    """
    env = FillOrBust()
    for _ in tqdm(range(n_episodes)):
        obs = env.reset_for_new_game()
        done = False

        # Play one episode (i.e. one game of Fill or Bust)
        while not done:
            # Pick an action and execute it
            action = agent.get_action(env, obs)
            next_obs, reward, terminated, truncated, stats = env.step(action)

            # Update the agent
            agent.update(
                obs,
                action,
                reward,
                terminated,
                next_obs,
                stats if terminated or truncated else None,
            )

            # Update the current obs and whether or not the episode is complete
            obs = next_obs
            done = terminated or truncated

    # Format the training stats as a DataFrame
    agent.training_stats = pd.DataFrame(agent._training_stats)


def plot_training_metric(
    agent: Agent,
    metric: str = "num_turns",
    downsample: bool = True,
) -> None:
    # Format the data
    if agent.training_stats is None:
        raise ValueError("Agent has not been trained yet.")
    if metric not in agent.training_stats.columns:
        raise ValueError(f"Invalid metric: {metric}")
    data = agent.training_stats[metric]
    if downsample:
        data = data[:: len(data) // min(2500, len(data))]

    if metric == "num_turns":
        s = pd.Series(data)
        source = (
            pd.DataFrame(-s.diff()[s.diff() < 0], columns=[metric])
            .reset_index()
            .rename(columns={"index": "episode"})
        )
    else:
        source = (
            pd.DataFrame(data, columns=[metric])
            .reset_index()
            .rename(columns={"index": "episode"})
        )
    print(f"Median `{metric}` after training: {source.tail(50)[metric].median()}")

    # Create the chart
    padding = int(0.1 * len(source))
    line = (
        alt.Chart(source)
        .mark_line(color="red", size=3)
        .transform_window(
            rolling_mean=f"median({metric})",
            frame=[-padding, padding],
        )
        .encode(x="episode", y="rolling_mean:Q")
    )
    points = (
        alt.Chart(source)
        .mark_point(opacity=0.2, color="orange")
        .encode(
            x="episode",
            y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 100])).title(metric),
        )
    )

    chart = (points + line).properties(width=400, height=250)
    chart.show()


def main(
    n_episodes: int = 1000,
    learning_rate: float | None = None,
    epsilon: float | None = None,
    discount_factor: float | None = None,
    use_smart_agent: bool = True,
) -> SmartAgent:
    """Train agents to play Fill or Bust.

    Args:
        n_episodes: The number of episodes to train the agents. Defaults to 1000.
        learning_rate: The initial learning rate. Defaults to 0.001.
        epsilon: The initial epsilon value. Defaults to 1.0.
        discount_factor: The discount factor for computing the Q-value.
    """
    agent: SmartAgent | Agent
    if use_smart_agent:
        agent = SmartAgent(
            "smart",
            learning_rate=learning_rate,
            epsilon=epsilon,
            discount_factor=discount_factor,
        )
    else:
        agent = Agent("dumb")
    train(agent, n_episodes=n_episodes)
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_episodes", type=int, default=10000, help="The number of episodes to train."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="The initial learning rate.",
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=None,
        help="The decay for the learning rate.",
    )
    parser.add_argument(
        "--final_lr", type=float, default=0.0001, help="The final learning rate."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="The initial epsilon value."
    )
    parser.add_argument(
        "--epsilon_decay",
        type=float,
        default=None,
        help="The decay for epsilon.",
    )
    parser.add_argument(
        "--final_epsilon",
        type=float,
        default=0.1,
        help="The final epsilon value.",
    )
    parser.add_argument(
        "--discount_factor",
        type=float,
        default=0.7,
        help="The discount factor for computing the Q-value.",
    )
    args = parser.parse_args()

    main(
        args.n_episodes,
        args.learning_rate,
        args.lr_decay,
        args.final_lr,
        args.epsilon,
        args.epsilon_decay,
        args.final_epsilon,
        args.discount_factor,
    )
