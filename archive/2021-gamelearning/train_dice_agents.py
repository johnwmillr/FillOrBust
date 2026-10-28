import argparse

import altair as alt
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

from dice.agents import Agent, SmartAgent
from dice.game import FillOrBust

# alt.renderers.set_embed_options(theme="dark")
# alt.renderers.enable("mimetype")


def train(
    agents: list[Agent],
    *,
    n_episodes: int = 1000,
) -> None:
    """Train agents to play Fill or Bust.

    Args:
        agents: The agents to train.
        n_episodes: The number of episodes to train the agents. Defaults to 1000.
    """
    env = FillOrBust(agents)
    for _ in tqdm(range(n_episodes)):
        obs = env.reset_for_new_game()
        done = False

        # Play one episode (i.e. one game of Fill or Bust)
        while not done:
            turn_is_over = False
            while not turn_is_over:
                # Pick an action and execute it
                action = env.current_player.get_action(obs)
                next_obs, reward, turn_is_over, terminated, truncated, info = env.step(
                    action
                )

                # Update the agent
                env.current_player.update(
                    obs, action, reward, turn_is_over, terminated, next_obs, info
                )

                # Update the current obs and whether or not the episode is complete
                obs = next_obs
                done = terminated or truncated

            if isinstance(env.current_player, SmartAgent):
                env.current_player.decay_epsilon()
                env.current_player.decay_learning_rate()
            obs = env.reset_for_new_turn()


def plot_training_metric(
    agent: Agent,
    metric: str = "num_turns",
    downsample: bool = True,
) -> None:
    # Format the data
    data = getattr(agent, metric)
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
            y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 90])).title(metric),
        )
    )

    chart = (points + line).properties(width=400, height=250)
    chart.show()


def main(
    n_episodes: int = 1000,
    start_learning_rate: float = 0.001,
    lr_decay: float | None = None,
    final_lr: float = 0.0001,
    start_epsilon: float = 1.0,
    epsilon_decay: float | None = None,
    final_epsilon: float = 0.1,
    discount_factor: float = 0.7,
) -> None:
    """Train agents to play Fill or Bust.

    Args:
        n_episodes: The number of episodes to train the agents. Defaults to 1000.
        start_learning_rate: The initial learning rate. Defaults to 0.001.
        lr_decay: The decay for the learning rate. Defaults to the
            learning rate divided by half the number of episodes.
        final_lr: The final learning rate. Defaults to 0.0001.
        start_epsilon: The initial epsilon value. Defaults to 1.0.
        epsilon_decay: The decay for epsilon. Defaults to epsilon
            divided by half the number of episodes.
        final_epsilon: The final epsilon value. Defaults to 0.1.
        discount_factor: The discount factor for computing the Q-value.
    """
    if lr_decay is None:
        lr_decay = start_learning_rate / (n_episodes // 2)
    if epsilon_decay is None:
        epsilon_decay = start_epsilon / (n_episodes // 2)

    kwargs = dict(
        initial_learning_rate=start_learning_rate,
        learning_rate_decay=lr_decay,
        final_learning_rate=final_lr,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    smart = SmartAgent("smart", **kwargs)
    dumb = Agent("dumb")
    agents: list[Agent] = [smart, dumb]

    train(agents, n_episodes=n_episodes)
    plot_training_metric(smart, metric="num_turns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_episodes", type=int, default=10000, help="The number of episodes to train."
    )
    parser.add_argument(
        "--start_learning_rate",
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
        "--start_epsilon", type=float, default=1.0, help="The initial epsilon value."
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
        args.start_learning_rate,
        args.lr_decay,
        args.final_lr,
        args.start_epsilon,
        args.epsilon_decay,
        args.final_epsilon,
        args.discount_factor,
    )
