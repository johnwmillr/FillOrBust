from tqdm import tqdm

from dice.agents import Agent
from dice.game import FillOrBustEnvironment, determine_available_actions


def main() -> None:
    # hyperparameters
    n_episodes = 10000
    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1

    agent = Agent(
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
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
