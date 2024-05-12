from tqdm import tqdm

from dice.agents import Agent, SmartAgent
from dice.game import FillOrBustEnvironment, determine_available_actions


def main() -> None:
    n_episodes = 1000
    start_learning_rate = 0.01
    lr_decay = start_learning_rate / (n_episodes // 2)
    final_lr = 0.001
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes // 1.5)
    final_epsilon = 0.1
    discount_factor = 0.5

    kwargs = dict(
        initial_learning_rate=start_learning_rate,
        learning_rate_decay=lr_decay,
        final_learning_rate=final_lr,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    agent1 = SmartAgent("agent1", **kwargs)
    agent2 = SmartAgent("agent2", **kwargs)

    env = FillOrBustEnvironment()
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset_for_new_game()
        done, turn_is_over = False, False

        # play one episode
        while not done:
            for agent in [agent1]:
                while not turn_is_over:
                    action = agent.get_action(obs)
                    next_obs, reward, terminated, truncated, turn_is_over, _ = env.step(
                        action
                    )

                    # update the agent
                    agent.update(obs, action, reward, terminated, next_obs)

                    # update if the environment is done and the current obs
                    done = terminated or truncated
                    obs = next_obs
                obs, _ = env.reset_for_new_turn()
                turn_is_over = False

        for agent in [agent1, agent2]:
            agent.decay_epsilon()
            agent.decay_learning_rate()

    print("Done training!")


if __name__ == "__main__":
    main()
