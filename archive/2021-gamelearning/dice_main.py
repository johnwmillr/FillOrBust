from tqdm import tqdm

from dice.agents import Agent, SmartAgent
from dice.game import FillOrBustEnvironment


def main() -> None:
    n_episodes = 1000
    start_learning_rate = 0.001
    lr_decay = start_learning_rate / (n_episodes // 2)
    final_lr = 0.0001
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes // 2)
    final_epsilon = 0.1
    discount_factor = 0.7

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

    # agent1 = Agent("agent1")
    # agent2 = Agent("agent2")
    # agent3 = Agent("agent3")

    agents: list[Agent] = [agent1, agent2]

    env = FillOrBustEnvironment()
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset_for_new_game()
        done, turn_is_over = False, False
        highest_score, winner = 0, None

        # play one episode
        while not done:
            for agent in agents:
                while not turn_is_over:
                    action = agent.get_action(obs)
                    next_obs, reward, terminated, truncated, turn_is_over, info = (
                        env.step(agent, action)
                    )

                    agent.update(
                        obs, action, reward, terminated, turn_is_over, next_obs, info
                    )

                    done = terminated or truncated
                    obs = next_obs
                turn_is_over = False
                if done:
                    break

        for agent in agents:
            if agent.game_score > highest_score:
                highest_score = agent.game_score
                winner = agent
            agent.game_score = 0
            if isinstance(agent, SmartAgent):
                agent.decay_epsilon()
                agent.decay_learning_rate()

        for agent in agents:
            if agent == winner:
                agent.wins += 1
                break


if __name__ == "__main__":
    main()
