from matplotlib import pyplot as plt

from DQN import DQNAgent
from Prod_line import ProductionLine

if __name__ == "__main__":
    cycle_times = [2, 3]
    buffer_capacity = 3
    env = ProductionLine(cycle_times, buffer_capacity)

    # Calculate state size based on your environment (adjust this to match the actual observation shape)
    state_size = 5  # Modify this based on your observation (machine states, buffer, robot states, etc.)
    action_size = 3  # 2 actions (machine 0 or 1, 2 is for no action)

    agent = DQNAgent(state_size=state_size, action_size=action_size)

    episodes = 1000
    steps_per_episode = 1000
    target_update_freq = 10

    rewards_per_episode = []
    last_action = None

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(steps_per_episode):
            action = agent.choose_action(state)

            # Check if the action is 2 and use last_action if available
            if action == 2 and last_action is not None:
                action = last_action  # Use the last action
            if state[3]==1:
                action = last_action
            next_state, reward = env.step(action)

            done = step == steps_per_episode - 1
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            total_reward += reward
            state = next_state
            # Update last_action with the current action
            last_action = action

            if done:
                break

        agent.decay_exploration()

        # Update target network every few episodes
        if episode % target_update_freq == 0:
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Exploration Rate: {agent.exploration_rate:.3f}")

    agent.save('dqn_v0_model.pth')

    # Plot reward curve
    plt.plot(range(episodes), rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Curve Over Episodes')
    plt.grid()
    # plt.savefig('reward_curve.png')
    plt.show()

