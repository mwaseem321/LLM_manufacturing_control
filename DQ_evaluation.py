# !pip install matplotlib
import numpy as np
import matplotlib.pyplot as plt

from DQN import DQNAgent
from Prod_line import ProductionLine


def evaluate_agent(agent, env, episodes=1, steps_per_episode=100, exploration_rate=0.10, runs_per_episode=1):
    log_data = []
    mean_rewards_per_episode = []
    std_rewards_per_episode = []
    last_action = None  # Initialize last_action outside the loop to track across steps
    for episode in range(episodes):
        rewards_for_episode = []

        # Run each episode for 'runs_per_episode' runs
        for run in range(runs_per_episode):
            state = env.reset()
            total_reward = 0

            for step in range(steps_per_episode):
                print("Step: ", step)
                print("State:", state)
                # Use a small exploration rate during evaluation
                if np.random.rand() <= exploration_rate:
                    action = agent.choose_action(state, exploit=False)  # Explore action space
                else:
                    action = agent.choose_action(state, exploit=True)  # Exploit learned policy

                # Check if the action is 2 and use last_action if available
                print("Action proposed based on policy: ", action)
                if action == 2 and last_action is not None:
                    action = last_action  # Use the last action
                # if state[3] == 1:
                #     action = last_action

                print("Action:", action)
                next_state, reward = env.step(action)
                total_reward += reward
                state = next_state

                # Update last_action with the current action
                last_action = action
            rewards_for_episode.append(total_reward)

        # Calculate mean and standard deviation for this episode
        mean_reward = np.mean(rewards_for_episode)
        std_reward = np.std(rewards_for_episode)

        mean_rewards_per_episode.append(mean_reward)
        std_rewards_per_episode.append(std_reward)

    # Convert to numpy arrays for easier plotting
    mean_rewards_per_episode = np.array(mean_rewards_per_episode)
    std_rewards_per_episode = np.array(std_rewards_per_episode)

    # Plotting the evaluation reward curve with upper and lower bounds
    episodes_range = np.arange(episodes)

    #     plt.plot(episodes_range, mean_rewards_per_episode, label='Mean Reward')
    #     plt.fill_between(episodes_range,
    #                      mean_rewards_per_episode - std_rewards_per_episode,
    #                      mean_rewards_per_episode + std_rewards_per_episode,
    #                      color='gray', alpha=0.3, label='Standard Deviation')

    #     plt.xlabel('Episode')
    #     plt.title('Evaluation Reward Curve with Standard Deviation Over Episodes')
    #     plt.grid()
    #     plt.legend()
    # #     plt.savefig('evaluation_reward_curve_with_std.png')
    #     plt.show()

    plt.bar(episodes_range, mean_rewards_per_episode, yerr=std_rewards_per_episode, alpha=0.7, capsize=10,
            label='Mean Reward')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation Reward Bar Chart with Standard Deviation')
    plt.grid()
    plt.xticks(episodes_range)  # Ensures proper placement of x-axis ticks
    plt.xlim(-0.5, episodes - 0.5)  # Limits the x-axis so bars are centered
    plt.legend()
    plt.show()


if __name__ == "__main__":
    cycle_times = [2, 3]
    buffer_capacity = 3
    env = ProductionLine(cycle_times, buffer_capacity)

    # Calculate state size based on your environment (adjust this to match the actual observation shape)
    state_size = 5  # Modify this based on your observation (machine states, buffer, robot states, etc.)
    action_size = 3  # 2 actions (machine 0 or 1)

    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.load('dqn_v0_model.pth')

    # Evaluate the agent with multiple runs per episode
    evaluate_agent(agent, env)
