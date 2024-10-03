import numpy as np
import random
import matplotlib.pyplot as plt


class Machine:
    def __init__(self, cycle_time):
        self.cycle_time = cycle_time
        self.state = 0  # 0: waiting, 1: processing, 2: starved, 3: blocked
        self.time_elapsed = 0

    def step(self):
        if self.state == 1:
            self.time_elapsed += 1
            if self.time_elapsed >= self.cycle_time:
                self.state = 0  # Reset after processing
                self.time_elapsed = 0
                return True  # Processing completed
        return False

    def start_processing(self):
        if self.state == 0:
            self.state = 1  # Start processing


class Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.count = 0

    def add(self):
        if self.count < self.capacity:
            self.count += 1

    def remove(self):
        if self.count > 0:
            self.count -= 1
            return True
        return False


class ProductionLine:
    def __init__(self, cycle_times, buffer_capacity, load_time=1, travel_time=1):
        self.machines = [Machine(cycle_time) for cycle_time in cycle_times]
        self.buffer = Buffer(buffer_capacity)
        self.total_reward = 0
        self.production_count = 0
        self.robot_state = 0  # 0: idle, 1: loading/unloading, 2: traveling
        self.robot_position = 0  # 0: at Machine 1, 1: at Machine 2
        self.time_elapsed = 0
        self.load_time = load_time
        self.travel_time = travel_time

    def step(self, action):
        reward = 0
        # Check if action is not 0 and buffer count is not greater than 0
        if action != 0 and self.buffer.count <= 0:
            # Process machine states for both machines, allowing processing regardless of robot state
            for index, machine in enumerate(self.machines):
                # machine.check_starvation_blockage(self.buffer)
                # print(type(machine))
                if machine.step():
                    if index == 0:  # Process Machine
                        self.buffer.count += 1
                    if index == 1:
                        self.production_count += 1
                        reward += 1
                        self.total_reward += 1
            #                         print("self.production_count: ", self.production_count)
            # if self.machines[action].step():  # Simulate unloading
            #     if self.buffer.count < self.buffer.capacity:
            #         self.buffer.add()  # Add to buffer after unloading
            return self._get_obs(), reward  # No action taken, just return observation and reward

        # Check if the machine corresponding to the action is idle (state 0)
        if action == 0 and self.machines[0].state != 0:
            # Process machine states for both machines, allowing processing regardless of robot state
            for index, machine in enumerate(self.machines):
                # machine.check_starvation_blockage(self.buffer)
                # print(type(machine))
                if machine.step():
                    if index == 0:  # Process Machine
                        self.buffer.count += 1
                    if index == 1:
                        self.production_count += 1
                        reward += 1
                        self.total_reward += 1
            # if self.machines[action].step():  # Simulate unloading
            #     if self.buffer.count < self.buffer.capacity:
            #         self.buffer.add()  # Add to buffer after unloading
            return self._get_obs(), reward  # Machine 1 is not idle, so action cannot be taken
        elif action == 1 and self.machines[1].state != 0:
            # Process machine states for both machines, allowing processing regardless of robot state
            for index, machine in enumerate(self.machines):
                # machine.check_starvation_blockage(self.buffer)
                # print(type(machine))
                if machine.step():
                    if index == 0:  # Process Machine
                        self.buffer.count += 1
                    if index == 1:
                        self.production_count += 1
                        reward += 1
                        self.total_reward += 1
            return self._get_obs(), reward  # Machine 2 is not idle, so action cannot be taken

        if self.robot_state == 0:  # Idle
            if self.robot_position == action:  # At the correct machine
                self.robot_state = 1  # Start loading/unloading
                self.time_elapsed = 0  # Reset time elapsed for loading
            else:  # Need to travel to the specified machine
                self.robot_state = 2  # Start traveling
                self.time_elapsed = 0  # Reset time elapsed for travel

        elif self.robot_state == 1:  # Loading/Unloading in progress
            if self.time_elapsed < self.load_time:
                self.time_elapsed += 1
                return self._get_obs(), reward  # Return without proceeding

            # After loading/unloading is done
            if action == 0:  # Machine 1
                if self.machines[action].state == 0:  # If Machine 1 is idle
                    if self.buffer.count < self.buffer.capacity:
                        self.machines[action].start_processing()  # Start processing
                elif self.machines[action].state != 0:  # Unload if Machine 1 is processing
                    if self.machines[action].step():  # Simulate unloading
                        if self.buffer.count < self.buffer.capacity:
                            self.buffer.add()  # Add to buffer after unloading

            elif action == 1:  # Machine 2
                if self.machines[action].state != 0:  # Unload if Machine 2 is processing
                    if self.machines[action].step():  # Simulate unloading
                        reward += 1  # Reward for unloading a processed part
                        self.total_reward += 1  # Update total reward
                        self.production_count += 1  # Increment production count
                else:  # If Machine 2 is idle
                    if self.buffer.count > 0:  # Check buffer before loading
                        self.buffer.remove()  # Remove part from buffer
                        self.machines[action].start_processing()  # Start processing

            self.robot_state = 0  # Transition back to idle after loading/unloading

        elif self.robot_state == 2:  # Traveling
            if self.time_elapsed < self.travel_time:
                self.time_elapsed += 1
                return self._get_obs(), reward  # Return without proceeding
            else:
                self.robot_state = 1  # Reset to idle after travel
                self.time_elapsed = 0  # Reset time elapsed after travel
                self.robot_position = action  # Update robot position after reaching

        # Process machine states for both machines, allowing processing regardless of robot state
        for index, machine in enumerate(self.machines):
            # machine.check_starvation_blockage(self.buffer)
            # print(type(machine))
            if machine.step():
                if index == 0:  # Process Machine
                    self.buffer.count += 1
                if index == 1:
                    self.production_count += 1
                    reward += 1
                    self.total_reward += 1
        return self._get_obs(), reward

    def reset(self):
        for machine in self.machines:
            machine.state = 0
            machine.time_elapsed = 0
        self.buffer.count = 0
        self.total_reward = 0
        self.robot_state = 0
        self.robot_position = 0
        self.time_elapsed = 0
        return self._get_obs()

    def _get_obs(self):
        machine_states = [machine.state for machine in self.machines]
        return np.array(machine_states + [self.buffer.count, self.robot_state, self.robot_position], dtype=np.float32)


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0,
                 exploration_decay=0.99, min_exploration_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table

    def choose_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.choice(range(self.action_size))  # Explore action space
        return np.argmax(self.q_table[state])  # Exploit learned values

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta

    def decay_exploration(self):
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay


def state_to_index(state, buffer_capacity):
    # Convert the state representation to a unique index for Q-table
    # state: [machine0_state, machine1_state, buffer_count, robot_state, robot_position]
    machine0_state = int(state[0])  # 0 or 1
    machine1_state = int(state[1])  # 0 or 1
    buffer_count = int(state[2])  # 0 to buffer_capacity
    robot_state = int(state[3])  # 0, 1, or 2
    robot_position = int(state[4])  # 0 or 1

    return (machine0_state * 2 + machine1_state) * (
                buffer_capacity + 1) * 6 + buffer_count * 6 + robot_state * 2 + robot_position


# Example usage
if __name__ == "__main__":
    cycle_times = [2, 3]
    buffer_capacity = 3
    env = ProductionLine(cycle_times, buffer_capacity)

    # Calculate the state size based on your specifications
    state_size = (2 ** 2) * (
                buffer_capacity + 1) * 3 * 2  # 2 machines (2 states each), buffer capacity + 1 states, robot states, robot position
    agent = QLearningAgent(state_size=state_size, action_size=2)  # 2 actions (machine 0 or 1)

    episodes = 500  # Number of episodes to train
    steps_per_episode = 1000

    log_data = []
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(steps_per_episode):
            state_index = state_to_index(state, buffer_capacity)
            action = agent.choose_action(state_index)
            next_state, reward = env.step(action)

            # Log the episode, step, state, action, and reward
            log_data.append(
                f"Episode: {episode + 1}, Step: {step + 1}, State: {tuple(state)}, Action: {action}, Reward: {reward}")

            next_state_index = state_to_index(next_state, buffer_capacity)
            agent.update_q_value(state_index, action, reward, next_state_index)
            total_reward += reward
            state = next_state

        agent.decay_exploration()
        rewards_per_episode.append(total_reward)  # Append total reward for this episode
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Exploration Rate: {agent.exploration_rate:.3f}")

    # Plotting the reward curve
    plt.plot(range(episodes), rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward Curve Over Episodes')
    plt.grid()
    plt.savefig('reward_curve.png')
    plt.show()

    # Save log data to a file
    with open('log_data.txt', 'w') as f:
        for entry in log_data:
            f.write(entry + "\n")
