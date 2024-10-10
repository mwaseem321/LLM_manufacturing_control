import numpy as np
import random
import matplotlib.pyplot as plt
import openai
import time

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
        if action ==2:
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


from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="")



def get_chatgpt_action(state):
    # Define the prompt based on the state
    prompt = f"""
    Consider a manufacturing system with two machines and one buffer. Each machine has a dedicated processing time and the buffer has a maximum capacity. The first machine processes materials and stores in-process parts into the buffer, whereas the second machine is loaded with these parts for further processing. A part is considered complete after it is processed by the last machine.
A machine status of 0 means idle and ready to be loaded whereas a status of 1 means it is processing.
- If the buffer is empty, the last machine (machine 2) gets starved.
- If the buffer is full, the first machine (machine 1) gets blocked.
- A mobile robot loads parts to machines. The robot can take the following actions: 
   - Load machine 1 (action 0),
   - Load machine 2 (action 1), 
   - Take no action (action 2).

- The robot's status is as follows: 
   - Status 0: free, 
   - Status 1: loading, 
   - Status 2: traveling.

- If the robot is free (status 0) and a machine is ready to be loaded, it can be assigned to that machine.
- If the robot is travelling (status 2) or loading (status 1), no action can be taken (action 2). 
- If both machines are ready, prioritize loading the machine where the robot is currently located.
- If the machine status is 1 i.e., processing, the robot cannot be assigned to that machine. 
- If the buffer has 3 units, action 0 must not be taken.
- If the buffer has 0 units, action 1 must not be taken.

The goal is to maximize throughput and minimize idle times by preventing blocking or starvation.

Now, tell me: if machine 1 is {state[0]}, machine 2 is {state[1]}, the buffer has {state[2]} units, the robot status is {state[3]}, 
and the robot position is {state[4]}, what action should be assigned to the robot? 

The output must only provide the action with no details. 
    """

    # Call OpenAI API with the new client
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="gpt-4",  # or "gpt-4", gpt-3.5-turbo
        max_tokens=10,  # Adjust as needed
        temperature=0.3
    )

    # Extract action from the response
    action = int(response.choices[0].message.content.strip())
    print('action:', action)
    return action

def evaluate_chatgpt_agent(env, steps_per_episode=100, runs_per_episode=1):
    log_data = []
    rewards_for_runs = []

    last_action = None  # Initialize last_action outside the loop to track across steps

    # Run the single episode for 3 runs
    for run in range(runs_per_episode):
        state = env.reset()
        run_reward = 0

        for step in range(steps_per_episode):
            print("Step: ", step)
            print("state: ", state)
            action = get_chatgpt_action(state)

            # Check if the action is 2 and use last_action if available
            if action == 2 and last_action is not None:
                action = last_action  # Use the last action
                print(f"Action 2 detected, using last action: {action}")

            # Take the step in the environment using the action
            next_state, reward = env.step(action)
            run_reward += reward
            state = next_state

            # Update last_action with the current action
            last_action = action

            # Introduce a small delay to avoid hitting API rate limits
            time.sleep(1)

        rewards_for_runs.append(run_reward)
        print(f"Run {run + 1}: Total Reward: {run_reward}")

    # Calculate mean reward and standard deviation across the runs
    mean_reward = np.mean(rewards_for_runs)
    std_reward = np.std(rewards_for_runs)

    return mean_reward, std_reward

def plot_rewards(mean_reward, std_reward):
    # Plot the reward as a bar with error bars (std deviation)
    plt.bar(0, mean_reward, yerr=std_reward, alpha=0.7, capsize=5, color='blue')
    plt.xticks([0], ['Episode 1'])
    plt.xlabel('Episode')
    plt.ylabel('Mean Total Reward')
    plt.title('Mean Reward for One Episode with Std Deviation (GPT-4)')
    plt.show()

if __name__ == "__main__":
    cycle_times = [2, 3]
    buffer_capacity = 3
    env = ProductionLine(cycle_times, buffer_capacity)

    # Evaluate the ChatGPT agent for one episode run 3 times
    mean_reward, std_reward = evaluate_chatgpt_agent(env, steps_per_episode=20, runs_per_episode=1)

    # Plot the rewards
    plot_rewards(mean_reward, std_reward)

