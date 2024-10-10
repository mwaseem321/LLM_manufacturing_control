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


# Set up OpenAI API key
openai.api_key = 'api-key-here'

# Function to call ChatGPT for action decision
def get_chatgpt_action(state):
    # Define the prompt based on the state
    prompt = f"""
    Consider a manufacturing system with two machines and one buffer. 
    Each machine has a dedicated processing time and the buffer has a maximum capacity. 
    The first machine processes materials and stores in-process parts into the buffer 
    whereas the second machine is loaded with these parts for further processing and a 
    part is considered complete after it is processed by this last machine. If the buffer 
    becomes empty, the last machine can get starved whereas if it is full, the first machine 
    can get blocked. There is one mobile robot which loads the parts to machines. The robot 
    position is defined by a machine number and need some time to travel between machines. 
    The production line runs continuously and the robot should be assigned an action based 
    on the system state. The actions for the robot include: assign to load machine 1 (referred 
    as 0), load machine 2 (referred as 1), no action can be taken (referred as 2). If the robot 
    is already travelling (robot status of 2 refers to travelling) to a machine or loading a 
    machine (robot status of 1 refers to loading), then no action can be taken. If the robot is
     free (robot status of 0 refers to free) and a machine is ready to be loaded (machine status 
     of 0 refers to machine is ready to be loaded), it can be assigned to that machine. If both 
     machines are ready to be loaded and the buffer neither empty nor full, the robot is assigned
      to the machine where it is currently positioned (robot position of 0 means robot is at 
      machine 1 and 1 mean robot is at machine 2). Every week, there is a new demand that needs to 
      be fulfilled. Demand is considered fulfilled only after the final machine complete that 
       product; not when the product is scheduled. Beyond the demand, remainder can be stored in 
       inventory for retail. With this context, consider a cycle time of 2 minutes for the first 
       machine, 3 minutes for the second and a buffer capacity of 3 parts.  Now, tell me if machine 
       1 is {state[0]}, machine 2 is {state[1]}, buffer has {state[2]} units, robot status is
{state[3]}, and robot position is {state[4]}, what action should be assigned to the robot? 
    The output must only provide the action with no details.
    """

    # Call OpenAI API with the new prompt structure
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,  # Adjust according to your needs
        temperature=0.7
    )

    # Extract action and reasoning from the response
    response_text = response.choices[0].text.strip()

    # Extract the action (0 or 1) from the response
    # Assuming the response is formatted as "Action: X, Reason: ..."
    try:
        action = int(response_text.split('Action:')[1].split(',')[0].strip())
        reason = response_text.split('Reason:')[1].strip()
    except Exception as e:
        print(f"Error parsing response: {response_text}, Error: {e}")
        action = 0  # Default action in case of an error
        reason = "Default action due to parsing error."

    return action, reason

if __name__ == "__main__":
    cycle_times = [2, 3]
    buffer_capacity = 3
    env = ProductionLine(cycle_times, buffer_capacity)

    # Calculate the state size based on your specifications
    state_size = (2 ** 2) * (
                buffer_capacity + 1) * 3 * 2  # 2 machines (2 states each), buffer capacity + 1 states, robot states, robot position
    agent = QLearningAgent(state_size=state_size, action_size=2)  # 2 actions (machine 0 or 1)

    episodes = 5  # Number of episodes to train
    steps_per_episode = 100

    log_data = []
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(steps_per_episode):
            print("state: ", state)
            state_index = state_to_index(state, buffer_capacity)

            # Use ChatGPT to decide the action
            action, reason = get_chatgpt_action(state)
            print(f"Action chosen by ChatGPT: {action}, Reason: {reason}")

            # Take the step in the environment using the action returned by ChatGPT
            next_state, reward = env.step(action)

            # Log the episode, step, state, action, and reward along with the reason
            log_data.append(
                f"Episode: {episode + 1}, Step: {step + 1}, State: {tuple(state)}, Action: {action}, Reason: {reason}, Reward: {reward}"
            )

            # Update Q-values using traditional Q-learning (optional if needed)
            next_state_index = state_to_index(next_state, buffer_capacity)
            agent.update_q_value(state_index, action, reward, next_state_index)
            total_reward += reward
            state = next_state

            # Introduce a small delay to avoid hitting API rate limits (if necessary)
            time.sleep(1)

        agent.decay_exploration()  # If you are still using exploration logic
        rewards_per_episode.append(total_reward)  # Append total reward for this episode
        print(
            f"Episode {episode + 1}: Total Reward: {total_reward}, Exploration Rate: {agent.exploration_rate:.3f}")
