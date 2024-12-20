"""
This program integrates simulation mode to control a robotic system (EMERGE robot). 
It uses random joint angle changes and goal positions to simulate the robot's movements. 
The program calculates distances, unit vectors, and normalized distances between the robot's 
current position and goal positions. It collects learning data for each step of the simulation, 
including the robot's movements and corresponding positions, and exports the results to a CSV file.
A progress bar is displayed during the simulation to indicate the current status and estimated remaining time.
"""

import time, copy, os, math, random
from emerge_joint_handler import *
from sim_joint_handler import *
from joint_handler import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import pandas as pd
from random import randint as ri
from random import uniform as ru

#  Get the current working directory and display system paths
route = sys.path
print(route)

# Generate a timestamp for file saving
timestr = time.strftime("%Y_%d%m_%H%M")

# Simulation settings and initialization
is_sim = True
client = RemoteAPIClient("localhost", 23000)
sim = client.getObject("sim")


# Determine whether the program runs in simulation or physical mode
def type_connection(SimActivated):

    if SimActivated == True:
        print("---------Simulation mode is activated---------\n")
        # Initialize simulation
        handler = JointHandler(is_sim, client, sim)

    else:
        #
        print("------------Physical mode is activated-----------\n")
        print(
            "------------World model could be only trained in the simulation mode-----------\n"
        )
        print("------------Program is ended-----------\n")
        exit()
    return handler


# Load predefined goal positions from a file
def load_goal_positions(file_path):
    try:
        df = pd.read_csv(file_path)
        df["Next Position"] = df["Next Position"].apply(eval)  # Convert string to tuple
        goal_positions = df["Next Position"].tolist()
        return goal_positions
    except Exception as e:
        print(f"Error loading goal positions: {e}")
        return []


# Get the next goal position from the loaded list
def get_next_goal_position(goal_positions, current_step):
    index = current_step % len(
        goal_positions
    )  # Loop back to start if index exceeds list
    return goal_positions[index]


# Reset robot joints to their home positions
def set_home(object):
    home_pos = 0
    print("\nRobot is going to home positions")
    for joint_n in range(object.num_joints):
        joint = object.joint_ids[joint_n]
        object.setJointTargetPosition(joint, home_pos)
        # print(f"Joint {joint} is set on the start position {next_pos:.2f} rad.")
    print("Robot is in home positions\n")


# Generate a random goal position within a defined range
def get_random_goal(max_distance):
    goal_position = tuple(random.uniform(0, max_distance) for _ in range(3))
    print(f"Random goal set on: {[round(pos, 3) for pos in goal_position]}")
    return goal_position


# Calculate the unit vector between two points
def calculate_unit_vector(point_a, point_b):
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    dz = point_b[2] - point_a[2]
    magnitude = math.sqrt(dx**2 + dy**2 + dz**2)
    if magnitude == 0:
        return (0, 0, 0)
    return (dx / magnitude, dy / magnitude, dz / magnitude)


# Calculate the distance between two points
def calculate_distance(point_a, point_b):
    return math.sqrt(sum((b - a) ** 2 for a, b in zip(point_a, point_b)))


# Normalize a distance based on the maximum possible axis length
def normalize_distance(max_axis, distance):
    max_distance = math.sqrt(2) * max_axis
    return distance / max_distance


# Generate random changes for robot joints
def rand_gen(obj):
    random_action = []
    change = []
    for num in range(obj.num_joints):
        if num == 0:
            alpha = (ri(0, 9000) * math.pi / 180) / 100
            random_action.append(alpha)
        elif num == 1:
            beta = (ri(0, 9000) * math.pi / 180) / 100
            random_action.append(beta)
        elif num == 2:
            # Nastav gama na nulu
            gamma = 0
            random_action.append(gamma)

    # print(f"Random action generated: {random_action}")
    change = random_action
    return change


# Class to store learning data for each step
class LearningData:
    def __init__(
        self,
        index,
        previous_position,
        new_position,
        goal_position,
        previous_distance,
        previous_norm_distance,
        previous_unit_vector,
        new_distance,
        new_norm_distance,
        new_unit_vector,
        change_to_next,
    ):
        self.index = index
        self.current_position = previous_position
        self.next_position = new_position
        self.goal_position = goal_position
        self.current_distance = previous_distance
        self.current_norm_distance = previous_norm_distance
        self.current_unit_vector = previous_unit_vector
        self.next_distance = new_distance
        self.next_norm_distance = new_norm_distance
        self.next_unit_vector = new_unit_vector
        self.action = change_to_next


# Class to store and manage all learning data
class Data:
    def __init__(self):
        self.info = []

    def add_learning_data(self, learning_data):
        self.info.append(copy.deepcopy(learning_data))

    def to_dataframe(self, sequence_index):
        data = {
            "Step Index": [f"{ld.index}" for ld in self.info],
            "Current Position": [ld.current_position for ld in self.info],
            "Next Position": [ld.next_position for ld in self.info],
            "Goal Position": [ld.goal_position for ld in self.info],
            "Current Distance": [ld.current_distance for ld in self.info],
            "Current Norm Distance": [ld.current_norm_distance for ld in self.info],
            "Current Unit Vector": [ld.current_unit_vector for ld in self.info],
            "Next Distance": [ld.next_distance for ld in self.info],
            "Next Norm Distance": [ld.next_norm_distance for ld in self.info],
            "Next Unit Vector": [ld.next_unit_vector for ld in self.info],
            "Action": [ld.action for ld in self.info],
        }
        return pd.DataFrame(data)


# Display a progress bar in the terminal
def update_progress_bar(current, total, start_time, bar_length=50):
    progress = current / total
    block = int(bar_length * progress)
    bar = "#" * block + "-" * (bar_length - block)
    percent = progress * 100

    # Calculate elapsed and remaining time
    elapsed_time = time.time() - start_time
    estimated_total_time = (elapsed_time / current) * total if current > 0 else 0
    remaining_time = estimated_total_time - elapsed_time

    # Format remaining time as HH:MM:SS
    remaining_hours, remaining_seconds = divmod(int(remaining_time), 3600)
    remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)
    print(
        f"\rProgress: [{bar}] {percent:.2f}% | Remaining Time: {remaining_hours:02}:{remaining_minutes:02}:{remaining_seconds:02}",
        end="",
        flush=True,
    )


def main():
    print("Current working directory:", os.getcwd())
    robot = type_connection(is_sim)  # Connect to the robot in simulation
    robot.connectEMERGE()
    robot.loadEMERGE()
    peak = robot.obj_ids[1]
    goal = robot.obj_ids[2]
    sequences = 2
    steps = 5

    all_data = []
    path = (
        route[0]
        + f"/Data/DataForWM/2Angles/EMERGEDist_LearningDataForWM_2Angles_{timestr}.csv"
    )
    max_axis = 0.3  # Maximum axis length in simulation in unit of sim

    first_goal_position = (max_axis, max_axis, max_axis)

    print("\nJoints:", robot.joint_ids)
    print(f"Object: {robot.obj_ids}, Peak:{peak}")
    print(f"Sequences: {sequences} Steps: {steps}")

    print("---------Movement is started---------\n")
    start_time = time.time()  # Start time for the simulation
    # time.sleep(0.2)
    goal_position = first_goal_position
    for seq_idx in range(1, sequences + 1):
        data = Data()
        set_home(robot)
        # goal_position = get_random_goal(max_axis)
        # goal_position = get_next_goal_position(goal_positions, seq_idx)
        current_position = robot.getObjectPosition(peak)
        current_distance = calculate_distance(current_position, goal_position)
        current_norm_distance = normalize_distance(max_axis, current_distance)
        current_unit_vector = calculate_unit_vector(current_position, goal_position)

        for step_idx in range(1, steps + 1):
            # Update the progress bar with time estimation
            total_steps = sequences * steps  # Total number of steps
            current_step = (seq_idx - 1) * steps + step_idx
            update_progress_bar(current_step, total_steps, start_time)

            # Set point to the goal position in sim
            robot.setObjectPosition(goal, goal_position)
            real_g_x, real_g_y, real_g_z = robot.getObjectPosition(goal)
            real_goal_pos = [real_g_x, real_g_y, real_g_z]
            print(f"Goal position:{[round(pos, 3) for pos in real_goal_pos]}")

            change_to_next = rand_gen(robot)
            robot.setAllJointTargetPositions(change_to_next)
            next_position = robot.getObjectPosition(peak)
            next_distance = calculate_distance(next_position, goal_position)
            next_norm_distance = normalize_distance(max_axis, next_distance)
            next_unit_vector = calculate_unit_vector(next_position, goal_position)

            learning_data = LearningData(
                index=float(f"{seq_idx}.{step_idx}"),
                previous_position=current_position,
                new_position=next_position,
                goal_position=goal_position,
                previous_distance=current_distance,
                previous_norm_distance=current_norm_distance,
                previous_unit_vector=current_unit_vector,
                new_distance=next_distance,
                new_norm_distance=next_norm_distance,
                new_unit_vector=next_unit_vector,
                change_to_next=change_to_next,
            )
            print(
                f"Goal: {[round(pos, 3) for pos in goal_position]},\nCur= Pos: {current_position} Dis: {current_distance} Norm: {current_norm_distance} Vector: {current_unit_vector} \nNext= Pos: {next_position} Dis: {next_distance} Norm: {next_norm_distance} Vector: {next_unit_vector}\n"
            )
            data.add_learning_data(learning_data)

            current_position = next_position
            current_distance = next_distance
            current_norm_distance = next_norm_distance
            current_unit_vector = next_unit_vector
            if step_idx == 1:
                goal_position = next_position
                print(
                    f"Goal positions set on: {[round(pos, 3) for pos in goal_position]}"
                )
            print(f"[Sequence {seq_idx}] step {step_idx} is finished")

        print(f"[Sequence {seq_idx}] is finished")
        sequence_df = data.to_dataframe(seq_idx)
        all_data.append(sequence_df)
    # export(data)
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(path, index=False)
        print(f"All data has been successfully saved.")

    robot.unloadEMERGE()
    robot.disconnectEMERGE()


if __name__ == "__main__":
    main()
