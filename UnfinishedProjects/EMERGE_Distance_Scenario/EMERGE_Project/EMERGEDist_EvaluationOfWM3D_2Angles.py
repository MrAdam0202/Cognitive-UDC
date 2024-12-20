"""
This program integrates simulation mode to control a robotic system (EMERGE robot). 
It utilizes a neural network-based World Model (WM) to predict the robot's next 
normalized distance and unit vector in 3D space based on the current position, 
randomly generated joint angle changes, and the goal position. The program evaluates 
the prediction by comparing it to the robot's actual actions and generates visualizations 
to analyze the performance, including 3D plots and comparisons of errors in normalized distances.
"""

import time, copy, os, sys, random
from emerge_joint_handler import *
from sim_joint_handler import *
from joint_handler import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import pandas as pd
from random import randint as ri
from random import uniform as ru
import numpy as np
from numpy import sqrt
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
from matplotlib.ticker import MaxNLocator

# Get the current working directory and display system paths
route = sys.path
print(route)

# Generate a timestamp for file saving
timestr = time.strftime("_%Y_%d%m_%H%M")

is_sim = True
client = RemoteAPIClient("localhost", 23000)
sim = client.getObject("sim")


# Determine whether the program runs in simulation or physical mode
def type_connection(SimActivated):

    if SimActivated == True:
        print("---------Simulation mode is activated---------\n")
        # Inicializace simulace
        handler = JointHandler(is_sim, client, sim)

    else:
        #
        print("------------Physical mode is activated-----------\n")
        print("------------Move for physical mode is not prepared yet-----------\n")
        print("------------Program is ended-----------\n")
        exit()
    return handler


# Generate random changes for robot joints within specified bounds
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
            gamma = 0
            random_action.append(gamma)

    # print(f"Random action generated: {random_action}")
    change = random_action
    return change


# Compute the unit vector and 3D distance between a current position and a goal position
def compute_unit_vector_and_distance(pos, goal_pos):

    delta_x = goal_pos[0] - pos[0]
    delta_y = goal_pos[1] - pos[1]
    delta_z = goal_pos[2] - pos[2]

    # Distance in 3D space
    distance_3d = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    # Compute unit vector
    if distance_3d != 0:
        unit_vector = (
            round(delta_x / distance_3d, 9),
            round(delta_y / distance_3d, 9),
            round(delta_z / distance_3d, 9),
        )
    else:
        unit_vector = (
            0.0,
            0.0,
            0.0,
        )

    return unit_vector, distance_3d


# Predict next position from current pos and changes
def predict_next_positions(
    current_pos, action_to_next, goal_pos, max_distance, W_model
):
    if len(current_pos) != 3 or len(action_to_next) != 3 or len(goal_pos) != 3:
        raise ValueError(
            "Inputs `current_pos`, `action_to_next`, and `goal_pos` must have exactly 3 elements each."
        )

    # Compute current unit vector and normalized distance
    current_unit_vector, current_distance = compute_unit_vector_and_distance(
        current_pos, goal_pos
    )
    current_norm_distance = current_distance / max_distance

    # Prepare input features for the model
    input_features = np.array(
        [[current_norm_distance, *current_unit_vector, *action_to_next]]
    )

    # Predict the next normalized distance and unit vector
    next_norm_distance, *next_unit_vector = W_model.predict(input_features, verbose=0)[
        0
    ]

    # Return the predicted values
    return next_norm_distance, next_unit_vector


# Reset all robot joints to their home position (0 radians).
def set_home(object):
    home_pos = 0
    # print("\n-----Robot is going to home positions-----")
    for joint_n in range(object.num_joints):
        joint = object.joint_ids[joint_n]
        object.setJointTargetPosition(joint, home_pos)
        # print(f"Joint {joint} is set on the start position {next_pos:.2f} rad.")

    # print("-----Robot is in home positions-----\n")


# Set a goal position randomly and retrieves the corresponding coordinates of the goal
def set_goal(object, peak):
    goal_angles = rand_gen(object)
    # print("\n-----Robot is going to home positions-----")
    object.setAllJointTargetPositions(goal_angles)
    cur_x, cur_y, cur_z = object.getObjectPosition(peak)
    goal_pos = [cur_x, cur_y, cur_z]

    # print(f"Goal position set on:{[round(pos,3) for pos in goal_pos]}\n")
    return goal_pos


# Store data for perception and analysis
class perception_data:
    def __init__(self):
        self.info_data = []


# Represent the robot's perceptions at each step
class perceptions:
    def __init__(self):

        self.index = 0
        self.goal_pos = []
        self.current_pos = []
        # Info about real and predict state
        self.current_distance = []
        self.current_norm_distance = []
        self.current_unit_vector = []
        self.action_to_next = []
        self.pred_norm_distance = []
        self.pred_unit_vector = []
        self.next_norm_distance = []
        self.next_unit_vector = []
        self.next_position = []
        self.real_error_norm_distance = []


# Compute differences between distances
def real_error_distance(self):
    self.real_error_norm_distance = abs(
        self.next_norm_distance - self.pred_norm_distance
    )


# Export perception data to a CSV file
def export(data):
    # Dump data in file
    path = (
        route[0]
        + f"/Data/DataForWM/2Angles/EMERGEDist_EvaluationOfWM_2Angles_{timestr}.csv"
    )

    df = pd.DataFrame(data.info_data)
    df.to_csv(
        path,
        index=False,
    )
    print("File of data created")
    csv_path = path
    return csv_path


# Plot the difference between actual and predicted normalized distances
def plot_comparison(info):
    timestr = time.strftime("_%Y_%d%m_%H%M")
    # Extract data from perception data
    next_norm_distances = [data["next_norm_distance"] for data in info.info_data]
    predicted_norm_distances = [data["pred_norm_distance"] for data in info.info_data]

    # Use a range for step numbers
    steps = range(1, len(next_norm_distances) + 1)

    # Create plot
    plt.figure(figsize=(10, 6))

    # Plot real and predicted normalized distances
    plt.plot(
        steps, next_norm_distances, label="Real Norm Distance", color="blue", marker="o"
    )
    plt.plot(
        steps,
        predicted_norm_distances,
        label="Predicted Norm Distance",
        color="orange",
        marker="x",
    )

    # Add title, labels, and legend
    plt.title("Comparison of Real and Predicted Normalized Distances")
    plt.xlabel("Steps")
    plt.ylabel("Normalized Distance")
    plt.xticks(steps)  # Set x-ticks to integers
    plt.legend()

    # Save the graph as an image
    plt.tight_layout()
    plt.savefig(
        route[0]
        + f"/Data/DataForWM/2Angles/GraphComparisonOfNormDistances_2Angles_{timestr}.png"
    )
    print(f"Graph saved: GraphComparisonOfNormDistances_{timestr}.png")
    plt.close()


def plot_real_error_norm_distance(info):
    timestr = time.strftime("_%Y_%d%m_%H%M")
    # Extract real error data from perception data and flatten any nested lists
    real_error_norm_distances = []
    for data in info.info_data:
        error_value = data.get("real_error_norm_distance", None)
        if isinstance(error_value, (float, np.float64)):
            real_error_norm_distances.append(float(error_value))
        elif isinstance(error_value, list):
            real_error_norm_distances.extend(
                [
                    float(item)
                    for item in error_value
                    if isinstance(item, (float, np.float64))
                ]
            )

    # Clean None values
    real_error_norm_distances = [
        val for val in real_error_norm_distances if val is not None
    ]

    # Steps for X-axis
    steps = range(1, len(real_error_norm_distances) + 1)

    # Create plot
    plt.figure(figsize=(12, 6))  # Wider figure for better X-axis display

    # Plot real error normalized distances
    plt.plot(
        steps,
        real_error_norm_distances,
        label="Real Error Norm Distance",
        color="red",
        marker="o",
    )

    # Add title, labels, and legend
    plt.title("Real Error in Normalized Distances")
    plt.xlabel("Steps")
    plt.ylabel("Error (Normalized Distance)")

    # Adjust X-axis ticks automatically
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune="both", nbins=15))

    # Rotate X-axis labels if needed
    plt.xticks(rotation=45)

    plt.legend()

    # Save the graph as an image
    plt.tight_layout()
    plt.savefig(
        route[0]
        + f"/Data/DataForWM/2Angles/GraphRealErrorNormDist_2Angles_{timestr}.png"
    )
    print(f"Graph saved: GraphRealErrorNormDist_2Angles_{timestr}.png")
    plt.close()


def random_goal(max_distance):
    goal_position = (
        random.uniform(0, max_distance),
        random.uniform(0, max_distance),
        random.uniform(0, max_distance),
    )
    return goal_position


# Function to load data and plot positions
def load_and_plot_for_3D(csv_path):
    timestr = time.strftime("_%Y_%d%m_%H%M")
    # Load the CSV file, specifying the first row as headers and skipping the second row
    df = pd.read_csv(csv_path, header=0, skiprows=[1])  # Skip second row
    df.columns = df.columns.str.strip()  # Clean column names (remove spaces)
    save_path = (
        route[0] + f"/Data/DataForWM/2Angles/3DGraphOfTestedAndGoalPos_{timestr}.png"
    )
    # Debugging: Print column names
    # print("Detected columns:", df.columns)

    # Check for required columns
    if "current_pos" in df.columns and "goal_pos" in df.columns:
        # Parse Current Position and Goal Position
        current_positions = df["current_pos"].apply(parse_tuple)
        goal_positions = df["goal_pos"].apply(parse_tuple)

        # Extract x, y, z coordinates
        current_x, current_y, current_z = zip(*current_positions)
        goal_x, goal_y, goal_z = zip(*goal_positions)

        # Create 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot Goal Positions first for layering
        ax.scatter(
            goal_x, goal_y, goal_z, color="red", s=20, label="Goal Position", zorder=2
        )
        ax.scatter(
            current_x,
            current_y,
            current_z,
            color="blue",
            s=10,
            label="Tested Position",
            zorder=1,
        )

        # Add labels and title
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("3D Plot of Tested and Goal Positions")
        ax.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Graph saved successfully at {save_path}")
        plt.show()
    else:
        print("Required columns 'Current Position' or 'Goal Position' not found.")


# Parse position tuples from strings
def parse_tuple(position_str):
    try:
        return ast.literal_eval(position_str)
    except:
        return None


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

    # Format remaining time as MM:SS
    remaining_minutes, remaining_seconds = divmod(int(remaining_time), 60)
    print(
        f"\rProgress: [{bar}] {percent:.2f}% | Remaining Time: {remaining_minutes:02}:{remaining_seconds:02}\n",
        end="",
        flush=True,
    )


def main():
    print("Current working directory:", os.getcwd())
    robot = type_connection(is_sim)  # creating object like robot
    robot.connectEMERGE()
    robot.loadEMERGE()
    print(robot.obj_ids)
    peak = robot.obj_ids[1]
    goal = robot.obj_ids[2]
    percept = perceptions()
    info = perception_data()

    sequences = 5
    steps = 5
    max_distance = 0.3  # units of sim

    world_model = load_model(
        route[0]
        + "/Data/Models/EMERGEDist_WorldModelforRobot_2Angles_2024_1812_0702.h5"
    )
    # goals_rad = [0.5, 0.5, 0.5]
    # info about robot
    print("\nJoints:", robot.joint_ids)
    print(f"Object: {robot.obj_ids}, Peak:{peak}")

    print("---------Movement is started---------\n")
    start_time = time.time()  # Start time for the simulation

    for joint_n in range(robot.num_joints):
        joint = robot.joint_ids[joint_n]
        print(f"Checking joint ID: {joint}")

    print(f"Joint IDs: {robot.joint_ids}")

    # time.sleep(0.2)
    for seq_n in range(sequences):
        set_home(robot)
        # goal_pos = random_goal(max_distance)
        # goal_pos = [-0.013266957430159486, -0.025056737196796314, 0.3295638731333684]
        goal_pos = set_goal(robot, peak)
        # Set point to the goal position in sim
        robot.setObjectPosition(goal, goal_pos)
        if seq_n == 0:
            set_home(robot)
        else:
            robot.setAllJointTargetPositions(change)
        for step_n in range(steps):
            index = float(f"{seq_n}.{step_n}")
            # Update the progress bar with time estimation
            total_steps = sequences * steps  # Total number of steps
            current_step = seq_n * steps + step_n + 1
            update_progress_bar(current_step, total_steps, start_time)
            cur_x, cur_y, cur_z = robot.getObjectPosition(peak)
            current_pos = [cur_x, cur_y, cur_z]
            cur_unit_vector, cur_distance = compute_unit_vector_and_distance(
                current_pos, goal_pos
            )
            cur_norm_distance = cur_distance / (math.sqrt(2) * max_distance)

            change = rand_gen(robot)
            # for joint_n in range(robot.num_joints):
            #     current_angles[joint_n] = robot.getJointPosition(robot.joint_ids[joint_n])
            # action_to_next = [current_angles[joint_n] + change[joint_n] for joint_n in range(robot.num_joints)]
            pred_norm_distance, pred_unit_vector = predict_next_positions(
                current_pos, change, goal_pos, max_distance, world_model
            )

            robot.setAllJointTargetPositions(change)

            next_x, next_y, next_z = robot.getObjectPosition(peak)
            next_pos = [next_x, next_y, next_z]
            next_unit_vector, next_distance = compute_unit_vector_and_distance(
                next_pos, goal_pos
            )

            next_norm_distance = next_distance / (math.sqrt(2) * max_distance)

            percept.index = index
            percept.goal_pos = goal_pos
            percept.current_pos = current_pos
            percept.current_distance = cur_distance
            percept.current_norm_distance = cur_norm_distance
            percept.current_unit_vector = cur_unit_vector
            percept.action_to_next = change
            percept.pred_norm_distance = pred_norm_distance
            percept.pred_unit_vector = pred_unit_vector
            percept.next_norm_distance = next_norm_distance
            percept.next_unit_vector = next_unit_vector
            percept.next_position = next_pos

            real_error_distance(percept)

            info.info_data.append(copy.deepcopy(vars(percept)))
            # print(
            #     f"{index:.1f} Robot was in unit vector: {[round(pos, 3) for pos in current_pos]} and distance: {cur_distance:.2f} (norm distance: {cur_norm_distance:.2f}),\n"
            #     f"Change: {[round(pos, 3) for pos in change]},Robot predicted unit vector: {[round(pos, 3) for pos in pred_unit_vector]} predicted norm distance: {pred_norm_distance:.2f},\n"
            #     f"Robot reached unit vector: {[round(pos, 3) for pos in next_unit_vector]} reached norm distance: {next_norm_distance:.2f},\n"
            # )
            current_pos = next_pos.copy()

    print("---------Movement is ended---------\n")
    csv_path = export(info)
    print("---------Data exported---------\n")

    plot_comparison(info)
    plot_real_error_norm_distance(info)
    load_and_plot_for_3D(csv_path)
    robot.unloadEMERGE()
    robot.disconnectEMERGE()


if __name__ == "__main__":
    main()
