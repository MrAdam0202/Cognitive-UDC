"""
This program integrates simulation for controlling the EMERGE robot system. 
It uses a utility model (UM) based on neural networks and a world model (WM) to predict optimal motions 
and the future position of the robot in 3D space. 
The program scores robot actions by comparing predicted and actual positions during motion. 
It executes predefined test cases with start and goal positions, computes robot movements to minimize the distance to the goal, 
and exports the collected data for further analysis.
"""

import time, copy, os, sys
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

# Get the current working directory and display system paths
route = sys.path
print(route)

timestr = time.strftime("_%Y_%d%m_%H%M")

# Determine whether simulation or physical mode is used
is_sim = True

# Initialize the simulation client
client = RemoteAPIClient("localhost", 23000)
sim = client.getObject("sim")


# Select the mode of operation (simulation or physical mode)
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


# Generate random changes
def rand_gen(obj, data):
    for num in range(obj.num_joints):
        n = ru(-90, 90) * math.pi / 180
        data.changes.append(n)


# Compute unit vector and distance between two points in 3D space
def compute_unit_vector_and_distance(pos, goal_pos):
    delta_x = goal_pos[0] - pos[0]
    delta_y = goal_pos[1] - pos[1]
    delta_z = goal_pos[2] - pos[2]

    # Distance in 3D space
    distance_3d = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    # Compute unit vector
    if distance_3d != 0:
        unit_vector = (
            round(delta_x / distance_3d, 3),
            round(delta_y / distance_3d, 3),
            round(delta_z / distance_3d, 3),
        )
    else:
        unit_vector = (
            0.0,
            0.0,
            0.0,
        )

    return unit_vector, distance_3d


# Predict utility value using the Utility Model
def predict_utility(next_norm_distance, next_unit_vector, utility_model):
    # Prepare input features for the model
    input_data = np.array(
        [
            [
                next_unit_vector[0],
                next_unit_vector[1],
                next_unit_vector[2],
                next_norm_distance,
            ]
        ]
    )

    # Predict utility value
    utility_value = utility_model.predict(input_data, verbose=0)[0][0]
    return utility_value


# Predict the next position using the World Model
def predict_next_positions(
    current_pos, action_to_next, goal_pos, max_distance, W_model
):
    # Validate input
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
    )  # Shape: (1, 7)

    # Predict the next normalized distance and unit vector
    next_norm_distance, *next_unit_vector = W_model.predict(input_features, verbose=0)[
        0
    ]
    return next_norm_distance, next_unit_vector


# Predict the best movement direction and evaluate utility
def predict_next_move(
    robot_angles, U_model, goal_pos, unit, current_pos, W_model, max_distance, index
):
    # Possible changes in angles (alpha, beta, gamma)
    move_options = [
        (unit, 0, 0),
        (-unit, 0, 0),
        (0, unit, 0),
        (0, -unit, 0),
        (unit, unit, 0),
        (-unit, -unit, 0),
        (unit, -unit, 0),
        (-unit, unit, 0),
    ]

    direction_names = [
        f"+{math.degrees(unit):.1f}° on alpha",
        f"-{math.degrees(unit):.1f}° on alpha",
        f"+{math.degrees(unit):.1f}° on beta",
        f"-{math.degrees(unit):.1f}° on beta",
        f"+{math.degrees(unit):.1f}° on alpha, +{math.degrees(unit):.1f}° on beta",
        f"-{math.degrees(unit):.1f}° on alpha, -{math.degrees(unit):.1f}° on beta",
        f"+{math.degrees(unit):.1f}° on alpha, -{math.degrees(unit):.1f}° on beta",
        f"-{math.degrees(unit):.1f}° on alpha, +{math.degrees(unit):.1f}° on beta",
    ]

    best_move = None
    best_utility = -np.inf
    all_utilities = {}
    all_distances = {}
    new_x, new_y, new_z = None, None, None  # To store the coordinates of the best move
    best_distance = -np.inf

    for idx, move in enumerate(
        move_options
    ):  #  Iterates through the list, providing the index of each iteration along with the corresponding list element
        next_alpha = robot_angles[0] + move[0]
        next_beta = robot_angles[1] + move[1]
        next_gamma = robot_angles[2] + move[2]
        action_to_next = [next_alpha, next_beta, next_gamma]

        # Predict new position
        next_norm_distance, next_unit_vector = predict_next_positions(
            current_pos, action_to_next, goal_pos, max_distance, W_model
        )

        # Calculate utility value for this new position
        pred_utility = predict_utility(next_norm_distance, next_unit_vector, U_model)
        pred_distance = next_norm_distance * max_distance

        all_utilities[direction_names[idx]] = pred_utility
        all_distances[direction_names[idx]] = pred_distance

        if pred_utility > best_utility:
            best_utility = pred_utility
            best_move = move
            best_angles = action_to_next
            best_direction = direction_names[idx]  # Save the best coordinates
            best_distance = pred_distance
    print(
        f"\n--------------------------------------------------------------------------------------------"
        f"\nCurrent Angles: {robot_angles[0]:.3f}, {robot_angles[1]:.3f}, {robot_angles[2]:.3f}; "
        f"Goal: {goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f};"
        f"\nUV{direction_names[0]}: {all_utilities[direction_names[0]]:.3f}, PredDist: {all_distances[direction_names[0]]:.3f} "
        f"\nUV{direction_names[1]}: {all_utilities[direction_names[1]]:.3f}, PredDist: {all_distances[direction_names[1]]:.3f} "
        f"\nUV{direction_names[2]}: {all_utilities[direction_names[2]]:.3f}, PredDist: {all_distances[direction_names[2]]:.3f} "
        f"\nUV{direction_names[3]}: {all_utilities[direction_names[3]]:.3f}, PredDist: {all_distances[direction_names[3]]:.3f} "
        f"\nUV{direction_names[4]}: {all_utilities[direction_names[4]]:.3f}, PredDist: {all_distances[direction_names[4]]:.3f} "
        f"\nUV{direction_names[5]}: {all_utilities[direction_names[5]]:.3f}, PredDist: {all_distances[direction_names[5]]:.3f} "
        f"\nUV{direction_names[6]}: {all_utilities[direction_names[6]]:.3f}, PredDist: {all_distances[direction_names[6]]:.3f} "
        f"\nUV{direction_names[7]}: {all_utilities[direction_names[7]]:.3f}, PredDist: {all_distances[direction_names[7]]:.3f} "
        f"\n--------------------------------------------------------------------------------------------"
    )
    print(
        f"\nStep: {index}, Robot choose: {best_direction} with Utility Value: {best_utility:.3f};"
        f"\nPredicted distance: {best_distance:.3f};Goal: {goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f};"
        f"\nNext_angles: {[round(ang, 3) for ang in best_angles]};"
    )
    return best_move, best_direction, best_utility, all_utilities


# Load test positions from a file
def load_test_positions(file_path):
    """
    Loads test positions from a file in the format:
    alpha_rad_start, beta_rad_start, gamma_rad_start; x_goal, y_goal, z_goal
    """
    test_positions = []  # List to store test positions
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                # Split line into start angles and goal coordinates
                start_angles, goal_coords = line.split(";")

                # Parse start angles
                start_alpha, start_beta, start_gamma = map(
                    float, start_angles.split(",")
                )

                # Parse goal coordinates
                goal_x, goal_y, goal_z = map(float, goal_coords.split(","))

                # Append to test positions as a tuple
                test_positions.append(
                    ([start_alpha, start_beta, start_gamma], [goal_x, goal_y, goal_z])
                )
            except ValueError:
                print(f"Error processing line: {line}")
                continue
    return test_positions


# Max distance for normalize
def calculate_max_distance(L1, L2):
    max_distance = sqrt((L1) ** 2 + (L2) ** 2)
    print(f"MaxDistance:{max_distance:.3f}")
    return max_distance


# Save collected perception data to a CSV file
def export(data):
    path = route[0] + f"/Data/EMERGEDist_DataFromPerceptions_{timestr}.csv"
    df = pd.DataFrame(data.info_data)
    df.to_csv(
        path,
        index=False,
    )
    print("File of data created")


# Classes to store perception and data
class perception_data:
    def __init__(self):
        self.info_data = []


class perceptions:
    def __init__(self):

        self.index = 0
        self.goal_position = []

        # Info about current state
        self.current_pos = []
        self.current_angles = []
        self.current_distance = 0

        # Info about next state
        self.next_pos = []
        self.change_to_next = []
        self.next_angles = []
        self.next_distance = 0
        self.next_utility = 0
        self.utility_of_changes = []


# Set robot to start position
def set_start_position(object, start_angles):
    print("\n -----Robot is going to start positions-----")
    print(f"Start angles:{[round(ang, 3) for ang in start_angles]}")
    for joint_n in range(object.num_joints):
        joint = object.joint_ids[joint_n]

        object.setJointTargetPosition(joint, start_angles[joint_n])
        # print(f"Joint {joint} is set on the start position {start_angles[joint_n]:.2f} rad.")
    print("-----Robot is in start positions-----\n")


# Set robot to home position
def set_home(object):
    home_pos = 0
    print("\n-----Robot is going to home positions-----")
    for joint_n in range(object.num_joints):
        joint = object.joint_ids[joint_n]
        object.setJointTargetPosition(joint, home_pos)
        # print(f"Joint {joint} is set on the start position {next_pos:.2f} rad.")

    print("-----Robot is in home positions-----\n")


# Save states to data of perception
def save_perception(
    obj,
    index,
    goal_pos,
    curr_pos,
    curr_ang,
    curr_dist,
    next_pos,
    next_change,
    next_ang,
    next_dist,
    next_utility,
    utilities,
):

    obj.index = index
    obj.goal_position = goal_pos

    # Info about current state
    obj.current_pos = curr_pos
    obj.current_angles = curr_ang
    obj.current_distance = curr_dist

    # Info about next state
    obj.next_pos = next_pos
    obj.change_to_next = next_change
    obj.next_angles = next_ang
    obj.next_distance = next_dist
    obj.next_utility = next_utility
    obj.utility_of_changes = utilities


def main():
    print("Current working directory:", os.getcwd())
    robot = type_connection(
        is_sim
    )  # Initialize the robot connection based on the simulation or physical mode.
    robot.connectEMERGE()  # Establish connection with the EMERGE robotic system.
    robot.loadEMERGE()  # Load the robotic system's components, such as joints and objects.
    peak = robot.obj_ids[1]  # The robot's "peak" or end effector
    goal = robot.obj_ids[2]  # The point in simulation for demonstration goal position
    percept = perceptions()
    info = perception_data()

    max_distance = 0.3  # Units of sim
    unit = math.radians(1)  # degrees to rads
    max_steps = 1000
    steps = 0
    tolerance_to_reach_goal = 0.015  # Units of sim

    utility_model = load_model(
        route[0]
        + f"/Data/Models/SphericalMotions_UtilityModelfor3DRobot__2024_0912_1601.h5"
    )
    world_model = load_model(
        route[0]
        + "/Data/Models/EMERGEDist_WorldModelforRobot_2Angles_2024_1812_0702.h5"
    )

    test_positions_file = route[0] + f"/Data/RobotTestPositions.txt"
    test_positions = load_test_positions(
        test_positions_file
    )  # Load the list of test positions, each containing starting joint angles and goal coordinates.

    # info about robot
    print("\nJoints:", robot.joint_ids)
    print(f"Object: {robot.obj_ids}, Peak:{peak}, Goal: {goal}")

    print("---------Movement is started---------\n")

    for joint_n in range(robot.num_joints):
        joint = robot.joint_ids[joint_n]
        print(f"Checking joint ID: {joint}")

    print(f"Joint IDs: {robot.joint_ids}")

    # Execution of all tested positions
    for idx, (start_angles, goal_position) in enumerate(test_positions):
        current_pos = []
        current_angles = [0] * robot.num_joints
        next_angles = [0, 0, 0]
        current_distance = None
        goal_pos = goal_position.copy()
        robot.setObjectPosition(goal, goal_pos)
        real_g_x, real_g_y, real_g_z = robot.getObjectPosition(goal)
        real_goal_pos = [real_g_x, real_g_y, real_g_z]
        print(f"Goal position:{[round(pos, 3) for pos in real_goal_pos]}")
        steps = 0
        # set_home(robot)
        set_start_position(robot, start_angles)

        for joint_n in range(robot.num_joints):
            current_angles[joint_n] = robot.getJointPosition(robot.joint_ids[joint_n])

        cur_x, cur_y, cur_z = robot.getObjectPosition(peak)
        current_pos = [cur_x, cur_y, cur_z]
        print(
            f"Robot is in positions: {[round(pos, 3) for pos in current_pos]}, Joints are in positions: {[round(ang, 3) for ang in current_angles]}rad."
        )

        print("\n=====Start predict=====\n")

        while (
            True
        ):  # Loop until the robot reaches the goal or the maximum number of steps is exceeded.
            index = float(f"{idx}.{steps}")
            best_move, next_change, next_utility, next_utilities = predict_next_move(
                current_angles,
                utility_model,
                goal_pos,
                unit,
                current_pos,
                world_model,
                max_distance,
                index,
            )
            next_angles = current_angles.copy()
            # Update angles based on best movement
            next_angles[0] += best_move[0]
            next_angles[1] += best_move[1]
            next_angles[2] += best_move[2]

            for joint_n in range(robot.num_joints):
                next_pos = next_angles[joint_n]
                joint = robot.joint_ids[joint_n]
                robot.setJointTargetPosition(joint, next_pos)
                # print(f"Joint {joint} is set on the start position {next_pos:.2f} rad.")

            next_x, next_y, next_z = robot.getObjectPosition(peak)
            next_pos = [next_x, next_y, next_z]
            for joint_n in range(robot.num_joints):
                next_angles[joint_n] = robot.getJointPosition(robot.joint_ids[joint_n])

            _, next_distance = compute_unit_vector_and_distance(next_pos, goal_pos)
            print(
                f"New position is: {[round(pos, 3) for pos in next_pos]}",
                f"New distance is: {next_distance:.3f}",
            )
            save_perception(
                percept,
                index,
                goal_pos,
                current_pos,
                current_angles,
                current_distance,
                next_pos,
                next_change,
                next_angles,
                next_distance,
                next_utility,
                next_utilities,
            )
            info.info_data.append(copy.deepcopy(vars(percept)))
            current_angles = next_angles
            current_pos = next_pos
            current_distance = next_distance
            if next_distance < tolerance_to_reach_goal:
                print(f"Robot reached the goal in {steps} steps!")
                break
            steps += 1
            if steps > max_steps:
                print(f"Test failed after {max_steps} steps!")
                break

    export(info)  # Export all collected perception data for analysis.
    robot.unloadEMERGE()  # Unload the robot's components after completing all test cases.
    robot.disconnectEMERGE()  # Disconnect from the robotic system or simulation environment.


if __name__ == "__main__":
    main()
