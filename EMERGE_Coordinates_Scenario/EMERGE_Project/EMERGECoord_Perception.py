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


route = sys.path
print(route)
timestr = time.strftime("_%Y_%d%m_%H%M")

is_sim = True
client = RemoteAPIClient("localhost", 23000)
sim = client.getObject("sim")


# Determine the type of connection (simulation or physical) and initialize handlers
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


# Generate random angle changes for the robot's joints
def rand_gen(obj, data):
    for num in range(obj.num_joints):
        n = ru(-90, 90) * math.pi / 180
        data.changes.append(n)


# Calculate the unit vector and the distance between the current position and the target position by projecting the 3D vector onto the 2D plane P
def compute_unit_vector_and_distance(next_pos, goal_pos):
    rel_x = goal_pos[0] - next_pos[0]
    rel_y = goal_pos[1] - next_pos[1]
    rel_z = goal_pos[2] - next_pos[2]
    rel_xy = math.sqrt(rel_x**2 + rel_y**2)  # 2D distance in the XY plane

    distance_3d = math.sqrt(rel_xy**2 + rel_z**2)  # Full 3D distance calculation

    # Calculation of the unit vector for projection on the plane P
    if distance_3d != 0:
        unit_vector = (round(rel_xy / distance_3d, 3), round(rel_z / distance_3d, 3))
    else:
        unit_vector = (0.0, 0.0)

    return unit_vector, distance_3d


# Predict the utility value of a movement based on position and distance
def predict_utility(next_pos, goal_pos, utility_model, max_distance):
    unit_vector, distance = compute_unit_vector_and_distance(next_pos, goal_pos)
    norm_distance = distance / max_distance
    unit_vec_x, unit_vec_y = unit_vector
    input_data = np.array([[norm_distance, unit_vec_x, unit_vec_y]])

    # Predict utility value
    utility_value = utility_model.predict(input_data, verbose=0)[0][0]
    return utility_value, unit_vector, distance, norm_distance


# Predict the next position based on current position and angle changes
def predict_next_positions(current_pos, change, W_model):
    if len(current_pos) != 3 or len(change) != 3:
        raise ValueError(
            "Input `current_pos` and `change` must have exactly 3 elements each."
        )

    # Combine current_pos and change into a single input array
    input_features = np.array([list(current_pos) + list(change)])  # Shape: (1, 6)

    # Predict the next position
    next_x, next_y, next_z = W_model.predict(input_features, verbose=0)[
        0
    ]  # Unpack directly

    # Return the predicted values
    return next_x, next_y, next_z


# Function to predict the best movement direction based on angle changes
def predict_next_move(
    robot_angles, U_model, goal_pos, unit, current_pos, W_model, max_distance, index
):
    # Possible changes in angles (alpha, beta, gamma)
    move_options = [
        (unit, 0, 0),
        (-unit, 0, 0),
        (0, unit, 0),
        (0, -unit, 0),
        (0, 0, unit),
        (0, 0, -unit),
    ]
    direction_names = [
        f"+{math.degrees(unit)}° on alpha",
        f"-{math.degrees(unit)}° on alpha",
        f"+{math.degrees(unit)}° on beta",
        f"-{math.degrees(unit)}° on beta",
        f"+{math.degrees(unit)}° on gamma",
        f"-{math.degrees(unit)}° on gamma",
    ]

    best_move = None
    best_utility = -np.inf
    all_utilities = {}
    new_x, new_y, new_z = None, None, None  # To store the coordinates of the best move
    best_distance = -np.inf

    for idx, move in enumerate(
        move_options
    ):  #  Iterates through the list, providing the index of each iteration along with the corresponding list element
        next_alpha = robot_angles[0] + move[0]
        next_beta = robot_angles[1] + move[1]
        next_gamma = robot_angles[2] + move[2]
        changes = [next_alpha, next_beta, next_gamma]

        # Calculate new position based on updated angles
        new_x, new_y, new_z = predict_next_positions(current_pos, changes, W_model)
        next_pos = [new_x, new_y, new_z]
        # print(f"\nChanges for next position are {[round(ang, 3) for ang in changes]}")
        # print(
        #     f"Next position for {move} is {[round(pos, 3) for pos in next_pos]} and goal is {[round(pos, 3) for pos in goal_pos]}"
        # )
        # Calculate utility value for this new position
        pred_utility, pred_unit_vector, pred_distance, pred_norm_distance = (
            predict_utility(next_pos, goal_pos, U_model, max_distance)
        )

        all_utilities[direction_names[idx]] = pred_utility

        if pred_utility > best_utility:
            best_utility = pred_utility
            best_move = move
            best_direction = direction_names[idx]
            best_pos = next_pos  # Save the best coordinates
            best_distance = pred_distance
    # print(
    #     f"\nCurrent Angles: {robot_angles[0]:.3f}, {robot_angles[1]:.3f}, {robot_angles[2]:.3f}; "
    #     f"Goal: {goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f};"
    #     f"+{unit} rad on alpha: {all_utilities[f'+{unit} rad on alpha']:.3f}, "
    #     f"-{unit} rad on alpha: {all_utilities[f'-{unit} rad on alpha']:.3f}, "
    #     f"+{unit} rad on beta: {all_utilities[f'+{unit} rad on beta']:.3f}, "
    #     f"-{unit} rad on beta: {all_utilities[f'-{unit} rad on beta']:.3f}, "
    #     f"+{unit} rad on gamma: {all_utilities[f'+{unit} rad on gamma']:.3f}, "
    #     f"-{unit} rad on gamma: {all_utilities[f'-{unit} rad on gamma']:.3f} "
    # )
    print(
        f"\nStep: {index}, Robot choose: {best_direction} with Utility Value: {best_utility:.3f};"
        f"\nPredicted distance: {best_distance:.3f};Goal: {goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}, Predicted Coordinates: [{next_pos[0]:.3f}, {next_pos[1]:.3f}, {next_pos[2]:.3f}]"
    )
    return best_move, best_direction, best_utility, all_utilities


# Load test positions from a file
def load_test_positions(file_path):
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


def export(data):
    # Dump data in file
    path = route[0] + f"/Data/EMERGECoord_PerceptionData_{timestr}.csv"
    df = pd.DataFrame(data.info_data)
    df.to_csv(
        path,
        index=False,
    )
    print("File of data created")


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


def set_start_position(object, start_angles):
    print("\n -----Robot is going to start positions-----")
    print(f"Start angles:{[round(ang, 3) for ang in start_angles]}")
    for joint_n in range(object.num_joints):
        joint = object.joint_ids[joint_n]

        object.setJointTargetPosition(joint, start_angles[joint_n])
        # print(f"Joint {joint} is set on the start position {start_angles[joint_n]:.2f} rad.")
    print("-----Robot is in start positions-----\n")


def set_home(object):
    home_pos = 0
    print("\n-----Robot is going to home positions-----")
    for joint_n in range(object.num_joints):
        joint = object.joint_ids[joint_n]
        object.setJointTargetPosition(joint, home_pos)
        # print(f"Joint {joint} is set on the start position {next_pos:.2f} rad.")

    print("-----Robot is in home positions-----\n")


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
    robot = type_connection(is_sim)  # creating object like robot
    robot.connectEMERGE()
    robot.loadEMERGE()
    peak = robot.obj_ids[1]
    goal = robot.obj_ids[2]
    percept = perceptions()
    info = perception_data()

    max_distance = 0.3  # Units of sim
    unit = math.radians(5)  # degrees to rads
    max_steps = 1000
    steps = 0
    tolerance_to_reach_goal = 0.01  # Units of sim

    utility_model = load_model(
        route[0] + f"/Data/Models/UtilityModelforRobotFromGridScenario.h5"
    )
    world_model = load_model(
        route[0] + f"/Data/Models/EMERGECoord_WorldModelforRobot.h5"
    )

    test_positions_file = route[0] + "/Data/EMERGECoord_TestPositions.txt"
    test_positions = load_test_positions(test_positions_file)

    # info about robot
    print("\nJoints:", robot.joint_ids)
    print(f"Object: {robot.obj_ids}, Peak:{peak}")

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

        # Visualize set goal
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

        while True:
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

    export(info)
    robot.unloadEMERGE()
    robot.disconnectEMERGE()


if __name__ == "__main__":
    main()
