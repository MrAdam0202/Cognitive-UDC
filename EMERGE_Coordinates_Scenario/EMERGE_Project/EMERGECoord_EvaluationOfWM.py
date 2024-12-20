"""
This program integrates simulation and real-world modes to control a robotic system (EMERGE robot). 
It uses a neural network-based World Model (WM) to predict the robot's next position based on its current position and randomly generated joint angle changes. 
The program evaluates predicted positions against actual robot actions, exports the evaluation data and generates a visual comparison between the prediction and reality.
"""

import time, copy, os, sys, math
from emerge_joint_handler import *
from sim_joint_handler import *
from joint_handler import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import pandas as pd
from random import uniform as ru
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# Get the system path for file handling
route = sys.path
print(route)

# Specify whether the simulation mode is active
is_sim = True

# Connect to CoppeliaSim's Remote API
client = RemoteAPIClient("localhost", 23000)
sim = client.getObject("sim")


# Select the type of connection (Simulation or Physical)
def type_connection(SimActivated):
    if SimActivated:
        print("---------Simulation mode is activated---------\n")
        handler = JointHandler(is_sim, client, sim)  # Simulation handler
    else:
        print("------------Physical mode is activated-----------\n")
        print("------------Move for physical mode is not prepared yet-----------\n")
        print("------------Program is ended-----------\n")
        exit()
    return handler


# Generate random joint angle changes
def rand_gen():
    change = [math.radians(ru(-90, 90)) for _ in range(3)]  # Random angles in radians
    return change


# Predict the next position based on the current position and joint angle changes
def predict_next_positions(current_pos, change, W_model):
    if len(current_pos) != 3 or len(change) != 3:
        raise ValueError(
            "Input `current_pos` and `change` must have exactly 3 elements each."
        )
    input_features = np.array([list(current_pos) + list(change)])  # Combine inputs
    next_x, next_y, next_z = W_model.predict(input_features, verbose=0)[0]  # Predict
    return next_x, next_y, next_z


# Reset the robot to its home position
def set_home(robot):
    home_pos = 0  # Home position angle
    print(f"\n -----Robot is going to home positions-----")
    for joint_n in range(robot.num_joints):
        joint = robot.joint_ids[joint_n]
        robot.setJointTargetPosition(joint, home_pos)
    print(f"-----Robot is in home positions-----\n")


# Class to store all perception data during the simulation
class perception_data:
    def __init__(self):
        self.info_data = []


# Class to store individual perception details
class perceptions:
    def __init__(self):
        self.index = 0
        self.current_pos = []
        self.change_to_next = []
        self.real_next_pos = []
        self.pre_next_pos = []


# Export simulation data to a CSV file
def export(data):
    timestr = time.strftime("_%Y_%d%m_%H%M")
    path = route[0] + f"/Data/EMERGECoord_EvaluationOfWM_{timestr}.csv"
    df = pd.DataFrame(data.info_data)
    df.to_csv(
        path,
        index=False,
    )
    print(f"[{timestr}] File of data created")


# Plot comparison graphs for real and predicted positions
def plot_comparison(info):
    timestr = time.strftime("_%Y_%d%m_%H%M")
    real_positions = [data["real_next_pos"] for data in info.info_data]
    predicted_positions = [data["pre_next_pos"] for data in info.info_data]

    # Extract x, y, z coordinates for real and predicted positions
    real_x = [pos[0] for pos in real_positions]
    real_y = [pos[1] for pos in real_positions]
    real_z = [pos[2] for pos in real_positions]

    pred_x = [pos[0] for pos in predicted_positions]
    pred_y = [pos[1] for pos in predicted_positions]
    pred_z = [pos[2] for pos in predicted_positions]

    steps = range(1, len(real_positions) + 1)  # Test step indices

    # Plot x, y, and z comparisons
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(steps, real_x, label="Real x", color="blue")
    plt.plot(steps, pred_x, label="Predicted x", color="orange")
    plt.title("Comparison of Real and Predicted Positions")
    plt.ylabel("x-coordinate")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(steps, real_y, label="Real y", color="blue")
    plt.plot(steps, pred_y, label="Predicted y", color="orange")
    plt.ylabel("y-coordinate")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(steps, real_z, label="Real z", color="blue")
    plt.plot(steps, pred_z, label="Predicted z", color="orange")
    plt.xlabel("Test Steps")
    plt.ylabel("z-coordinate")
    plt.legend()

    plt.tight_layout()
    plt.savefig(route[0] + f"/Data/EMERGECoord_GraphEvaluationOfWM_{timestr}.png")
    print(f"Graph saved: EMERGECoord_GraphEvaluationOfWM_{timestr}.png")
    plt.close()


# Main function to execute the program logic
def main():
    timestr = time.strftime("_%Y_%d%m_%H%M")
    print(f" Current working directory: {os.getcwd()}")

    # Connect to the robot and initialize
    robot = type_connection(is_sim)
    robot.connectEMERGE()
    robot.loadEMERGE()

    peak = robot.obj_ids[1]  # Get peak object
    percept = perceptions()  # Initialize perception class
    info = perception_data()  # Initialize perception data class

    sequences = 2  # Number of test sequences
    steps = 5  # Number of steps per sequence

    # Load the pre-trained World Model
    world_model = load_model(
        route[0] + "/Data/Models/EMERGECoord_WorldModelforRobot.h5"
    )

    print(f" Joints: {robot.joint_ids}")
    print(f" Object: {robot.obj_ids}, Peak: {peak}")

    print(f" ---------Movement is started---------\n")

    for seq_n in range(sequences):
        # Reset robot to home position at the start of each sequence
        set_home(robot)

        for step_n in range(steps):
            index = float(f"{seq_n}.{step_n}")  # Step index
            cur_x, cur_y, cur_z = robot.getObjectPosition(peak)  # Get current position
            current_pos = [cur_x, cur_y, cur_z]
            change = rand_gen()  # Generate random angle changes

            # Predict next position using the World Model
            pre_next_x, pre_next_y, pre_next_z = predict_next_positions(
                current_pos, change, world_model
            )

            pre_next_pos = [pre_next_x, pre_next_y, pre_next_z]
            percept.pre_next_pos = pre_next_pos

            # Apply angle changes to the robot
            for joint_n in range(robot.num_joints):
                next_ang = change[joint_n]
                joint = robot.joint_ids[joint_n]
                robot.setJointTargetPosition(joint, next_ang)

            # Get the real next position
            next_x, next_y, next_z = robot.getObjectPosition(peak)
            next_pos = [next_x, next_y, next_z]
            percept.index = index
            percept.change_to_next = change
            percept.current_pos = current_pos
            percept.real_next_pos = next_pos
            info.info_data.append(copy.deepcopy(vars(percept)))  # Save perception data

            # Log step information
            print(
                f"{index:.1f} Robot was in position: {[round(pos, 3) for pos in current_pos]}, "
                f"Change: {[round(pos, 3) for pos in change]},\n"
                f"Robot reached position: {[round(pos, 3) for pos in next_pos]}, "
                f"Predicted position: {[round(pos, 3) for pos in pre_next_pos]}"
            )
            current_pos = next_pos.copy()

    print(f" ---------Movement is ended---------\n")

    # Export data and generate comparison plots
    export(info)
    print(f" ---------Data exported---------\n")
    plot_comparison(info)

    # Disconnect the robot
    robot.unloadEMERGE()
    robot.disconnectEMERGE()


# Run the main function
if __name__ == "__main__":
    main()
