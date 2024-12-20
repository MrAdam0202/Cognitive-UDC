"""
This program simulates the generation of training data for a robotic system using a simulated environment. 
It records the robot's current positions, the changes applied to its joints, and the resulting positions to build a dataset. 
The dataset can be used for training predictive models for robotic movement. The program supports simulation mode only.
"""

import time, copy, os, sys
from emerge_joint_handler import *
from sim_joint_handler import *
from joint_handler import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import pandas as pd
from random import randint as ri
from random import uniform as ru

# Get the system path for file handling
route = sys.path
print(route)

timestr = time.strftime("_%Y_%d%m_%H%M")

# Enable simulation mode
is_sim = True
client = RemoteAPIClient("localhost", 23000)  # Connect to the CoppeliaSim API
sim = client.getObject("sim")  # Create a simulation object


# Determine the type of connection (simulation or physical mode)
def type_connection(SimActivated):
    if SimActivated:
        print("---------Simulation mode is activated---------\n")
        handler = JointHandler(is_sim, client, sim)  # Initialize simulation handler
    else:
        print("------------Physical mode is activated-----------\n")
        print(
            "------------World model training is available only in simulation mode-----------\n"
        )
        print("------------Program is ended-----------\n")
        exit()
    return handler


# Generate random changes
def rand_gen(obj, data):
    for num in range(obj.num_joints):
        n = ru(-90, 90) * math.pi / 180
        data.changes.append(n)


# Export the collected training data to a CSV file
def export(perception):
    path = route[0] + f"/Data/EMERGECoord_DataForWM_{timestr}.csv"
    df = pd.DataFrame(perception.training_data)
    df.to_csv(
        path,
        index=False,
    )
    print("File of data created")


# A class to store data for training the robot's world model
class training_data:
    def __init__(self):
        self.index = []  # Step index
        self.act_pos = []  # Current position of the robot
        self.next_pos = []  # Position after applying changes
        self.changes = []  # Random joint angle changes


# A class to store perceptions and training-related information
class perceptions:
    def __init__(self):
        # Stores all the training data to be saved
        self.training_data = []
        # Stores initial perceptions measured in the load function
        self.initial_j_positions = []
        self.initial_pos_x = 0
        self.initial_pos_y = 0
        self.initial_pos_z = 0
        # Execution data
        self.sequences = 0
        self.steps = 0
        self.goal_positions_deg = []
        self.goal_positions_rad = []
        self.random_changes = []
        self.start_positions = []
        # Initial perceptions
        self.prev_j_positions = []
        self.prev_pos_x = 0
        self.prev_pos_y = 0
        self.prev_pos_z = 0
        # Resulting perceptions
        self.post_j_positions = []
        self.post_pos_x = 0
        self.post_pos_y = 0
        self.post_pos_z = 0


# Set the robot's joints to their home (neutral) positions
def set_home(object):
    home_pos = 0
    print("\nRobot is going to home positions")
    for joint_n in range(object.num_joints):
        joint = object.joint_ids[joint_n]
        object.setJointTargetPosition(joint, home_pos)
        # print(f"Joint {joint} is set on the start position {next_pos:.2f} rad.")
    print("Robot is in home positions\n")


def main():
    print("Current working directory:", os.getcwd())
    robot = type_connection(is_sim)  # Initialize the robot handler
    robot.connectEMERGE()  # Connect to the robot system
    robot.loadEMERGE()  # Load the robot's simulation
    percept = perceptions()  # Initialize perceptions
    data = training_data()  # Initialize training data
    peak = robot.obj_ids[1]  # Get the peak object ID
    percept.sequences = 2  # Number of sequences
    percept.steps = 5  # Steps per sequence

    # Log robot and simulation info
    print("\nJoints:", robot.joint_ids)
    print(f"Object: {robot.obj_ids}, Peak: {peak}")
    print(f"Sequences: {percept.sequences}, Steps: {percept.steps}")

    print("---------Movement is started---------\n")

    # Simulate sequences and steps
    for seq_n in range(percept.sequences):
        set_home(robot)  # Set robot to home position
        for step_n in range(percept.steps):
            data.index = step_n  # Store step index
            data.changes = []  # Reset changes
            rand_gen(robot, data)  # Generate random changes
            goals_rad = data.changes
            data.act_pos = robot.getObjectPosition(peak)  # Get current position
            print(
                f"Sequence: {seq_n}, Step: {step_n}, Actual Positions: {[round(act, 3) for act in data.act_pos]}, Changes: {[round(change, 3) for change in data.changes]}"
            )

            for joint_n in range(robot.num_joints):
                next_pos = goals_rad[joint_n]
                joint = robot.joint_ids[joint_n]
                robot.setJointTargetPosition(joint, next_pos)  # Apply joint changes
                data.next_pos = robot.getObjectPosition(peak)  # Get new position

            data.index = float(f"{seq_n}.{step_n}")  # Store index for training
            percept.training_data.append(copy.deepcopy(vars(data)))  # Append data

        print(f"New position is: {[round(new, 3) for new in data.next_pos]}")

    print("---------Movement is ended---------\n")

    # Export data to CSV
    export(percept)
    print("---------Data exported---------\n")

    # Unload and disconnect robot
    robot.unloadEMERGE()
    robot.disconnectEMERGE()


if __name__ == "__main__":
    main()
