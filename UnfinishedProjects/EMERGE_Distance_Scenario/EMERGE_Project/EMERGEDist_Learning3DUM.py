"""
This program simulates a robotic movement in 3D spherical space and generates data for training utility models.
It creates sequences of movements starting from random positions and aiming towards random goals, calculating
distances, normalized distances, and unit vectors at each step. The program also assigns utility values (UVs) 
to the last 10 steps of each sequence and saves the data for further analysis. Additionally, it supports 
visualization of trajectories in 3D space.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import pandas as pd
import time, sys

# Get the current working directory and display system paths
route = sys.path
print(route)

# Generate a timestamp for file saving
timestr = time.strftime("%Y_%d%m_%H%M")


# Class to store data for individual steps in a sequence
class LearningData:
    def __init__(
        self,
        step_index,
        previous_position,
        new_position,
        distance,
        unit_vector,
        norm_distance,
        uv=None,
    ):
        self.step_index = step_index
        self.previous_position = previous_position
        self.new_position = new_position
        self.distance = distance
        self.unit_vector = unit_vector
        self.norm_distance = norm_distance
        self.uv = uv  # Utility Value


# Class to manage data across multiple steps in a sequence
class Data:
    def __init__(self):
        self.info = []

    # Add a new step's data to the sequence
    def add_learning_data(self, learning_data):
        self.info.append(learning_data)
        if len(self.info) > 10:
            self.info.pop(0)

    # Assign utility values (UV) to the last 10 steps
    def assign_uv(self):
        utility_values = [round(1.0 - i * 0.1, 1) for i in range(len(self.info))]
        for i, ld in enumerate(self.info):
            ld.uv = utility_values[-(i + 1)]

    # Convert data to a pandas DataFrame
    def to_dataframe(self, sequence_index):
        data = {
            "Step Index": [f"{sequence_index}_{ld.step_index}" for ld in self.info],
            "Previous Position": [ld.previous_position for ld in self.info],
            "New Position": [ld.new_position for ld in self.info],
            "Distance": [ld.distance for ld in self.info],
            "Unit Vector": [ld.unit_vector for ld in self.info],
            "Norm_Distance": [ld.norm_distance for ld in self.info],
            "UV": [ld.uv for ld in self.info],
        }
        return pd.DataFrame(data)


# Generate a random position in 3D space within a spherical boundary
def generate_random_position(length):
    theta = math.radians(random.randint(0, 90))  # Polar angle (0 to π/2)
    phi = math.radians(random.randint(0, 180))  # Azimuthal angle (0 to π)
    x = length * math.sin(theta) * math.cos(phi)
    y = length * math.sin(theta) * math.sin(phi)
    z = length * math.cos(theta)
    return (x, y, z), theta, phi


# Calculate the unit vector from point_a to point_b
def calculate_unit_vector(point_a, point_b):
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    dz = point_b[2] - point_a[2]
    magnitude = math.sqrt(dx**2 + dy**2 + dz**2)
    if magnitude == 0:
        return (0, 0, 0)
    return (dx / magnitude, dy / magnitude, dz / magnitude)


# Calculate the Euclidean distance between two points
def calculate_distance(point_a, point_b):
    return math.sqrt(sum((b - a) ** 2 for a, b in zip(point_a, point_b)))


# Normalize the distance to the range [0, 1] based on the maximum possible distance
def normalize_distance(L, distance):
    max_distance = math.sqrt(2) * L
    return distance / max_distance


# Simulate spherical movement for a sequence and collect data
def simulate_spherical_movement(
    L, max_steps, sequence_index, all_data, all_trajectories
):
    start_position, start_theta, start_phi = generate_random_position(L)
    goal_position, goal_theta, goal_phi = generate_random_position(L)

    print(
        f"[Sequence {sequence_index}] Start Position: {[round(pos, 3) for pos in start_position]}, Goal Position: {[round(pos, 3) for pos in goal_position]}"
    )

    current_position = start_position
    current_theta = start_theta
    current_phi = start_phi
    step_index = 0
    tolerance = 0.1

    data = Data()
    steps = [start_position]

    # Perform movement steps
    while step_index < max_steps:
        distance_to_goal = calculate_distance(current_position, goal_position)
        norm_distance_to_goal = normalize_distance(L, distance_to_goal)
        if distance_to_goal <= tolerance:
            learning_data = LearningData(
                step_index=step_index,
                previous_position=current_position,
                new_position=current_position,
                distance=distance_to_goal,
                unit_vector=calculate_unit_vector(current_position, goal_position),
                norm_distance=norm_distance_to_goal,
            )
            data.add_learning_data(learning_data)
            steps.append(current_position)
            print(
                f"[Sequence {sequence_index}] Step {step_index}, Distance: {distance_to_goal:.3f} - Goal reached!"
            )
            break

        # Determine step size and direction
        step_size = math.radians(1)
        if current_theta < goal_theta:
            new_theta = current_theta + step_size
        elif current_theta > goal_theta:
            new_theta = current_theta - step_size
        else:
            new_theta = current_theta
        if current_phi < goal_phi:
            new_phi = current_phi + step_size
        elif current_phi > goal_phi:
            new_phi = current_phi - step_size
        else:
            new_phi = current_phi

        # Calculate new position
        new_theta = max(0, min(math.pi / 2, new_theta))
        new_phi = max(0, min(math.pi, new_phi))
        new_x = L * math.sin(new_theta) * math.cos(new_phi)
        new_y = L * math.sin(new_theta) * math.sin(new_phi)
        new_z = L * math.cos(new_theta)
        new_position = (new_x, new_y, new_z)

        # Calculate the unit vector and update learning data
        unit_vector = calculate_unit_vector(new_position, goal_position)
        learning_data = LearningData(
            step_index=step_index,
            previous_position=current_position,
            new_position=new_position,
            distance=distance_to_goal,
            unit_vector=unit_vector,
            norm_distance=norm_distance_to_goal,
        )
        data.add_learning_data(learning_data)
        steps.append(new_position)
        print(
            f"[Sequence {sequence_index}] Step {step_index}, Distance: {distance_to_goal:.3f}"
        )

        # Update current state
        current_position = new_position
        current_theta = new_theta
        current_phi = new_phi
        step_index += 1

    if step_index >= max_steps:
        print(
            f"[Sequence {sequence_index}] Error: Goal was not reached in the range of max steps: ({max_steps})."
        )
    else:
        data.assign_uv()
        sequence_df = data.to_dataframe(sequence_index)
        all_data.append(sequence_df)

        all_trajectories.append((sequence_index, steps, start_position, goal_position))


# Plot all trajectories in 3D space
def plot_all_trajectories(all_trajectories, visualize):
    if not visualize:
        print("Visualization is disabled.")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for sequence_index, steps, start_position, goal_position in all_trajectories:
        x = [step[0] for step in steps]
        y = [step[1] for step in steps]
        z = [step[2] for step in steps]

        ax.plot(
            x,
            y,
            z,
            marker="o",
            markersize=2,
            linewidth=0.5,
            label=f"Seq {sequence_index}",
        )
        ax.scatter(*start_position, color="green", marker="x", s=40)  # Start
        ax.scatter(*goal_position, color="red", marker="x", s=40)  # Goal

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize="small", ncol=2, title="Sequences")

    path = route[0] + f"/Data/DataForUM/3DGraph_AllSequences_{timestr}.png"
    plt.savefig(path, dpi=300)
    print(f"Graph is saved.")
    plt.show()


def main():
    L = 10
    sequences = 20000
    max_steps = 200
    path = route[0] + f"/Data/DataForUM/EMERGEDist_SphericalDataForUM_{timestr}.csv"
    visualize = False

    all_data = []
    all_trajectories = []

    for sequence_index in range(1, sequences + 1):
        simulate_spherical_movement(
            L, max_steps, sequence_index, all_data, all_trajectories
        )

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(path, index=False)
        print(f"Data saved.")

    if all_trajectories:
        plot_all_trajectories(all_trajectories, visualize)


if __name__ == "__main__":
    main()
