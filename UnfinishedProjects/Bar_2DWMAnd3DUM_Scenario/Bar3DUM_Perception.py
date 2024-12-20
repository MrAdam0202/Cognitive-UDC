"""
Topic: Using 3D Utility Model in 2D Space

This program simulates the movement of a two-jointed robot arm in a 2D space. 
The robot's task is to reach a given goal position starting from specified initial angles for its joints. 
The program uses utility model learned in a 3D space to predict the best movement direction with highest utility value at each step, 
based on normalized distances and unit vectors, and saves simulation data for further analysis.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from numpy import cos, sin, sqrt
import matplotlib.pyplot as plt
import math
import time

import sys

# Get the current working directory and display system paths
route = sys.path
print(route)

# Generate a timestamp for file saving
timestr = time.strftime("%Y_%d%m_%H%M")


# Saves a message to both the terminal and a specified log file
def log_message(message, log_file):
    print(message)
    with open(log_file, "a") as file:
        file.write(message + "\n")


# Calculate the distance from the predicted position to the goal
def compute_distance(goal_x, goal_y, pred_x, pred_y):
    return sqrt((goal_x - pred_x) ** 2 + (goal_y - pred_y) ** 2)


# Normalizes a distance value to obtain required input for utility model
def normalize_distance(distance, max_distance):
    return distance / max_distance


# Calculate unit vector
def compute_unit_vector(goal_x, goal_y, pred_x, pred_y):
    dx = goal_x - pred_x
    dy = goal_y - pred_y
    magnitude = sqrt(dx**2 + dy**2)
    if magnitude == 0:
        return [0.0, 0.0]
    else:
        return [dx / magnitude, dy / magnitude]


# Convert 2D unit vector to 3D unit vector
def unit_vector_2D_to_3D(UV_2D):
    UV_3D = [0.0, 0.0, 0.2]
    UV_3D[0] = UV_2D[0]
    UV_3D[1] = UV_2D[1]
    return UV_3D


# Calculate new position based on alpha and beta
def calculate_position_from_angles(alpha_rad, beta_rad, L1, L2):
    x = L1 * cos(alpha_rad) + L2 * cos(beta_rad)
    y = L1 * sin(alpha_rad) + L2 * sin(beta_rad)
    return x, y


# Use a utility model to predict utility value of a potential movement
def predict_utility(next_norm_distance, next_unit_vector, utility_model):
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
    utility_value = utility_model.predict(input_data, verbose=0)[0][0]
    return utility_value


# Determinate the best joint movements to minimize distance to the goal
def predict_next_move(robot_angles, UM, goal_pos, L1, L2, max_distance, log_file, unit):
    # Possible actions in angles (alpha, beta)
    move_options = [(unit, 0), (-unit, 0), (0, unit), (0, -unit)]
    direction_names = [
        f"+{math.degrees(unit)}° on alpha",
        f"-{math.degrees(unit)}° on alpha",
        f"+{math.degrees(unit)}° on beta",
        f"-{math.degrees(unit)}° on beta",
    ]

    best_move = None
    best_utility = -np.inf
    all_utilities = {}
    best_x, best_y = None, None

    for idx, move in enumerate(move_options):
        new_alpha = robot_angles[0] + move[0]
        new_beta = robot_angles[1] + move[1]

        # Calculate new position input data based on updated angles
        next_x, next_y = calculate_position_from_angles(new_alpha, new_beta, L1, L2)
        next_distance = compute_distance(goal_pos[0], goal_pos[1], next_x, next_y)
        next_norm_distance = normalize_distance(next_distance, max_distance)
        next_unit_vec_2D = compute_unit_vector(goal_pos[0], goal_pos[1], next_x, next_y)
        next_unit_vec_3D = unit_vector_2D_to_3D(next_unit_vec_2D)

        # Calculate utility value for this new position
        pred_utility = predict_utility(next_norm_distance, next_unit_vec_3D, UM)

        # Save all specified utility values
        all_utilities[direction_names[idx]] = pred_utility

        # Determinate the best joint movement based on best utility value
        if pred_utility > best_utility:
            best_utility = pred_utility
            best_move = move
            best_direction = direction_names[idx]
            best_x, best_y = next_x, next_y  # Save the best coordinates
            best_unit_vec = next_unit_vec_3D
            best_distance = next_distance
            best_norm_distance = next_norm_distance

    # Log information about the current state and best move
    message = (
        f"Current Angles: {robot_angles[0]:.3f}, {robot_angles[1]:.3f};"
        f"Goal: {goal_pos[0]:.3f}, {goal_pos[1]:.3f};"
        f"+{math.degrees(unit)}° on alpha: {all_utilities[f'+{math.degrees(unit)}° on alpha']:.3f}, "
        f"-{math.degrees(unit)}° on alpha: {all_utilities[f'-{math.degrees(unit)}° on alpha']:.3f}, "
        f"+{math.degrees(unit)}° on beta: {all_utilities[f'+{math.degrees(unit)}° on beta']:.3f}, "
        f"-{math.degrees(unit)}° on beta: {all_utilities[f'-{math.degrees(unit)}° on beta']:.3f}; "
        f"\n\nRobot choose: {best_direction} with Utility Value: {best_utility:.3f}, Predicted Coordinates: (x={best_x:.3f}, y={best_y:.3f}); "
        f"\n-----------------------------------------------------------------\n"
        f"Next distance: {best_distance:.3f} (NormDis: {best_norm_distance:.3f}), Unit Vector: {[round(pos,3) for pos in best_unit_vec]}"
        f"\n-----------------------------------------------------------------\n"
    )
    log_message(message, log_file)

    return best_move


# Visualize graph of move with steps visualization
def save_robot_movement_graph(moves, L1, L2, goal_position, save_path, start_angles):
    plt.figure()
    plt.title("Movement of Arm")
    scale = 1.5
    # Set the graph limits to the first quadrant
    plt.xlim(-scale, scale)
    plt.ylim(-scale, scale)

    num_moves = len(moves)

    for i, move in enumerate(moves):
        # Plot every 10th step and the last 5 steps for clarity
        if i < num_moves - 5 and i % 20 != 0:
            continue

        alpha, beta = move
        x1 = L1 * np.cos(alpha)
        y1 = L1 * np.sin(alpha)
        x2, y2 = calculate_position_from_angles(alpha, beta, L1, L2)

        # Draw the arm's segments
        plt.plot([0, x1], [0, y1], "k-", zorder=1)
        plt.plot([x1, x2], [y1, y2], "-", color="gray", zorder=1)

        # Plot end-effector positions
        plt.scatter(
            [x2],
            [y2],
            color="black",
            s=6,
            zorder=2,
        )

    # Highlight the goal position in red
    plt.scatter(
        goal_position[0], goal_position[1], color="red", label="Goal", zorder=10
    )
    # Highlight the start position in blue
    alpha_start, beta_start = start_angles
    x1_start = L1 * np.cos(alpha_start)
    y1_start = L1 * np.sin(alpha_start)
    x2_start, y2_start = calculate_position_from_angles(alpha_start, beta_start, L1, L2)
    plt.plot([0, x1_start], [0, y1_start], "b-", zorder=8)
    plt.plot(
        [x1_start, x2_start], [y1_start, y2_start], "-", color="lightblue", zorder=8
    )
    plt.scatter(x2_start, y2_start, color="blue", label="Start", zorder=9)

    # Add labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()

    # Save the graph to the specified path
    plt.savefig(save_path)
    plt.close()


# Load test positions (start angles and goal positions) from a text file
def load_test_positions(file_path):
    test_positions = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                # Parse the line into start angles and goal coordinates
                start_angles, goal_coords = line.split(";")
                start_alpha, start_beta = map(float, start_angles.split(","))
                goal_x, goal_y = map(float, goal_coords.split(","))
                test_positions.append(([start_alpha, start_beta], [goal_x, goal_y]))
            except ValueError:
                print(f"Error processing line: {line}")
                continue
    return test_positions


# Save the robot's movements and corresponding scores to a text file for self learning
def save_learning_data(moves, file_path, goal_x, goal_y, L1, L2):

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Creating new file.")
        open(file_path, "w").close()

    with open(file_path, "a") as file:
        file.write(f"Test\n")
        last_steps = min(len(moves), 10)  # Limit to the last 10 steps
        for i in range(last_steps):
            if i == 0:  # Final step where the robot reached the goal
                distance = 0.0
                unit_vec_x, unit_vec_y = 0.0, 0.0
                score = 1.0
            else:
                alpha, beta = moves[-(i + 1)]  # Save steps in reverse order
                x, y = calculate_position_from_angles(alpha, beta, L1, L2)
                distance = compute_distance(goal_x, goal_y, x, y)
                unit_vec_x, unit_vec_y = compute_unit_vector(goal_x, goal_y, x, y)
                score = max(0, 1 - 0.1 * i)  # Avoid negative scores

            # Save the data to the file
            file.write(
                f"{distance:.3f};({unit_vec_x:.3f},{unit_vec_y:.3f});{score:.1f}\n"
            )
        file.write("\n")


# Simulate the robot's motion to reach the target position by iteratively predicting the best motion
def run_simulation(
    model,
    start_angles,
    robot_angles,
    goal_position,
    L1,
    L2,
    max_distance,
    learning_file_path,
    log_file,
    tolerance_to_goal,
    graph_save_path,
    unit=1,
    max_steps=1000,
):
    steps = 0
    moves = []  # List to store the robot's actions
    while True:
        # Predict the best move
        best_move = predict_next_move(
            robot_angles, model, goal_position, L1, L2, max_distance, log_file, unit
        )
        # Update robot angles based on the best move
        robot_angles[0] += best_move[0]
        robot_angles[1] += best_move[1]

        # Calculate new position
        new_x, new_y = calculate_position_from_angles(
            robot_angles[0], robot_angles[1], L1, L2
        )
        moves.append((robot_angles[0], robot_angles[1]))

        # Check if the robot has reached the goal
        distance_to_goal = compute_distance(
            goal_position[0], goal_position[1], new_x, new_y
        )
        if distance_to_goal < tolerance_to_goal:  # Goal reached within set tolerance
            log_message(f"\nRobot reached the goal in {steps} steps!\n", log_file)
            save_learning_data(
                moves, learning_file_path, goal_position[0], goal_position[1], L1, L2
            )
            save_robot_movement_graph(
                moves, L1, L2, goal_position, graph_save_path, start_angles
            )
            return True

        steps += 1
        if steps > max_steps:  # Max steps exceeded
            log_message(f"\nTest failed after {max_steps} steps!\n", log_file)
            return False


# Calculate the maximum possible distance for the robot's arm in 2D space
def calculate_max_distance(L1, L2):
    max_distance = sqrt(2) * (L1 + L2)
    print(f"MaxDistance:{max_distance:.3f}")
    return max_distance


def main():
    # Define the log file path
    log_file_path = route[0] + f"/Data/BarRecords_{timestr}.txt"
    if os.path.exists(log_file_path):
        os.remove(log_file_path)  # Remove the log file if it already exists

    # Load the pre-trained utility model for prediction of utility values
    model = load_model(
        route[0]
        + f"/Data/Models/SphericalMotions_UtilityModelfor3DRobot_2024_0912_1601.h5"
    )

    # Define the lengths of the robot arm segments
    L1, L2 = 0.5, 0.5

    # Load test positions (start angles and goal positions) from a file
    test_positions_file = route[0] + "/Data/BarTestPositions.txt"
    test_positions = load_test_positions(test_positions_file)

    # Define the file path for saving self-learning data to improve the utility model
    save_selflearning_file_path = (
        route[0] + f"/Data/BarDataForSelfLearning_{timestr}.txt"
    )

    # Calculate the maximum distance for normalization
    max_distance = 5  # Adjusted manually to approximate the similarity of the input data to the utility model
    # max_distance = calculate_max_distance(L1, L2)

    # Define parameters for the simulation
    unit = math.radians(5)  # Unit step in radians for changing angles
    tolerance_to_goal = 0.1  # Tolerance for considering the goal as reached
    max_steps = 50  # Maximum number of steps to reach the goal allowed per test

    # Run the simulation
    for idx, (start_angles, goal_position) in enumerate(test_positions):
        graph_save_path = route[0] + f"/Data/BarMoveGraph_test_{timestr}_{idx + 1}.png"

        robot_angles = start_angles.copy()
        run_simulation(
            model,
            start_angles,
            robot_angles,
            goal_position,
            L1,
            L2,
            max_distance,
            save_selflearning_file_path,
            log_file_path,
            tolerance_to_goal,
            graph_save_path,
            unit=unit,
            max_steps=max_steps,
        )


if __name__ == "__main__":
    main()
