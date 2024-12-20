"""
This program simulates the movement of a two-jointed robotic arm in a 2D space toward a predefined goal position. 
It uses a trained Utility Model (UM) learned from grid scenarion to predict the best movement direction by evaluating utility scores for possible joint angle changes. 
The program logs data, visualizes the movements, and saves the results for further learning or evaluation.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from numpy import cos, sin, sqrt
import matplotlib.pyplot as plt
import math
import time

import sys

# Route setup for file paths
route = sys.path
print(route)
timestr = time.strftime("_%Y_%d%m_%H%M")  # Timestamp for file naming


# Log messages to the terminal and a log file
def log_message(message, log_file):
    print(message)  # Print the message to the terminal
    with open(log_file, "a") as file:
        file.write(message + "\n")  # Append the message to the log file


# Calculate Euclidean distance between two points
def compute_distance(goal_x, goal_y, pred_x, pred_y):
    return sqrt((goal_x - pred_x) ** 2 + (goal_y - pred_y) ** 2)


# Calculate the unit vector from the current position to the goal
def compute_unit_vector(goal_x, goal_y, pred_x, pred_y):
    dx = goal_x - pred_x
    dy = goal_y - pred_y
    magnitude = sqrt(dx**2 + dy**2)
    if magnitude == 0:
        return 0.0, 0.0  # Avoid division by zero
    else:
        return dx / magnitude, dy / magnitude


# Calculate the new position of the arm's endpoint based on joint angles
def calculate_position_from_angles(alpha_rad, beta_rad, L1, L2):
    x = L1 * cos(alpha_rad) + L2 * cos(beta_rad)
    y = L1 * sin(alpha_rad) + L2 * sin(beta_rad)
    return x, y


# Function to predict the best movement direction based on angle changes
def predict_next_move(robot_angles, model, goal_pos, L1, L2, log_file, unit):
    # Possible changes in angles (alpha, beta)
    move_options = [(unit, 0), (-unit, 0), (0, unit), (0, -unit)]
    direction_names = [
        f"+{unit} rad on alpha",
        f"-{unit} rad on alpha",
        f"+{unit} rad on beta",
        f"-{unit} rad on beta",
    ]

    best_move = None
    best_utility = -np.inf
    all_utilities = {}
    best_x, best_y = None, None  # To store the coordinates of the best move

    for idx, move in enumerate(move_options):
        new_alpha = robot_angles[0] + move[0]
        new_beta = robot_angles[1] + move[1]

        new_x, new_y = calculate_position_from_angles(
            new_alpha, new_beta, L1, L2
        )  # Calculate new position based on updated angles
        distance = compute_distance(
            goal_pos[0], goal_pos[1], new_x, new_y
        )  # Calculate utility value for this new position
        max_distance = calculate_max_distance(L1, L2)
        norm_distance = distance / max_distance  # Normalization of distance
        unit_vec_x, unit_vec_y = compute_unit_vector(
            goal_pos[0], goal_pos[1], new_x, new_y
        )
        pred_utility = model.predict(
            np.array([[norm_distance, unit_vec_x, unit_vec_y]]), verbose=0
        )[0][0]

        # Update best move if the utility is higher
        all_utilities[direction_names[idx]] = pred_utility
        if pred_utility > best_utility:
            best_utility = pred_utility
            best_move = move
            best_direction = direction_names[idx]
            best_x, best_y = new_x, new_y  # Save the best coordinates

    # Log the chosen move and utility values
    message = (
        f"Current Angles: {robot_angles[0]:.3f}, {robot_angles[1]:.3f}; "
        f"Goal: {goal_pos[0]:.3f}, {goal_pos[1]:.3f};"
        f"+{unit} rad on alpha: {all_utilities[f'+{unit} rad on alpha']:.3f}, "
        f"-{unit} rad on alpha: {all_utilities[f'-{unit} rad on alpha']:.3f}, "
        f"+{unit} rad on beta: {all_utilities[f'+{unit} rad on beta']:.3f}, "
        f"-{unit} rad on beta: {all_utilities[f'-{unit} rad on beta']:.3f}; "
        f"Robot choose: {best_direction} with Utility Value: {best_utility:.3f}; "
        f"Distance: {distance:.3f}; Predicted Coordinates: (x={best_x:.3f}, y={best_y:.3f})"
    )

    log_message(message, log_file)

    return best_move


# Visulize graph of move with steps visualization
def save_robot_movement_graph(moves, L1, L2, goal_position, save_path, start_angles):
    plt.figure()
    plt.title("Movement of Arm")

    # Define plot limits
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    num_moves = len(moves)

    # Draw moves
    for i, move in enumerate(moves):
        # Plot every tenth step to the last 10 steps
        if i < num_moves - 5 and i % 20 != 0:
            continue

        alpha, beta = move

        x1 = L1 * np.cos(alpha)
        y1 = L1 * np.sin(alpha)
        x2, y2 = calculate_position_from_angles(alpha, beta, L1, L2)

        # Draw robot's segments
        plt.plot([0, x1], [0, y1], "k-", zorder=1)
        plt.plot([x1, x2], [y1, y2], "-", color="gray", zorder=1)

        # Draw joints as points
        # plt.scatter([x1], [y1], color="green", label=f"1.Joint {i+1}" if i == 0 else "")
        plt.scatter(
            [x2],
            [y2],
            color="black",
            s=6,
            zorder=2,
        )

    # Plot goal position
    plt.scatter(
        goal_position[0], goal_position[1], color="red", label="Goal", zorder=10
    )

    # Plot start position
    alpha_start, beta_start = start_angles
    x1_start = L1 * np.cos(alpha_start)
    y1_start = L1 * np.sin(alpha_start)
    x2_start, y2_start = calculate_position_from_angles(alpha_start, beta_start, L1, L2)
    plt.plot([0, x1_start], [0, y1_start], "b-", zorder=8)
    plt.plot(
        [x1_start, x2_start], [y1_start, y2_start], "-", color="lightblue", zorder=8
    )
    plt.scatter(x2_start, y2_start, color="blue", label="Start", zorder=9)

    # Axis descriptions and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()

    # Save the graph to the specified path
    plt.savefig(save_path)
    plt.close()


# Load test positions from a file
def load_test_positions(file_path):
    test_positions = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                start_angles, goal_coords = line.split(";")
                start_alpha, start_beta = map(float, start_angles.split(","))
                goal_x, goal_y = map(float, goal_coords.split(","))
                test_positions.append(([start_alpha, start_beta], [goal_x, goal_y]))
            except ValueError:
                print(f"Error processing line: {line}")
                continue
    return test_positions


# Save simulation data to a text file with correction for the final step and limit to 10 steps
def save_learning_data(moves, file_path, goal_x, goal_y):
    # Check if the file exists, if not, create it
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

            file.write(
                f"{distance:.3f};({unit_vec_x:.3f},{unit_vec_y:.3f});{score:.1f}\n"
            )
        file.write("\n")


# Run a simulation to move the robot arm toward the goal
def run_simulation(
    model,
    start_angles,
    robot_angles,
    goal_position,
    L1,
    L2,
    learning_file_path,
    log_file,
    tolerance_to_goal,
    graph_save_path,
    unit=1,
    max_steps=1000,
):
    steps = 0
    moves = []  # Save each action

    while True:
        # Predict the best movement
        best_move = predict_next_move(
            robot_angles, model, goal_position, L1, L2, log_file, unit
        )

        # Update joint angles based on the best move
        robot_angles[0] += best_move[0]
        robot_angles[1] += best_move[1]

        # Calculate the new position of the arm's endpoint
        new_x, new_y = calculate_position_from_angles(
            robot_angles[0], robot_angles[1], L1, L2
        )
        moves.append((robot_angles[0], robot_angles[1]))

        # Check if the goal is reached
        distance_to_goal = compute_distance(
            goal_position[0], goal_position[1], new_x, new_y
        )
        if distance_to_goal < tolerance_to_goal:
            log_message(f"Robot reached the goal in {steps} steps!", log_file)
            save_learning_data(
                moves, learning_file_path, goal_position[0], goal_position[1]
            )
            save_robot_movement_graph(
                moves, L1, L2, goal_position, graph_save_path, start_angles
            )
            return True

        steps += 1
        if steps > max_steps:
            log_message(f"Test failed after {max_steps} steps!", log_file)
            return False


# Calculate the maximum distance for normalization
def calculate_max_distance(L1, L2):
    max_distance = sqrt((L1) ** 2 + (L2) ** 2)
    print(f"MaxDistance:{max_distance:.3f}")
    return max_distance


def main():
    log_file_path = route[0] + f"/Data/BarRecords_{timestr}.txt"

    # Delete existing log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # Load the utility model
    model = load_model(route[0] + "/Models/UtilityModelforRobotFromGridScenario.h5")

    # Define arm segment lengths
    L1, L2 = 1, 1

    # Load test positions for simulation
    test_positions_file = route[0] + "/Data/BarTestPositions.txt"
    test_positions = load_test_positions(test_positions_file)

    # Set parameters for the simulation
    save_selflearning_file_path = (
        route[0] + f"/Data/BarDataForSelfLearning_{timestr}.txt"
    )
    unit = math.radians(1)  # Angle step in radians
    tolerance_to_goal = 0.1  # Tolerance for reaching the goal
    max_steps = 1000  # Maximum steps allowed

    # Run simulations for each test position
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
            save_selflearning_file_path,
            log_file_path,
            tolerance_to_goal,
            graph_save_path,
            unit=unit,
            max_steps=max_steps,
        )


if __name__ == "__main__":
    main()
