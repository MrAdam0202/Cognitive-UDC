"""
This program simulates a robot navigating on a 2D grid towards a predefined goal position. 
It utilizes a trained utility model to predict the most optimal movement direction based on the robot's current position an the goal position.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from numpy import sqrt
import sys

route = sys.path
# print(route)


# Function to log messages to the terminal and a text file
def log_message(message, log_file):
    print(message)  # Print to terminal
    with open(log_file, "a") as file:
        file.write(message + "\n")  # Append to the log file


# Calculate the distance from the predicted position to the goal
def compute_distance(goal_x, goal_y, pred_x, pred_y):
    return sqrt((goal_x - pred_x) ** 2 + (goal_y - pred_y) ** 2)


# Calculate unit vector
def compute_unit_vector(goal_x, goal_y, pred_x, pred_y):
    dx = goal_x - pred_x
    dy = goal_y - pred_y
    magnitude = sqrt(dx**2 + dy**2)
    if magnitude == 0:
        return 0.0, 0.0
    else:
        return dx / magnitude, dy / magnitude


# Function to predict the best movement direction
def predict_next_move(robot_pos, model, goal_pos, log_file, max_distance):
    move_options = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    direction_names = ["+Y", "-Y", "+X", "-X"]
    best_move = None
    best_utility = -np.inf
    all_utilities = {}

    # Calculate the distance between the robot and the goal
    distance_to_goal = compute_distance(
        goal_pos[0], goal_pos[1], robot_pos[0], robot_pos[1]
    )

    for idx, move in enumerate(move_options):
        # Compute the new position after applying the move
        new_pos = [robot_pos[0] + move[0], robot_pos[1] + move[1]]

        # Calculate distance and normalized distance for the new position
        distance = compute_distance(goal_pos[0], goal_pos[1], new_pos[0], new_pos[1])
        norm_distance = distance / max_distance  # Normalization of distance

        # Calculate unit vector for the new position
        unit_vec_x, unit_vec_y = compute_unit_vector(
            goal_pos[0], goal_pos[1], new_pos[0], new_pos[1]
        )

        # Perform prediction using normalized distance
        pred_utility = model.predict(
            np.array([[norm_distance, unit_vec_x, unit_vec_y]]), verbose=0
        )[0][0]

        # Store the predicted utility value for the current direction
        all_utilities[direction_names[idx]] = pred_utility

        # Check if this move has the highest utility so far
        if pred_utility > best_utility:
            best_utility = pred_utility
            best_move = move
            best_direction = direction_names[idx]

    # Create a log message including the normalized distance
    message = (
        f"Robot Position: {robot_pos}; Goal: {goal_pos}; Distance: {distance_to_goal:.3f}; "
        f"+X: {all_utilities['+X']:.3f}, -X: {all_utilities['-X']:.3f}, "
        f"+Y: {all_utilities['+Y']:.3f}, -Y: {all_utilities['-Y']:.3f}; "
        f"Robot chose: {best_direction} with Utility Value: {best_utility:.3f}"
    )

    # Log the message
    log_message(message, log_file)

    return best_move


# Save simulation data to a text file with correction for the final step and limit to 10 steps
def save_learning_data(moves, file_path, goal_x, goal_y):
    # Check if the file exists, if not, create it
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Creating new file.")
        open(file_path, "w").close()

    with open(file_path, "a") as file:
        file.write(f"Test\n")
        steps = min(len(moves), 10)  # Limit to the last 10 steps
        for i in range(steps):
            if i == 0:  # Final step where the robot reached the goal
                distance = 0.0
                unit_vec_x, unit_vec_y = 0.0, 0.0
                score = 1.0
            else:
                x, y = moves[-(i + 1)]  # Save steps in reverse order
                distance = compute_distance(goal_x, goal_y, x, y)
                unit_vec_x, unit_vec_y = compute_unit_vector(goal_x, goal_y, x, y)
                score = max(0, 1 - 0.1 * i)  # Avoid negative scores

            file.write(
                f"{distance:.3f};({unit_vec_x:.3f},{unit_vec_y:.3f});{score:.1f}\n"
            )
        file.write("\n")


# Load test positions from a file
def load_test_positions(file_path):
    test_positions = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                robot_coords, goal_coords = line.split(";")
                robot_x, robot_y = map(int, robot_coords.split(","))
                goal_x, goal_y = map(int, goal_coords.split(","))
                test_positions.append(([robot_x, robot_y], [goal_x, goal_y]))
            except ValueError:
                print(f"Error processing line: {line}")
                continue
    return test_positions


# Simulate a robot's movement to the goal
def run_simulation(
    model, robot_start, goal_position, file_path, log_file, max_distance
):
    robot_x, robot_y = robot_start
    goal_x, goal_y = goal_position
    moves = []  # Store moves
    steps = 0
    while (robot_x, robot_y) != (goal_x, goal_y):
        best_move = predict_next_move(
            [robot_x, robot_y], model, [goal_x, goal_y], log_file, max_distance
        )
        robot_x += best_move[0]
        robot_y += best_move[1]
        moves.append((robot_x, robot_y))  # Save the move
        steps += 1
        if steps > 500:
            log_message(f"Test failed after 500 steps!", log_file)
            return False
    log_message(f"Robot reached the goal in {steps} steps!", log_file)
    save_learning_data(moves, file_path, goal_x, goal_y)  # Save data to file
    return True


# Run multiple simulations
def run_multiple_simulations(model, test_positions, file_path, log_file, max_distance):
    for i, (robot_start, goal_position) in enumerate(test_positions):
        log_message(f"Test {i + 1}:", log_file)
        success = run_simulation(
            model, robot_start, goal_position, file_path, log_file, max_distance
        )
        if not success:
            log_message(f"Test {i + 1} failed after 50 steps!\n", log_file)


# Function to load distances from UtilityValuesNoNorm.txt and get the max distance
def load_max_distance_from_utility(file_path):
    distances = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("Test"):
                continue
            try:
                distance = float(line.split(";")[0])
                distances.append(distance)
            except ValueError:
                print(f"Error processing line: {line}")
                continue
    return max(distances) if distances else None


# Function to normalize distances in TestPositions.txt
def normalize_test_positions(test_file_path, max_distance, output_file_path):
    if max_distance is None:
        print("Error: Max distance not found.")
        return
    normalized_positions = []
    with open(test_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                robot_coords, goal_coords = line.split(";")
                robot_x, robot_y = map(int, robot_coords.split(","))
                goal_x, goal_y = map(int, goal_coords.split(","))

                # Calculate the distance between robot and goal
                distance = sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)
                # Normalize the distance
                normalized_distance = distance / max_distance
                normalized_positions.append(
                    f"{robot_x},{robot_y};{goal_x},{goal_y};{normalized_distance:.3f}\n"
                )
            except ValueError:
                print(f"Error processing line: {line}")
                continue

    # Save normalized positions to output file
    with open(output_file_path, "w") as out_file:
        out_file.writelines(normalized_positions)
    print(f"Normalized positions saved to {output_file_path}")


def main():
    # Log file path
    log_file_path = route[0] + "/Data/GridRecords.txt"

    # Check if log file exists, and remove it if it does
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # Load the trained model
    model = load_model(
        "/home/adam/Go1Project/SelfLearningApps/AdvancedGridAndBarsScenario/Grid/Models/UtilityModelforRobotFromGridScenario.h5"
    )

    # Load test coordinates from the text file
    test_positions_file = route[0] + "/Data/TestPositions.txt"
    test_positions = load_test_positions(test_positions_file)

    # Additional part for normalization:
    utility_file_path = route[0] + "/Data/UtilityValuesNoNorm.txt"
    max_distance = load_max_distance_from_utility(utility_file_path)
    output_file_path = route[0] + "/Data/NormalizedTestPositions.txt"

    # Normalize the distances in TestPositions.txt and save to new file
    normalize_test_positions(test_positions_file, max_distance, output_file_path)

    # File for saving new data
    learning_file_path = route[0] + "/Data/GridUtilityValuesForSelfLearning.txt"

    print(max_distance)

    # Run simulations with the normalized test positions
    run_multiple_simulations(
        model, test_positions, learning_file_path, log_file_path, max_distance
    )


# Main entry point
if __name__ == "__main__":
    main()
