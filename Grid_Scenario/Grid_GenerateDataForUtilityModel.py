"""
This program simulates the movement of a robot on a 10x10 grid towards a randomly placed goal. 
The robot moves randomly in one of four directions until it reaches the goal or exceeds a maximum number of moves. 
The program logs the last 10 movements of the robot, including the distance to the goal, the unit vector, and a utility value. 
Optional visualization of the grid is available.
"""

import numpy as np
import random
import os, sys
import matplotlib.pyplot as plt

# Print the system path for debugging purposes
route = sys.path
print(route)

# Constants for grid representation and settings
EMPTY_CELL = 0  # Representation of an empty cell
ROBOT_CELL = 1  # Representation of the robot's position
GOAL_CELL = 2  # Representation of the goal's position
ROWS = 10  # Number of rows in the grid
COLS = 10  # Number of columns in the grid
MAX_MOVES = 1000  # Maximum allowed moves for the robot
GraficVisual = False  # Enable or disable graphical visualization
FILE_PATH = route[0] + "/Data/UtilityValuesNoNorm.txt"  # Path to the output file


# Function to check if a file exists and remove it if necessary
def check_and_remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} has been removed.")
    else:
        print(f"File {file_path} does not exist. A new one will be created.")


# Function to calculate the Euclidean distance between the robot and the goal
def calculate_distance(goal_x, goal_y, robot_x, robot_y):
    return np.sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)


# Function to calculate the unit vector between the robot and the goal
def calculate_unit_vector(goal_x, goal_y, robot_x, robot_y):
    vector_x = goal_x - robot_x
    vector_y = goal_y - robot_y
    distance = np.sqrt(vector_x**2 + vector_y**2)
    if distance == 0:
        return 0, 0  # Unit vector is zero when positions are identical
    unit_vector_x = vector_x / distance
    unit_vector_y = vector_y / distance
    return unit_vector_x, unit_vector_y


# Function to visualize the grid with the robot and goal positions
def plot_grid(data, robot_x, robot_y, goal_x, goal_y):
    plt.clf()
    plt.imshow(data, cmap="coolwarm", origin="upper")
    plt.grid(which="both", color="black", linewidth=1)
    plt.xticks(np.arange(-0.5, 10, 1), [])
    plt.yticks(np.arange(-0.5, 10, 1), [])
    plt.title(f"Robot: ({robot_x}, {robot_y}) | Goal: ({goal_x}, {goal_y})")
    plt.pause(0.001)  # Small pause for smoother updates


# Function to randomly move the robot on the grid
def random_move(grid, robot_x, robot_y):
    move_options = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, down, right, left
    while True:
        move = random.choice(move_options)
        new_x = robot_x + move[0]
        new_y = robot_y + move[1]
        if 0 <= new_x < ROWS and 0 <= new_y < COLS:  # Check boundaries
            return new_x, new_y


# Function to reset the robot to a random position
def reset_robot(grid):
    robot_x = random.randint(0, ROWS - 1)
    robot_y = random.randint(0, COLS - 1)
    grid[robot_x, robot_y] = ROBOT_CELL
    return robot_x, robot_y


# Function to save the last 10 moves of the robot to a file
def save_test_results(moves, test_number, file_name, goal_x, goal_y):
    with open(file_name, "a") as file:
        file.write(f"Test{test_number}\n")  # Header for the test
        for i in range(len(moves)):
            x, y = moves[-(i + 1)]
            utility_value = 1 - i * 0.1
            distance = calculate_distance(goal_x, goal_y, x, y)
            unit_vector_x, unit_vector_y = calculate_unit_vector(goal_x, goal_y, x, y)
            file.write(
                f"{distance:.3f};({unit_vector_x:.3f},{unit_vector_y:.3f});{utility_value:.1f}\n"
            )
        file.write("\n")  # Add a blank line between tests


# Function to execute a single test
def run_single_test(test_number, file_name, GraficVisual):
    grid = np.zeros((ROWS, COLS))  # Initialize the grid

    # Place the goal randomly on the grid
    goal_x = random.randint(0, ROWS - 1)
    goal_y = random.randint(0, COLS - 1)
    grid[goal_x, goal_y] = GOAL_CELL

    # Place the robot randomly on the grid
    robot_x, robot_y = reset_robot(grid)

    # Store robot moves
    moves = []

    # Enable visualization if specified
    if GraficVisual:
        plt.ion()
        fig, ax = plt.subplots()

    move_count = 0  # Track the number of moves
    while (robot_x, robot_y) != (goal_x, goal_y):
        if move_count >= MAX_MOVES:
            print(
                f"Test {test_number}: Robot did not reach the goal in {MAX_MOVES} moves."
            )
            grid[robot_x, robot_y] = EMPTY_CELL
            robot_x, robot_y = reset_robot(grid)
            move_count = 0
            moves.clear()
        else:
            grid[robot_x, robot_y] = EMPTY_CELL
            robot_x, robot_y = random_move(grid, robot_x, robot_y)
            grid[robot_x, robot_y] = ROBOT_CELL
            move_count += 1
            moves.append((robot_x, robot_y))
            if move_count % 100 == 0:
                print(f"Test {test_number}: {move_count} moves performed.")
            if GraficVisual:
                plot_grid(grid, robot_x, robot_y, goal_x, goal_y)

    print(f"Test {test_number}: Robot reached the goal in {move_count} moves!")
    save_test_results(moves[-10:], test_number, file_name, goal_x, goal_y)

    if GraficVisual:
        plt.ioff()
        plt.show()


def main():
    """
    Main function to execute multiple tests of robot navigation on the grid.
    """
    check_and_remove_file(FILE_PATH)  # Remove existing file, if any
    for test_number in range(1, 1001):
        run_single_test(test_number, FILE_PATH, GraficVisual)


if __name__ == "__main__":
    main()
