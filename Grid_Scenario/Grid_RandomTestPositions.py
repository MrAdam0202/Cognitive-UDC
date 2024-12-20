"""
This program generates random test positions for a robot and its goal within a 2D grid.
The positions are saved to a text file for learning utility model.
"""

import random
import sys

# Initialize paths
route = sys.path
# print(route)


# Generates random test positions for the robot and its goal within a grid
def generate_test_positions(
    num_tests, grid_size=10, max_distance=3, limit_max_distance=True
):
    positions = []

    while len(positions) < num_tests:
        # Generate a random position for the robot
        robot_x = random.randint(0, grid_size - 1)
        robot_y = random.randint(0, grid_size - 1)

        while True:
            # Generate a random goal position based on distance constraints
            if limit_max_distance:
                goal_x = robot_x + random.randint(-max_distance, max_distance)
                goal_y = robot_y + random.randint(-max_distance, max_distance)
            else:
                goal_x = random.randint(0, grid_size - 1)
                goal_y = random.randint(0, grid_size - 1)

            # Ensure the goal position is distinct from the robot position
            if (goal_x, goal_y) != (robot_x, robot_y):
                break

        # Append the robot and goal positions as a formatted string
        positions.append(f"{robot_x}, {robot_y}; {goal_x}, {goal_y}")

    return positions


def main():
    # Parameters for test position generation
    num_tests = 5  # Number of test positions to generate
    grid_size = 100  # Size of the 2D grid
    max_distance = 10  # Maximum distance between robot and goal
    limit_max_distance = False  # Option to disable maximum distance constraint

    # Generate the positions
    positions = generate_test_positions(
        num_tests, grid_size, max_distance, limit_max_distance
    )

    # Save the positions to a text file
    file_path = route[0] + "/Data/TestPositions.txt"
    with open(file_path, "w") as file:
        for position in positions:
            file.write(f"{position}\n")
        print(f"Souradnice byly úspěšně uloženy do souboru: {file_path}")


if __name__ == "__main__":
    main()
