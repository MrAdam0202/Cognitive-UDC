"""
This program generates random positions for a two-joint robotic arm in a 2D plane. 
The positions are saved to a file, and the arm's movement is visualized in a graph. 
The program can generate joint positions in radians or degrees and calculates the arm's end effector positions 
based on the specified joint angles. It also supports saving the visualization as an image file.
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
import sys
import math


# Determines whether angles are input in degrees (True) or radians (False)
inputs_degrees = False

route = sys.path
print(route)
timestr = time.strftime("_%Y_%d%m_%H%M")  # Timestamp for file naming

# Arm segment lengths
L1, L2 = 1, 1

# Number of positions to generate
num_positions = 5

# Paths for output files
output_path = route[0] + f"/Data/BarTestPositions_{timestr}.txt"
graph_output_path = route[0] + f"/Data/RandomPositionsOfJointsGraph_{timestr}.png"


# Function to calculate joint positions (first and second joint positions)
def joints_positions(alpha, beta):
    x1 = L1 * np.cos(alpha)  # X position of the first joint
    y1 = L1 * np.sin(alpha)  # Y position of the first joint

    x2 = L1 * np.cos(alpha) + L2 * np.cos(beta)  # X position of the second joint
    y2 = L1 * np.sin(alpha) + L2 * np.sin(beta)  # Y position of the second joint

    return x1, y1, x2, y2


def main():
    # Check if the output file exists and remove it if necessary
    if os.path.exists(output_path):
        print(
            f"Soubor {output_path} již existoval, následně byl vymazán a bude vytvořen nový."
        )

    positions = []  # Initialize a list to store positions for graphing

    if inputs_degrees:
        print("Generating positions using degree values.")
        # Open file for writing
        with open(output_path, "w") as file:
            # Generate joint positions
            for _ in range(num_positions):
                # Generate random angles in degrees (0 to 90)
                alpha_deg = np.random.randint(0, 91)
                beta_deg = np.random.randint(0, 91)

                # Convert angles to radians
                alpha_rad = np.radians(alpha_deg)
                beta_rad = np.radians(beta_deg)

                # Calculate joint positions
                x1, y1, x2, y2 = joints_positions(alpha_rad, beta_rad)

                # Print the angles and positions
                print(
                    f"Alpha: {alpha_rad:.3f} rad ({alpha_deg}°), Beta: {beta_rad:.3f} rad ({beta_deg}°) => "
                    f"Pos1: ({x1:.3f}, {y1:.3f}), Pos2: ({x2:.3f}, {y2:.3f})"
                )

                # Write values to the file
                file.write(f"{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}\n")

                # Save positions for visualization
                positions.append([(0, 0), (x1, y1), (x2, y2)])

    else:
        # Open file for writing
        with open(output_path, "w") as file:
            file.write("alpha_rad_start, beta_rad_start; x_goal, y_goal\n")
            # Generate joint positions
            for _ in range(num_positions):
                # Generate random angles in radians (0 to π/2)
                alpha_rad_goal = np.round(np.random.uniform(0, math.pi / 2), 2)
                beta_rad_goal = np.round(np.random.uniform(0, math.pi / 2), 2)
                alpha_rad_start = np.round(np.random.uniform(0, math.pi / 2), 2)
                beta_rad_start = np.round(np.random.uniform(0, math.pi / 2), 2)

                # Calculate goal and start positions
                x1_goal, y1_goal, x2_goal, y2_goal = joints_positions(
                    alpha_rad_goal, beta_rad_goal
                )
                x1_start, y1_start, x2_start, y2_start = joints_positions(
                    alpha_rad_start, beta_rad_start
                )

                # Print angles and positions for start and goal
                print(
                    f"Start: Alpha: {alpha_rad_start:.3f} rad, Beta: {beta_rad_start:.3f} rad => "
                    f"Pos1: ({x1_start:.3f}, {y1_start:.3f}), Pos2: ({x2_start:.3f}, {y2_start:.3f})\n"
                    f"Goal: Alpha: {alpha_rad_goal:.3f} rad, Beta: {beta_rad_goal:.3f} rad => "
                    f"Pos1: ({x1_goal:.3f}, {y1_goal:.3f}), Pos2: ({x2_goal:.3f}, {y2_goal:.3f})"
                )

                # Write values to the file
                file.write(
                    f"{alpha_rad_start:.3f}, {beta_rad_start:.3f}; {x2_goal:.3f}, {y2_goal:.3f}\n"
                )

                # Save positions for visualization
                positions.append([(0, 0), (x1_goal, y1_goal), (x2_goal, y2_goal)])

    print("Positions successfully generated.")

    # Create the visualization
    plt.figure()

    # Plot each position as a separate line
    for i, pos in enumerate(positions):
        x_vals = [p[0] for p in pos]
        y_vals = [p[1] for p in pos]
        plt.plot(x_vals, y_vals)

    # Set the aspect ratio to 1:1
    plt.gca().set_aspect("equal", adjustable="box")

    # Set axis limits
    plt.xlim(0, 2.5)
    plt.ylim(0, 2.5)

    # Plot scatter points for each end effector position
    for i, pos in enumerate(positions):
        x2 = pos[2][0]
        y2 = pos[2][1]
        plt.scatter(x2, y2, label=f"Goal {i + 1}")

    # Add title and axis labels
    plt.title("Robot Arm Movement Simulation")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Add grid and legend
    plt.grid(True)
    plt.legend(loc="upper right", fontsize="small")

    # Save the graph as an image
    plt.savefig(graph_output_path, bbox_inches="tight")
    print(f"Graph saved as an image at {graph_output_path}")

    # Display the graph
    plt.show()

    print("Graph successfully created.")


if __name__ == "__main__":
    main()
