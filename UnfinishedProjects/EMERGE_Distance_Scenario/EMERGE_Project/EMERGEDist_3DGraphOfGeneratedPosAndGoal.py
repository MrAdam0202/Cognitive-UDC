"""
This program visualizes the robot's movements and goal positions in 3D space. 
It loads data from a CSV file containing 'Current Position' and 'Goal Position' columns, 
parses the data into tuples, and creates a 3D scatter plot to compare the positions.
The resulting visualization helps in analyzing the robot's behavior relative to its goals.
The plot is saved as an image in a specified directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast  # For safe literal evaluation of tuple strings
import time, sys

#  Get the current working directory and display system paths
route = sys.path
print(route)

# Generate a timestamp for file saving
timestr = time.strftime("%Y_%d%m_%H%M")


# Function to parse position tuples from strings
def parse_tuple(position_str):
    try:
        return ast.literal_eval(position_str)
    except:
        return None


# Function to load data and plot positions
def load_and_plot(csv_path):
    # Load the CSV file, specifying the first row as headers and skipping the second row
    df = pd.read_csv(csv_path, header=0, skiprows=[1])  # Skip second row
    df.columns = df.columns.str.strip()  # Clean column names (remove spaces)
    save_path = (
        route[0] + f"/Data/DataForWM/2Angles/3DGraphOfGeneratedPosAndGoal_{timestr}.png"
    )
    # Debugging: Print column names
    print("Detected columns:", df.columns)

    # Check for required columns
    if "Current Position" in df.columns and "Goal Position" in df.columns:
        # Parse Current Position and Goal Position
        current_positions = df["Current Position"].apply(parse_tuple)
        goal_positions = df["Goal Position"].apply(parse_tuple)

        # Extract x, y, z coordinates
        current_x, current_y, current_z = zip(*current_positions)
        goal_x, goal_y, goal_z = zip(*goal_positions)

        # Create 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot Goal Positions first for layering
        ax.scatter(
            goal_x, goal_y, goal_z, color="red", s=20, label="Goal Position", zorder=2
        )
        ax.scatter(
            current_x,
            current_y,
            current_z,
            color="blue",
            s=10,
            label="Current Position",
            zorder=1,
        )

        # Add labels and title
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("3D Plot of Current and Goal Positions")
        ax.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Graph saved successfully at {save_path}")
        plt.show()
    else:
        print("Required columns 'Current Position' or 'Goal Position' not found.")


timestr = time.strftime("_%Y_%d%m_%H%M")
# Path to the CSV and output file
csv_path = (
    route[0]
    + f"/Data/DataForWM/2Angles/EMERGEDist_LearningDataForWM_2Angles_2024_1712_2244.csv"
)


# Execute the function
load_and_plot(csv_path)
