"""
This program processes a text file containing utility data for a robot. 
It normalizes the distances in the dataset, ensuring they fall within a [0, 1] range based on the maximum distance, 
and saves the normalized data to a new file for further analysis.
"""

import numpy as np
import os
import sys

# Initialize paths
route = sys.path
print(route)

# Paths to input and output files
input_file_path = (
    route[0] + "/Data/UtilityValuesNoNorm.txt"
)  # Input file with raw utility values
output_file_path = (
    route[0] + "/Data/UtilityValues.txt"
)  # Output file for normalized utility values


# Function to load data from a text file while ignoring headers and blank lines
def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("Test"):  # Skip headers or blank lines
                continue
            try:
                # Split line into components and parse them
                distance, unit_vector, utility = line.split(";")
                dist = float(distance)  # Convert distance to float
                unit_vec_x, unit_vec_y = map(
                    float, unit_vector.strip("()").split(",")
                )  # Parse unit vector
                utility = float(utility)  # Convert utility to float
                data.append([dist, (unit_vec_x, unit_vec_y), utility])
            except ValueError:
                print(f"Warning: Unable to process line - {line}")
                continue
    return data


# Function to normalize the distances in the dataset
def normalize_data(data):
    distances = [row[0] for row in data]  # Extract all distances
    max_distance = max(distances)  # Find the maximum distance
    for row in data:
        row[0] = row[0] / max_distance  # Normalize each distance
    print("Max value:", max_distance)  # Debug: Print the maximum distance
    return data


# Function to save the normalized data to a file
def save_data(file_path, data):
    # Delete the file if it already exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # Create a new file and write the normalized data
    with open(file_path, "w") as file:
        for row in data:
            distance = f"{row[0]:.3f}"  # Format normalized distance
            unit_vector = f"({row[1][0]:.3f},{row[1][1]:.3f})"  # Format unit vector
            utility = f"{row[2]:.1f}"  # Format utility value
            file.write(f"{distance};{unit_vector};{utility}\n")  # Write to file


def main():
    data = load_data(input_file_path)  # Load data from the input file
    normalized_data = normalize_data(data)  # Normalize distances in the dataset
    save_data(
        output_file_path, normalized_data
    )  # Save normalized data to the output file

    print("Data have been successfully normalized and saved.")


if __name__ == "__main__":
    main()
