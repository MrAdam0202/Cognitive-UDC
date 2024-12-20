"""
This program trains a neural network (World Model) to predict the next position of a EMERGE robot in a 3D space. 
It uses input data consisting of the robot's current position and joint angle changes to predict the next position. 
The program loads data from a CSV file, trains the model, visualizes learning curves and model architecture, 
and saves the trained model for future use.
"""

import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from numpy import sqrt
from matplotlib import pyplot
from tensorflow.keras.callbacks import EarlyStopping
import time, sys

# Get the system path for file handling
route = sys.path
print(route)

timestr = time.strftime("_%Y_%d%m_%H%M")


# Load data from a CSV file and prepares it for training
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=",")
        print(f"Total rows in the input CSV: {len(df)}")  # Log total rows

        # Extract and process input features and target outputs
        act_pos = np.array(
            df["act_pos"].apply(eval).tolist()
        )  # Convert strings to tuples
        changes = np.array(
            df["changes"].apply(eval).tolist()
        )  # Convert strings to lists
        next_pos = np.array(
            df["next_pos"].apply(eval).tolist()
        )  # Convert strings to tuples

        # Combine current position and joint changes into a single input array
        X = np.hstack([act_pos, changes])  # Shape: (N, 6)
        y = next_pos  # Shape: (N, 3)

        print(f"Rows prepared for training: {X.shape[0]}")  # Log prepared rows
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


# Plot the training and validation loss curves over epochs
def plot_learning_curves(history):
    pyplot.title("Learning Curves")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.plot(history.history["loss"], label="train")  # Training loss
    pyplot.plot(history.history["val_loss"], label="val")  # Validation loss
    pyplot.ylim([0.0, 0.01])  # Set the y-axis range for better visibility
    pyplot.legend()
    pyplot.savefig(
        route[0] + f"/Data/Models/GraphWorldModelforRobot_{timestr}.png"
    )  # Save graph as an image
    print("Graph saved.")  # Log success


# Train a neural network to predict `next_pos` (x, y, z) based on `act_pos` and `changes`
def train_model(X, y, save_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = Sequential()
    model.add(
        Dense(10, activation="relu", kernel_initializer="he_normal", input_shape=(6,))
    )
    model.add(Dense(8, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(3))  # Output layer for predicting x, y, z
    model.compile(optimizer="adam", loss="mse")
    # es = EarlyStopping(monitor="val_loss", patience=3)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=2000,
        batch_size=32,
        verbose=0,
        validation_split=0.3,
        # callbacks=[es],
    )
    error = model.evaluate(X_test, y_test, verbose=0)

    # Plot learning curves
    plot_learning_curves(history)
    # Visualize the model architecture
    path = route[0] + f"/Data/Models/Bar_WorldModelforRobot_{timestr}.png"
    plot_model(
        model,
        path,
        show_shapes=True,
    )
    print("MSE: %.8f, RMSE: %.8f" % (error, sqrt(error)))
    model.save(save_path)
    print("Model saved.")
    print(f"Training was over {len(history.history['loss'])} epoch.")

    return model


def main():
    file_path = route[0] + f"/Data/EMERGECoord_DataForWM.csv"
    save_path = route[0] + f"/Data/Models/EMERGECoord_WorldModelforRobot_{timestr}.h5"

    # Load input data and target outputs
    X, y = load_data(file_path)
    if X is None or y is None:
        print("Error: Data could not be loaded.")  # Log error if loading fails
        return

    # Train the model
    train_model(X, y, save_path)
    pyplot.show()  # Show the learning curve graph


if __name__ == "__main__":
    main()
