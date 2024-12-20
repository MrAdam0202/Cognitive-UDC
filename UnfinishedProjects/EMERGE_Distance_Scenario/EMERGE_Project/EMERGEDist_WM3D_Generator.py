"""
This program trains a neural network-based World Model (WM) to predict the robot's 
next normalized distance and unit vector in 3D space based on the current position 
and action. 
The training data includes the current norm distance, current unit vector, 
and joint action as inputs, while the outputs are the next norm distance and next 
unit vector. 
The program includes functionality for loading data, visualizing training 
progress, saving the trained model, and estimating training time.
"""

import time, sys
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from numpy import sqrt
from matplotlib import pyplot
from tensorflow.keras.callbacks import EarlyStopping

# Get the current working directory and display system paths
route = sys.path
print(route)

# Generate a timestamp for file saving
timestr = time.strftime("_%Y_%d%m_%H%M")


# Custom callback to estimate remaining time for model training
class TimeEstimator(Callback):

    def on_train_begin(self, logs=None):
        self.start_time = None
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time()
        epoch_duration = end_time - self.start_time
        self.epoch_times.append(epoch_duration)

        avg_time_per_epoch = np.mean(self.epoch_times)
        remaining_epochs = self.params["epochs"] - (epoch + 1)
        remaining_time = avg_time_per_epoch * remaining_epochs

        mins, secs = divmod(int(remaining_time), 60)
        print(
            f"Epoch {epoch + 1}/{self.params['epochs']} finished. Estimated remaining time: {mins}m {secs}s"
        )


# Loads data from a CSV file and prepares it for training
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)

        df["Current Unit Vector"] = df["Current Unit Vector"].apply(eval)
        df["Action"] = df["Action"].apply(eval)
        df["Next Unit Vector"] = df["Next Unit Vector"].apply(eval)

        current_norm_distance = df["Current Norm Distance"].values.reshape(-1, 1)
        current_unit_vector = np.array(df["Current Unit Vector"].tolist())
        action = np.array(df["Action"].tolist())

        X = np.hstack([current_norm_distance, current_unit_vector, action])

        next_norm_distance = df["Next Norm Distance"].values.reshape(-1, 1)
        next_unit_vector = np.array(df["Next Unit Vector"].tolist())

        y = np.hstack([next_norm_distance, next_unit_vector])

        print(f"Rows prepared for training: {X.shape[0]}")
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


# Plot the training and validation loss curves over epoch
def plot_learning_curves(history):
    pyplot.title("Learning Curves")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.plot(history.history["loss"], label="train")
    pyplot.plot(history.history["val_loss"], label="val")
    pyplot.legend()
    filepath = route[0] + f"/Data/Models/GraphWorldModelforRobot_{timestr}.png"
    pyplot.savefig(filepath)
    print(f"Graph saved: {filepath}")


# Train a neural network to predict Next Norm Distance and Next Unit Vector
def train_model(X, y, save_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = Sequential()
    model.add(
        Dense(
            16,
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(X.shape[1],),
        )
    )
    model.add(Dense(12, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer="adam", loss="mse")

    # Add custom callback for time estimation
    time_estimator = TimeEstimator()

    history = model.fit(
        X_train,
        y_train,
        epochs=1500,
        batch_size=32,
        verbose=1,
        validation_split=0.3,
        callbacks=[time_estimator],
    )
    error = model.evaluate(X_test, y_test, verbose=0)

    # Plot learning curves
    plot_learning_curves(history)

    # Safe structure of the model
    structure_path = (
        route[0] + f"/Data/Models/WorldModelforRobot_Structure_{timestr}.png"
    )
    plot_model(
        model,
        to_file=structure_path,
        show_shapes=True,
    )
    print(f"Model structure saved: {structure_path}")

    # Save the model
    save_path = save_path.replace(".h5", f"{timestr}.h5")
    model.save(save_path)
    print(f"Model saved: {save_path}")

    print("MSE: %.8f, RMSE: %.8f" % (error, sqrt(error)))

    return model


def main():
    file_path = (
        route[0]
        + f"/Data/DataForWM/2Angles/EMERGEDist_LearningDataForWM_2Angles_2024_1712_2244.csv"
    )
    save_path = (
        route[0] + f"/Data/Models/EMERGEDist_WorldModelforRobot_2Angles_{timestr}.h5"
    )

    X, y = load_data(file_path)
    if X is None or y is None:
        print("Error: Data could not be loaded.")
        return

    train_model(X, y, save_path)
    pyplot.show()


if __name__ == "__main__":
    main()
