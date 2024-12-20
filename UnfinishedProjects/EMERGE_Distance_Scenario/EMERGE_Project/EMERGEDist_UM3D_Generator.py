"""
This program trains a neural network-based Utility Model (UM) to predict the Utility Value (UV)
based on unit vectors and normalized distances in a 3D space. It processes data from a CSV file,
trains a model using Keras, and saves the trained model, its structure, and learning curves for
future analysis.
"""

import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from numpy import sqrt
from matplotlib import pyplot  # Learning curves
from tensorflow.keras.callbacks import EarlyStopping
import time, sys

# Get the current working directory and display system paths
route = sys.path
print(route)

# Generate a timestamp for file saving
timestr = time.strftime("%Y_%d%m_%H%M")


# Load data from a CSV file into a structured NumPy array
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)

        unit_vectors = df["Unit Vector"].apply(
            lambda x: list(map(float, x.strip("()").split(", ")))
        )
        norm_distance = df["Norm_Distance"]
        uv = df["UV"]

        data = np.array(
            [
                [*unit_vector, norm_dist, uv_val]
                for unit_vector, norm_dist, uv_val in zip(
                    unit_vectors, norm_distance, uv
                )
            ]
        )

        print(f"Data loaded, number of records: {len(data)}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Plot learning curves for training and validation losses
def plot_learning_curves(history, timestr):
    pyplot.title("Learning Curves")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Mean Squared Error Loss")
    pyplot.plot(history.history["loss"], label="train")
    pyplot.plot(history.history["val_loss"], label="val")
    pyplot.ylim([0.0, 0.1])
    pyplot.legend()
    filename = route[0] + f"/Data/Models/UM_LearningGraph_{timestr}.png"
    pyplot.savefig(filename)
    print(f"Learning curve graph saved: {filename}")
    pyplot.close()


# Train the Utility Model (UM) using the provided data
def train_model(data):
    X = data[:, :4]  # Unit Vector (x, y, z) + Normalized Distance
    y = data[:, 4]  # UV (Utility Value)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = Sequential()
    model.add(
        Dense(10, activation="relu", kernel_initializer="he_normal", input_shape=(4,))
    )
    model.add(Dense(8, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # EarlyStopping callback
    # es = EarlyStopping(monitor="val_loss", patience=3)

    # Train the model and capture training history
    history = model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=64,
        verbose=1,
        validation_split=0.3,
        # callbacks=[es],
    )

    error = model.evaluate(X_test, y_test, verbose=0)

    plot_learning_curves(history, timestr)

    model_structure_filename = route[0] + f"/Data/Models/UM_structure_{timestr}.png"
    plot_model(
        model,
        to_file=model_structure_filename,
        show_shapes=True,
        show_layer_names=True,
    )
    print(f"Model structure saved: {model_structure_filename}")

    model_filename = route[0] + f"/Data/Models/UtilityModelfor3DRobot_{timestr}.h5"
    model.save(model_filename)
    print(f"Model saved: {model_filename}")

    print("Mean Squared Error (MSE): %.3f" % error)
    print("Root Mean Squared Error (RMSE): %.3f" % sqrt(error))

    return model


def main():
    # Path to the CSV file containing data
    data_file_path = (
        route[0] + f"/Data/DataForUM/EMERGEDist_SphericalDataForUM__2024_0912_1300.csv"
    )

    # Load the data from the CSV file
    data = load_data(data_file_path)
    if data is not None:
        print(f"Data loaded, number of records: {data.shape[0]}")

        # Train the Utility Model using the loaded data
        train_model(data)


if __name__ == "__main__":
    main()
