"""
This program trains a neural network model to predict utility values based on input features such as distance and unit vectors. 
It uses supervised learning with data loaded from a text file.
"""

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from numpy import sqrt
from matplotlib import pyplot  # learning curves
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys

# Initialize paths
route = sys.path
# print(route)


# Function to load data from a text file
def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("Test"):
                continue
            try:
                distance, unit_vector, utility = line.split(";")
                dist = float(distance)
                unit_vec_x, unit_vec_y = map(float, unit_vector.strip("()").split(","))
                utility = float(utility)
                data.append([dist, unit_vec_x, unit_vec_y, utility])
            except ValueError:
                print(f"Varování: Nelze zpracovat řádek - {line}")
                continue
    return np.array(data)


# Function to plot the learning curves during training
def plot_learning_curves(history):
    pyplot.title("Learning Curves")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], label="train")
    pyplot.plot(history.history["val_loss"], label="val")
    pyplot.legend()
    pyplot.savefig(
        "/home/adam/Go1Project/SelfLearningApps/AdvancedGridAndBarsScenario/Grid/Data/ModelGrafOfLearning.png"
    )
    print("Graf is saved")


# Function to train a neural network model
def train_model(data):
    X = data[:, :3]
    y = data[:, 3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = Sequential()
    model.add(
        Dense(10, activation="relu", kernel_initializer="he_normal", input_shape=(3,))
    )
    model.add(Dense(8, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=3)  # EarlyStopping
    # Trénink modelu a uložení historie
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        verbose=0,
        validation_split=0.3,
        callbacks=[es],
    )
    error = model.evaluate(X_test, y_test, verbose=0)

    # Plot learning curves
    plot_learning_curves(history)

    # Save the model architecture as a visualization
    model.summary()
    plot_model(
        model,
        "/home/adam/Go1Project/SelfLearningApps/AdvancedGridAndBarsScenario/Grid/Data/ModelDescription.png",
        show_shapes=True,
    )
    print("MSE: %.3f, RMSE: %.3f" % (error, sqrt(error)))

    # Save the trained model
    model.save(
        "/home/adam/Go1Project/SelfLearningApps/AdvancedGridAndBarsScenario/Grid/Models/UtilityModelforRobot.h5"
    )
    print("Model is saved.")


def main():
    # Load data and train the model
    data = load_data(
        "/home/adam/Go1Project/SelfLearningApps/AdvancedGridAndBarsScenario/Grid/Data/GridUtilityValues.txt"
    )
    train_model(data)


if __name__ == "__main__":
    main()
