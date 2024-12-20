======================================================================
Grid_Scenario: Building a Utility Model for Robotic Navigation
======================================================================

The Grid_Scenario is a foundational part of the project designed to develop and train a Utility Model (UM), a machine learning model that scores and predicts the effectiveness of robot actions in achieving a goal. The scenario simulates a robot navigating a 2D grid toward a randomly placed target, capturing key metrics like distance, direction, and success rate. These simulations are then used to train the Utility Model, which can later be applied to real-world robotic tasks or other simulations requiring decision-making.

The Utility Model evaluates potential actions by assigning a utility score, reflecting how advantageous an action is for approaching a given goal. This score enables the robot to make informed decisions about its next move, optimizing its path to the target.

======================================================================
Project Structure
======================================================================

1. Grid_GenerateDataForUtilityModel.py
   - Purpose: Generates the dataset for training the Utility Model. It simulates robot movements on a grid toward randomly placed goals, capturing metrics like:
       - Distance to the goal.
       - Unit vector for direction.
       - A utility score for the robot's final movements.
   - Output: Logs the last 10 movements of the robot for each test in a text file.

2. Grid_NormalizeData.py
   - Purpose: Normalizes the dataset by scaling distances to a [0, 1] range, ensuring consistent training and better model performance.
   - Usage: Preprocesses data for compatibility with machine learning models.

3. Grid_UtilityModelGenerator.py
   - Purpose: Trains a neural network model using the normalized dataset. The model predicts utility scores based on input features such as distance and direction.
   - Output: A trained Utility Model saved for future use.

4. Grid_RandomTestPositions.py
   - Purpose: Generates random test scenarios, creating starting and goal positions for the robot. These coordinates can be used to test the trained Utility Model.

5. Grid_Perceptions.py
   - Purpose: Simulates the robot navigating a grid toward a target using the trained Utility Model. It selects the best move based on utility scores predicted by the model.
   - Significance: Demonstrates the application of the Utility Model in decision-making.

======================================================================
Each program includes inline comments and function-level explanations to help understand its logic. 
======================================================================

