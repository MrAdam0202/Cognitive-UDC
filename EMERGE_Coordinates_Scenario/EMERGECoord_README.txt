======================================================================
EMERGE_Coordinates_Scenario: Using Utility Model from 2D Grid Scenario in EMERGE scenario in 3D space
======================================================================

EMERGE_Coordinates_Scenario is another part of the project designed to develop and train the Utility Model (UM), a machine learning model that evaluates and predicts the effectiveness of a robot's actions in achieving a goal. The focus of this part of the project is to prove that the UM learned in 2D space can be used in 3D space. The scenario simulates the motion of a three-joint real EMERGE robot moving in 3D space towards a randomly placed target by predicting utility values for possible motion actions, which are then used to determine the best execution of the actions.

The utility model evaluates potential actions by assigning a utility score that reflects how advantageous a given action is for approaching a given target. This score allows the robot to make informed decisions about the next course of action and optimize the path to the goal.

======================================================================
Project Structure
======================================================================

1. EMERGECoord_LearningWorldModelOfRobot.py
   - Purpose: The program simulates the generation of training data for learning world model of EMERGE movements.

2. EMERGECoord_WMGenerator.py
   - Purpose: Trains a neural network model using the dataset. The model predicts next postions based on input features such as current positions and actions.
   - Output: A trained Wold Model saved for future use.

3. EMERGECoord_EvaluationOfWM.py
   - Purpose: The program evaluates predicted positions against actual robot actions by World Model.

4. EMERGECoord_Perception.py
   - Purpose: Simulates the robot movement in 3D space toward a goal using the trained Utility Model from 2D space dataset and World Model. 
   - Significance: Demonstrates the application of the Utility Model learned in 2D space in decision-making in 3D space.
======================================================================
Each program includes inline comments and function-level explanations to help understand its logic.
======================================================================
Libraries for controlling both the physical and simulation EMERGE robot were obtained from GitHub at the following link: https://github.com/carlosjaravas/TFG_CarlosJara.
The sim_joint_handler.py library has been slightly modified for better universality for this project.
======================================================================
