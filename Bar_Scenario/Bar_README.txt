======================================================================
Bar_Scenario: Using Utility Model from Grid Scenario in Bar scenario
======================================================================

Bar_Scenario is another part of the project designed to develop and train the Utility Model (UM), a machine learning model that evaluates and predicts the effectiveness of a robot's actions in achieving a goal. The scenario simulates the motion of a 2-jointed robot moving in 2D space towards a randomly located target by predicting utility values for possible motion actions, which are then used to determine the best execution of the actions.

The Utility Model evaluates potential actions by assigning a utility score that reflects how advantageous an action is for approaching a given goal. This score allows the robot to make informed decisions about its next course of action and optimize its path to the goal.

======================================================================
Project Structure
======================================================================

1. Bar_GenerateRandomTestPositions.py
   - Purpose: Generates random test scenarios, creating starting and goal positions for the robot. These coordinates can be used to test the trained Utility Model.

2. Bar_Perception.py
   - Purpose: Simulates the 2-jointed robot navigating a grid toward a target using the trained Utility Model. It selects the best move based on utility scores predicted by the model.
   - Significance: Demonstrates the application of the Utility Model in decision-making.

======================================================================
Each program includes inline comments and function-level explanations to help understand its logic. 
======================================================================

