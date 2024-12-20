======================================================================
Bar_2DWMAnd3DUM_Scenario: Using Utility Model from 3D Simple Spherical Motions in Bar Scenario in 2D space
======================================================================

Bar_2DWMAnd3DUM_Scenario is another part of the project designed to develop and train the Utility Model (UM), a machine learning model that evaluates and predicts the effectiveness of robot actions in achieving a goal. The scenario simulates the motion of a 2-joint mathematical robot moving in 2D space towards a randomly placed target by predicting utility values for possible motion actions through UM learned by simple spherical motions in 3D space, which are then used to determine the best execution of actions.

The utility model evaluates potential actions by assigning a utility score that reflects how advantageous a given action is for approaching a given target. This score allows the robot to make informed decisions about the next course of action and optimize the path to the goal.

This scenario presents the possibility of applying UM learned in 3D space using simple spherical motions in a mathematical WM taking place only in 2D.

Unfortunately, this scenario could not be completed within the given timeframe.  The shortcoming of this project was the extent of the steps learned in UM. If the robot was more than 0.1 units away from the target, the utility values for all actions were the same, so a higher value for max_distance was used to approximate the size of the learning data for UM. There was also an oscillation problem due to the lack of accuracy of the Utility Values.
======================================================================
Recommendations for the next steps of the project
======================================================================
   - Create a larger dataset for learning WM. This means creating more steps for more random goals.
   - Set more last steps to score Utility Values when reaching the goal to expand the range of Utility Values scores. Also, propose a better normalization of the data with distance values.
======================================================================
Project Structure
======================================================================
1. Bar3DUM_Perception.py
   - Purpose: This program simulates the movement of a two-jointed robot arm in a 2D space using a 3D Utility Model in 2D Space.
======================================================================
Each program includes inline comments and function-level explanations to help understand its logic.
======================================================================
