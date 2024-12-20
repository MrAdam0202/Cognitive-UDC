======================================================================
EMERGE_Distance_Scenario: Using Utility Model from 3D Simple Spherical Motions in EMERGE scenario in 3D space
======================================================================

EMERGE_Distance_Scenario is another part of the project designed to develop and train the Utility Model (UM), a machine learning model that evaluates and predicts the effectiveness of robot actions in achieving a goal. The scenario simulates the motion of a 3-joint real EMERGE robot moving (motion limited to only 2 joints) in 3D space towards a randomly placed target by predicting utility values for possible motion actions through UM learned by simple spherical motions in 3D space, which are then used to determine the best execution of actions.

The utility model evaluates potential actions by assigning a utility score that reflects how advantageous a given action is for approaching a given target. This score allows the robot to make informed decisions about the next course of action and optimize the path to the goal.

In contrast to the EMERGE_Coordinates scenario, where the output data from WM had to be converted to the required inputs to UM (Next Coordinates of Next Position to Next Distance and Next Unit Vector), this scenario focuses on obtaining the next state in the same format as the input to UM. Thus, the output from WM is the predicted Next Distance and Next Unit Vector.

Unfortunately, this scenario was not completed within the time frame. Even though the movement options were limited from 3 joints to only 2 joints, the data generated for WM was insufficient. In my case, 5,000 random goals were created with 10 random actions. This caused the robot to behave differently than expected and oscillate in some of the variations specified by the random goal position and start position. Another shortcoming was the range of steps learned in UM. If the robot was more than 0.1 units away from the target, the Utility Values were the same for all actions.
When the robot's joint motion was constrained to a range of 0 to 30 degrees and the creation of WM within that range, the robot behaved with more deviation, but the behavior was closer to expectations. But after changing the range to 90 degrees, the robot had large errors due to insufficient data.
======================================================================
Recommendations for the next steps of the project
======================================================================
   - Create a larger dataset for learning WM. This means creating more steps for more random goals.
   - Set more last steps to score Utility Values when reaching the goal to expand the range of Utility Values scores. Also, propose a better normalization of the data with distance values.
======================================================================
Project Structure
======================================================================
1. EMERGEDist_Learning3DWM_2Angles.py
   - Purpose: The program simulates the generation of training data for learning world model of EMERGE movements.

2. EMERGEDist_3DGraphOfGeneratedPosAndGoal.py
   - Purpose: The program visualizes the robot's movements and goal positions from WMLearning dataset in 3D space

3. EMERGEDist_WM3D_Generator.py
   - Purpose: This program trains a neural network-based World Model (WM) to predict the robot's next normalized distance and unit vector in 3D space based on the current position and action.
   - Output: A trained Wold Model saved for future use.

4. EMERGEDist_EvaluationOfWM3D_2Angles.py
   - Purpose: The program evaluates predicted positions against actual robot actions by World Model.

5. EMERGEDist_Learning3DUM.py
   - Purpose: This program simulates a robotic movement in 3D spherical space and generates data for training Utility Model.
6. EMERGEDist_UM3D_Generator.py
   - Purpose: This program trains a neural network-based Utility Model (UM) to predict the Utility Value (UV) based on unit vectors and normalized distances in a 3D space.
7. EMERGEDist_3DRobotPerception_2Angles.py
   - Purpose: This program integrates simulation for controlling the EMERGE robot movement via the UM and the WM learned in 3D space. 
   - Significance: Demonstrates the application of the Utility Model learned bz 3D Simple Spherical Motions in decision-making in 3D space using WM.
======================================================================
Each program includes inline comments and function-level explanations to help understand its logic.
======================================================================
Libraries for controlling both the physical and simulation EMERGE robot were obtained from GitHub at the following link: https://github.com/carlosjaravas/TFG_CarlosJara.
The sim_joint_handler.py library has been slightly modified for better universality for this project.
======================================================================
