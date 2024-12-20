#     |\__/,|   (`\
#   _.|o o  |_   ) )
# -(((---(((--------
import time
import math
import copy
from emerge_classes import *

SIM = Sim_setup(False)
HANDLERS = SIM.joint_handlers()
OBJ_HANDLERS = SIM.obj_handlers()


class JointHandlerSim:
    def __init__(self, client, sim):
        self.client = client  # client = RemoteAPIClient('localhost', PORT_CONECTION)
        self.sim = sim  # sim = self.client.getObject('sim')

        # Method needed ariables
        self.exact_rad = 1.0 / 180.0 * math.pi
        self.joint_ids = []
        self.num_joints = 0
        self.obj_ids = []
        self.obj_pos_relative_to = self.sim.handle_world

    def getObjectPosition(self, object):
        x, y, z = self.sim.getObjectPosition(object, self.obj_pos_relative_to)
        return x, y, z

    def setObjectPosition(self, object, position):
        self.sim.setObjectPosition(object, position)

    def getJointPosition(self, joint):
        position = self.sim.getJointPosition(joint)
        return position

    # Moves the module until it gets to a final position in a range
    def setJointTargetPosition(self, joint, target_position_rad):
        self.sim.setJointTargetPosition(joint, target_position_rad)
        self.client.step()

        act_pos = self.getJointPosition(joint)
        counter = 0
        counter_loop = 0
        diff = act_pos - target_position_rad
        start_time = time.time()
        max_time = 3
        diff_degree = math.degrees(diff)
        new_pos = act_pos
        # print("\n Check tolerance\n")
        while (
            abs(diff) > self.exact_rad
            and counter < 5
            and time.time() - start_time < max_time
        ):
            counter_loop += 1
            self.sim.setJointTargetPosition(joint, target_position_rad)
            self.client.step()

            new_pos = self.getJointPosition(joint)

            if round(act_pos, 2) == round(new_pos, 2):
                counter += 1
            else:
                counter = counter

            act_pos = new_pos
            diff = new_pos - target_position_rad
            diff_degree = math.degrees(diff)
            # print(f"Loop: {counter_loop}, diff {diff_degree:.3f}, counter {counter}, act {act_pos}, targ {target_position_rad}")
            # time.sleep(0.1)
            # Evaluation of movement
        # print(
        #     f"Loop: {counter_loop}, diff {diff_degree:.3f}, counter {counter}, act {act_pos}, targ {target_position_rad}"
        # )
        # time.sleep(0.5)
        # if abs(diff) > self.exact_rad:
        #     print(
        #         f"Warning: Joint {joint} did not reach the target position {target_position_rad:.2f} rad within the tolerance {self.exact_rad:.3f} rad. Joint reached positiosn {act_pos:.3f} rad."
        #     )

        # else:
        #     print(
        #         f"Joint {joint} reached target position {new_pos:.2f} rad with the tolerance {self.exact_rad:.3f} rad."
        #     )

    def getAllJointPositions(self):
        act_positions = []
        for joint_n in range(self.num_joints):
            position = self.getJointPosition(
                self.joint_ids[joint_n]
            )  # Získání pozice kloubu
            act_positions.append(position)
        return act_positions

    # Moves all modules until it gets to a final position in a range
    def setAllJointTargetPositions(self, target_positions_rad):
        for joint_n in range(self.num_joints):
            joint = self.joint_ids[joint_n]
            target_position_rad = target_positions_rad[joint_n]
            self.sim.setJointTargetPosition(joint, target_position_rad)

        self.client.step()

        act_positions = self.getAllJointPositions()
        counter = 0
        counter_loop = 0
        differences = [a - b for a, b in zip(act_positions, target_positions_rad)]
        start_time = time.time()
        max_time = 3
        differences_degree = [math.degrees(diff) for diff in differences]
        new_positions = act_positions.copy()
        # print("\n Check tolerance\n")
        while (
            any(abs(diff) > self.exact_rad for diff in differences)
            and counter < 5
            and time.time() - start_time < max_time
        ):
            counter_loop += 1
            for joint_n in range(self.num_joints):
                joint = self.joint_ids[joint_n]
                target_position_rad = target_positions_rad[joint_n]
                self.sim.setJointTargetPosition(joint, target_position_rad)

            self.client.step()

            new_positions = self.getAllJointPositions()

            if all(
                round(a, 2) == round(b, 2) for a, b in zip(act_positions, new_positions)
            ):
                counter += 1
            else:
                counter = counter

            act_positions = new_positions.copy()
            differences = [a - b for a, b in zip(act_positions, target_positions_rad)]
            differences_degree = [math.degrees(diff) for diff in differences]

    # Moves to initial position
    def setJointInitialPosition(self, joint, initial_position_rad):
        self.sim.setJointTargetPosition(joint, initial_position_rad)
        self.sim.setJointPosition(joint, initial_position_rad)

    def connectEMERGE(self):
        self.client.setStepping(True)
        # Joints
        HANDLERS.J0 = self.sim.getObject("/J0")
        HANDLERS.J1 = self.sim.getObject("/J1")
        HANDLERS.J2 = self.sim.getObject("/J2")

        # Setting obj handlers
        OBJ_HANDLERS.p = self.sim.getObject("/p")
        OBJ_HANDLERS.peak = self.sim.getObject("/peak")
        OBJ_HANDLERS.sphere = self.sim.getObject("/Goal")

        # Stores the different id for every joint
        self.joint_ids = list(HANDLERS.__dict__.values())
        self.num_joints = len(self.joint_ids)
        self.obj_ids = list(OBJ_HANDLERS.__dict__.values())

    def disconnectEMERGE(self):
        pass

    # Allows to start testing with the robot
    # Starts simulation (not the same as for robot since the order of use should be the same and makes more sense for the robot)
    def loadEMERGE(self):
        self.sim.startSimulation()
        self.client.step()

    def unloadEMERGE(self):
        self.sim.stopSimulation()
