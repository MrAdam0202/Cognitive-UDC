#     |\__/,|   (`\
#   _.|o o  |_   ) )
# -(((---(((--------

import math
from dynamixel_sdk import *                    # Uses Dynamixel SDK library
from emerge_classes import *

MODULE = EMERGE_setup()
RAM_ADDR = MODULE.RAM_ADDR()
EEPROM_ADDR = MODULE.EEPROM_ADDR()
RAM_BYTES = MODULE.RAM_BYTES()
EEPROM_BYTES = MODULE.EEPROM_BYTES()

class JointHandlerEMERGE():
    def __init__(self, portHandler, packetHandler, sensorHandler):
        self.portHandler = portHandler # Get methods and members of PortHandler: PortHandler(DEVICENAME)
        self.packetHandler = packetHandler # Get methods and members of Protocol1PacketHandler: PacketHandler(PROTOCOL_VERSION)
        self.sensorHandler = sensorHandler # Get methods and members of SensorHandler to read the sensors via Arduino

        # Robot depending values
        self.joint_min_position = 203
        self.joint_max_position = 821
        self.joint_min_velocity = 62
        self.joint_max_velocity = 123

        self.exact_rad =1.0/180.0*math.pi
        self.increment = 7.5/180.0*math.pi
        self.max_num_joint = 12        
        
        # Method needed variables
        self.joint_ids = []
        self.num_joints = 0 

        # Orientation control
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
    
    # Gives a the position of the motor as an int
    def rad2AX(self, rad):
        offset = 15.0/18.0*math.pi
        rel = 512.0/offset
        AX_pos = (rad + offset)*rel
        return round(AX_pos)
    
    # Gives a the position of the motor in rad
    def AX2rad(self, AX_pos):
        offset = 15.0/18.0*math.pi
        rel = 512.0/offset
        rad = AX_pos/rel - offset
        return rad
    
    # Gives the velocity in dps --------------------- Needs validation
    def AX2dps(self, AX_speed):
        dps = float(AX_speed)/0.666
        return dps

    def enableJointMovement(self, joint):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, joint, RAM_ADDR.TORQUE_ENABLE, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            pass

    def disbleJointMovement(self, joint):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, joint, RAM_ADDR.TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            pass

    def setJointMinTargetPosition(self, joint):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, joint, EEPROM_ADDR.CW_ANGLE_LIMIT, self.joint_min_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            pass
    
    def setJointMaxTargetPosition(self, joint):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, joint, EEPROM_ADDR.CCW_ANGLE_LIMIT, self.joint_max_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            pass

    def getObjectPosition(self, object): # To be defined using cameras
        #x, y, z = self.sim.getObjectPosition(object, self.obj_pos_relative_to)
        #return x, y, z
        pass

    # Functions to move joints
    def getJointPosition(self, joint):
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, joint, RAM_ADDR.PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            pass
        position = self.AX2rad(dxl_present_position)
        return position
    
    # Moves the module to a position
    def setJointAngularPosition(self, joint, target_position_rad):
        target_position = self.rad2AX(target_position_rad)
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, joint, RAM_ADDR.GOAL_POSITION, target_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            pass
    
    # Sets up the joint velocity
    def setJointTargetVelocity(self, joint, target_speed): # 0 and 1023 = max, 1 = min
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, joint, RAM_ADDR.MOVING_SPEED, target_speed)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            pass

    # Velocity profiles
    def trapezoidalVelocityProfile(self, progress):
        min_velocity = self.joint_min_velocity
        max_velocity = self.joint_max_velocity
        p1 = 0.3
        p2 = 0.7
        p3 = 1
        # Remaining percentage to goal
        if progress <= p1:
            x = progress
            m = (max_velocity-min_velocity)/p1
            b = min_velocity
            velocity = x * m + b
        elif p1 < progress and progress < p2:
            velocity = max_velocity
        else:
            x = progress
            m = (min_velocity-max_velocity)/(p3-p2)
            b = min_velocity - p3* m
            velocity = x * m + b
        return round(velocity)

    # Instead of keeping max velocity, once it reaches the max velocity it starts to slow down gradually between 30% and 70%
    def trapezoidalModVelocityProfile(self, progress):
        min_velocity = self.joint_min_velocity
        max_velocity = self.joint_max_velocity
        max_perc = 4/5
        p1 = 0.3
        p2 = 0.7
        p3 = 1
        # Remaining percentage to goal
        if progress <= p1:
            x = progress
            m = (max_velocity-min_velocity)/p1
            b = min_velocity
            velocity = x * m + b
        elif p1 < progress and progress < p2:
            x = progress
            m = ((max_perc*max_velocity)-max_velocity)/(p2-p1)
            b = max_velocity - p1* m
            velocity = x * m + b
        else:
            x = progress
            m = (min_velocity-(max_perc*max_velocity))/(p3-p2)
            b = min_velocity - p3* m
            velocity = x * m + b
        return round(velocity)

    # Moves the module until it gets to a final position in a range    
    def setJointTargetPosition(self, joint, target_position_rad):
        self.setJointAngularPosition(joint, target_position_rad)

        act_pos = self.getJointPosition(joint)
        counter = 0
        diff = act_pos - target_position_rad
        error = abs(diff)

        while abs(error) > self.exact_rad or counter < 5:

            # Adjusts velocity of the motion inversely proportional to the distance to the target position
            if abs(diff) > self.increment:
                dist_percent = abs(error/diff)
                progress = 1 - dist_percent
                velocity = self.trapezoidalVelocityProfile(progress)               
            else: 
                velocity = self.joint_min_velocity

            self.setJointTargetVelocity(joint,velocity)
            self.setJointAngularPosition(joint, target_position_rad)
            
            new_pos = self.getJointPosition(joint)

            if round(act_pos, 3) == round(new_pos, 3):
                counter += 1
            else:
                counter = counter
        
            act_pos = new_pos
            error = new_pos - target_position_rad

    # Moves to initial position
    def setJointInitialPosition(self, joint, initial_position_rad = 0):
        self.setJointTargetPosition(joint, initial_position_rad)
        
    # Pings a Dynamixel using its ID to get the model number
    def getJointStatus(self, joint): 
        # Get Dynamixel model number, for AX-12A should be 12
        dxl_model_number, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, joint)
        if dxl_comm_result != COMM_SUCCESS:
            #print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            #print("[ID:%03d] ping Failed." % (joint))
            return False
        elif dxl_error != 0:
            #print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            #print("[ID:%03d] ping Failed." % (joint))
            return False
        else:
            #print("[ID:%03d] ping Succeeded. Dynamixel model number : %d" % (joint, dxl_model_number))
            print("[ID:%03d] ping Succeeded." % (joint))
            return True
    
    # Pings all joints in a range and updates the list with the ones available
    def getJointList(self):
        self.joint_ids = []
        for joint in range(1,self.max_num_joint+1):
            isJoint = self.getJointStatus(joint)
            if isJoint:
                self.joint_ids.append(joint)
        self.num_joints = len(self.joint_ids)


    # -------------------------------- SENSORS --------------------------------
    def connectArduino(self):
        self.sensorHandler.connectArduino()
    
    def disconnectArduino(self):
        self.sensorHandler.disconnectArduino()
    
    def getHeight(self):
        height = self.sensorHandler.getDistance()
        return height
    
    def getAngularVelocity(self):
        xdps,ydps,zdps = self.sensorHandler.getDPS()
        xrps = xdps*math.pi/180
        yrps = ydps*math.pi/180
        zrps = zdps*math.pi/180
        return xrps, yrps, zrps

    
    
    # Connects to the EMERGE Modules
    def connectEMERGE(self):
        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
        # Set port baudrate
        if self.portHandler.setBaudRate(MODULE.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate") 

    def disconnectEMERGE(self):
        pass

    # Allows to start testing with the robot
    def loadEMERGE(self):
        self.connectEMERGE()
        self.getJointList()
        for joint in self.joint_ids:
            self.setJointMinTargetPosition(joint)
            self.setJointMaxTargetPosition(joint)
            self.enableJointMovement(joint)
            self.setJointTargetVelocity(joint,self.joint_min_velocity)
            self.setJointInitialPosition(joint)

    def unloadEMERGE(self):
        for joint in self.joint_ids:
            self.disbleJointMovement(joint)
        self.portHandler.closePort()