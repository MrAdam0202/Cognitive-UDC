#     |\__/,|   (`\
#   _.|o o  |_   ) )
# -(((---(((--------

import sys

# Simulation Class
class Sim_setup():
    def __init__(self, cluster = False):
        self.cluster = cluster
        if self.cluster == True:
            self.port_conection = int(sys.argv[1])
            print ('Base Port:', self.port_conection)
        # Whether the CESGA is not used, select the port '23000' by default.
        else:
            self.port_conection = 23000

    # When adding new joint handlers add them in loadEMERGE in sim_joint_handlers.py
    class joint_handlers():
        def __init__(self):
            self.J0 = 0
            self.J1 = 0
            self.J2 = 0
    
    class obj_handlers():
        def __init__(self):
            self.p = 0


# EMERGE Class
class EMERGE_setup():
    def __init__(self):
        self.PROTOCOL_VERSION = 1.0    #Protocol version is used in the Dynamixel AX12
        self.BAUDRATE = 1000000        #Dynamixel default baudrate
        self.DEVICENAME = 'COM5'       #Check in device manager

        self.SENSOR_PORT = 'COM6'
        self.SENSOR_BAUD = 250000

    # Control table addresses
    class EEPROM_ADDR():
        def __init__(self):
            self.MODEL_NUMBER = 		0   #2 byte
            self.FIRMWARE_VER = 		2   #1 byte
            self.ID = 					3   #1 byte
            self.BAUD_RATE = 			4   #1 byte
            self.RETURN_DELAY_T = 		5   #1 byte
            self.CW_ANGLE_LIMIT = 		6   #2 byte
            self.CCW_ANGLE_LIMIT = 		8   #2 byte
            self.TEMPERATURE_LIMIT = 	11  #1 byte
            self.MIN_VOLTAGE_LIMIT = 	12  #1 byte
            self.MAX_VOLTAGE_LIMIT = 	13  #1 byte
            self.MAX_TORQUE = 			14  #2 byte
            self.STATUS_RETURN_LVL = 	16  #1 byte
            self.ALARM_LED = 			17  #1 byte
            self.SHUTDOWN = 			18  #1 byte

    class RAM_ADDR():
        def __init__(self):
            self.TORQUE_ENABLE = 		24  #1 byte
            self.LED = 					25  #1 byte
            self.CW_COMP_MARGIN = 		26  #1 byte
            self.CCW_COMP_MARGIN = 		27  #1 byte
            self.CW_COMP_SLOPE = 		28  #1 byte
            self.CCW_COMP_SLOPE = 		29  #1 byte
            self.GOAL_POSITION = 		30  #1 byte
            self.MOVING_SPEED = 		32  #2 byte
            self.TORQUE_LIMIT = 		34  #2 byte
            self.PRESENT_POSITION = 	36  #2 byte
            self.PRESENT_SPEED = 		38  #2 byte
            self.PRESENT_LOAD = 		40  #2 byte
            self.PRESENT_VOLTAGE = 		42  #1 byte
            self.PRESENT_TEMPERATURE =  43  #1 byte
            self.REGISTERED = 			44  #1 byte
            self.MOVING = 				46  #1 byte
            self.LOCK_EEPOM = 			47  #1 byte
            self.PUNCH = 				48  #2 byte
    
    # Control table bytes
    class RAM_BYTES():
        def __init__(self):
            self.TORQUE_ENABLE =    	1   #byte
            self.LED =              	1   #byte
            self.CW_COMP_MARGIN =   	1   #byte
            self.CCW_COMP_MARGIN =      1   #byte
            self.CW_COMP_SLOPE =        1   #byte
            self.CCW_COMP_SLOPE =       1   #byte
            self.GOAL_POSITION =        2   #byte
            self.MOVING_SPEED =         2   #byte
            self.TORQUE_LIMIT =         2   #byte
            self.PRESENT_POSITION =     2   #byte
            self.PRESENT_SPEED =        2   #byte
            self.PRESENT_LOAD =         2   #byte
            self.PRESENT_VOLTAGE =      1   #byte
            self.PRESENT_TEMPERATURE =  1   #byte
            self.REGISTERED =           1   #byte
            self.MOVING =               1   #byte
            self.LOCK_EEPOM =           1   #byte
            self.PUNCH =                2   #byte
    
    class EEPROM_BYTES():
        def __init__(self):
            self.MODEL_NUMBER =         2   #byte
            self.FIRMWARE_VER =         1   #byte
            self.ID =                   1   #byte
            self.BAUD_RATE =            1   #byte
            self.RETURN_DELAY_T =       1   #byte
            self.CW_ANGLE_LIMIT =       2   #byte
            self.CCW_ANGLE_LIMIT =      2   #byte
            self.TEMPERATURE_LIMIT =    1   #byte
            self.MIN_VOLTAGE_LIMIT =    1   #byte
            self.MAX_VOLTAGE_LIMIT =    1   #byte
            self.MAX_TORQUE =           2   #byte
            self.STATUS_RETURN_LVL =    1   #byte
            self.ALARM_LED =            1   #byte
            self.SHUTDOWN =             1   #byte           