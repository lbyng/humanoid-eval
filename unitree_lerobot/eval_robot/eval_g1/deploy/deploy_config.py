# Model inference server
SERVER_URL = "https://95ea22471a00.ngrok-free.app/act"

# Task instruction
TASK_INSTRUCTION = "trash bussing"

# Control parameters
CONTROL_FREQUENCY = 10.0
MAX_STEPS = 10000
DISPLAY_STATUS = True

# Camera configuration
CAMERA_CONFIG = {
    'fps': 30,
    'head_camera_type': 'opencv',
    'head_camera_image_shape': [720, 1280],
    'head_camera_id_numbers': ["335622071386"],
    }

# Initial robot pose (set to None to use zero position)
INITIAL_ARM_POSE = None

# Safety parameters
CAMERA_TIMEOUT = 1000000
STEP_TIMEOUT = 1000000.0

# Performance monitoring
PRINT_FREQUENCY_EVERY_N_STEPS = 10

# Action chunking size
CHUNK_SIZE = 1 

# Whether to wait after gripper state changes
WAIT_FOR_GRIPPER = True

# Path to folder containing images
IMAGE_FOLDER = "/home/sean/avp_teleoperate/teleop/utils/converted_data/trash_bussing/0048_rgb/0048"

# Whether to display images
DISPLAY_IMAGES = False