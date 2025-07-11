# Server configuration
SERVER_URL = "https://ca179e8e1acc.ngrok-free.app/act"
TASK_INSTRUCTION = "clean plate"

# Control parameters
CONTROL_FREQUENCY = 10.0  # Hz
MAX_STEPS = 10000
DISPLAY_STATUS = True
CHUNK_SIZE = 1

# Timeouts
CAMERA_TIMEOUT = 1000  # seconds
STEP_TIMEOUT = 500.0   # seconds

# Display settings
PRINT_FREQUENCY_EVERY_N_STEPS = 10
WAIT_FOR_GRIPPER = True

# Camera configuration
CAMERA_CONFIG = {
    'fps': 30,
    'head_camera_type': 'realsense',
    'head_camera_image_shape': [720, 1280],  # Head camera resolution
    'head_camera_id_numbers': ["335622071386"],
    'fps': 30,
    'wrist_camera_type': 'realsense',
    'wrist_camera_image_shape': [720, 1280],  # Fix camera resolution
    'wrist_camera_id_numbers': ["336222076815"],
}