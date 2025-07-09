# deploy_config.py for offline deployment

# Server configuration
SERVER_URL = "https://93257ed8660f.ngrok-free.app/act"
TASK_INSTRUCTION = "lift tray"

# Control parameters
CONTROL_FREQUENCY = 15.0  # Hz
MAX_STEPS = 10000
DISPLAY_STATUS = True
CHUNK_SIZE = 8

# Timeouts
STEP_TIMEOUT = 500.0   # seconds

# Display settings
PRINT_FREQUENCY_EVERY_N_STEPS = 100
WAIT_FOR_GRIPPER = True
DISPLAY_IMAGES = False

# Camera folder configuration
HEAD_FOLDER = "/home/sean/humanoid-teleop/data/converted_data/lift_tray/head/0025"
WRIST_FOLDER = "/home/sean/humanoid-teleop/data/converted_data/lift_tray/fix/0025"
