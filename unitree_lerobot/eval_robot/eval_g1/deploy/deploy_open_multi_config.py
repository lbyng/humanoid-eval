# deploy_config.py for offline deployment

# Server configuration
SERVER_URL = "https://ca179e8e1acc.ngrok-free.app/act"
TASK_INSTRUCTION = "close box"
START_INDEX = 30

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
HEAD_FOLDER = "/home/sean/humanoid-teleop/data/converted_data/close_box/head/0008"
WRIST_FOLDER = "/home/sean/humanoid-teleop/data/converted_data/close_box/fix/0008"

LEFT_ARM = [
                        -0.9479023870683034,
                        0.33169206269747153,
                        0.21263253235147372,
                        1.3366596586716044,
                        -0.003964927671710803,
                        -0.3762602732885445,
                        -0.233602907724086
                    ]
RIGHT_AMR = [
                        -0.9110879153188621,
                        -0.2433729344705992,
                        -0.3355893694050562,
                        1.3085806679473562,
                        0.21248294275070462,
                        -0.5420431844370477,
                        0.6783106657060436
                    ]
