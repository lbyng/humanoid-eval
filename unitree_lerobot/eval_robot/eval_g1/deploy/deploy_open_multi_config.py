# deploy_config.py for offline deployment

# Server configuration
SERVER_URL = "https://ca179e8e1acc.ngrok-free.app/act"
TASK_INSTRUCTION = "clean plate"
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
HEAD_FOLDER = "/home/sean/humanoid-teleop/data/converted_data/clean_plate/head/0005"
WRIST_FOLDER = "/home/sean/humanoid-teleop/data/converted_data/clean_plate/fix/0005"

LEFT_ARM = [
            -0.7814386519874144,
            0.26664621835562186,
            -0.07327171430414153,
            1.2399993242358711,
            0.29410360768561616,
            -0.5346406380603802,
            0.05183655009729242
        ]

RIGHT_AMR = [
            -0.7188564620423048,
            -0.22946260222567763,
            -0.26948017917637934,
            1.199751473983983,
            0.13455140941625637,
            -0.5741487435915177,
            0.19707843870303807
        ]
