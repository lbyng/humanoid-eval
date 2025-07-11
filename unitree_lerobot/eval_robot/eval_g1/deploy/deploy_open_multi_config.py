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
HEAD_FOLDER = "/home/sean/humanoid-teleop/data/converted_data/close_box/head/0002"
WRIST_FOLDER = "/home/sean/humanoid-teleop/data/converted_data/close_box/fix/0002"

LEFT_ARM = [
                        -0.7310029517388426,
                        0.282537611070542,
                        0.294296086304665,
                        1.2364572987833264,
                        -0.15890418330143624,
                        -0.5808883844587759,
                        -0.448898914469791
                    ]
RIGHT_AMR = [
                        -0.9151563465485182,
                        -0.2626081436879704,
                        -0.3158276106903784,
                        1.302297506571812,
                        0.12354234668691412,
                        -0.5351948051081976,
                        0.29942777685528993
                    ]
