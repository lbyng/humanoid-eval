import time
import numpy as np
import requests
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Array, Lock, shared_memory
import json_numpy
json_numpy.patch()

from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller
from unitree_lerobot.eval_robot.eval_g1.image_server.image_client import ImageClient
from lerobot.common.utils.utils import init_logging

import deploy_config as config

class CameraReceiver:
    """Camera interface for G1 head camera"""
    def __init__(self, camera_config: Dict = None):
        # Camera config
        self.camera_config = camera_config
        self.frame_count = 0
        self.receiving = False
        
        # Image shape
        self.img_shape = (self.camera_config['head_camera_image_shape'][0], 
                         self.camera_config['head_camera_image_shape'][1], 3)
        
        # Initialize shared memory
        self.img_shm = None
        self.img_array = None
        self.img_client = None
        self.image_receive_thread = None
        
        print(f"[INFO] Camera receiver initialized - Shape: {self.img_shape}")
        
    def start(self):
        """Start camera receiver"""
        try:
            # Create shared memory
            self.img_shm = shared_memory.SharedMemory(
                create=True, 
                size=np.prod(self.img_shape) * np.uint8().itemsize
            )
            self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img_shm.buf)
            
            # Initialize image client
            self.img_client = ImageClient(
                tv_img_shape=self.img_shape, 
                tv_img_shm_name=self.img_shm.name
            )
            
            # Start image receive thread
            self.image_receive_thread = threading.Thread(
                target=self.img_client.receive_process, 
                daemon=True
            )
            self.image_receive_thread.daemon = True
            self.image_receive_thread.start()
            
            self.receiving = True
            print("[INFO] Camera started successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to start camera: {e}")
            self.receiving = False
            
    def get_frame(self):
        """Get current camera frame"""
        if self.receiving and self.img_array is not None:
            self.frame_count += 1
            return self.img_array.copy()
        return None
    
    def stop(self):
        """Stop camera receiver"""
        print("[INFO] Stopping camera...")
        self.receiving = False
        
        # Clean up shared memory
        if self.img_shm is not None:
            self.img_shm.close()
            self.img_shm.unlink()
            
        print(f"[INFO] Camera stopped. Total frames: {self.frame_count}")


class G1DeployController:
    """G1 robot controller for deployment with Dex3 hands"""
    def __init__(self, frequency: float = 50.0):
        self.frequency = frequency
        self.arm_ctrl = None
        self.hand_ctrl = None
        
        # Track current arm positions
        self.current_left_arm = np.zeros(7)
        self.current_right_arm = np.zeros(7)
        
        # Track current hand states
        self.prev_left_hand_state = 0
        self.prev_right_hand_state = 0
        
        # Hand arrays for Dex3
        self.left_hand_array = Array('d', 7, lock=True)
        self.right_hand_array = Array('d', 7, lock=True)
        self.dual_hand_data_lock = Lock()
        self.dual_hand_state_array = Array('d', 14, lock=False)
        self.dual_hand_action_array = Array('d', 14, lock=False)
        
        self.initialized = False
        
    def init(self):
        """Initialize robot controllers"""
        try:
            # Initialize arm controller
            print("[INFO] Initializing G1-29 arm controller...")
            self.arm_ctrl = G1_29_ArmController()
            
            # Initialize Dex3 hand controller
            print("[INFO] Initializing Dex3 hand controller...")
            self.hand_ctrl = Dex3_1_Controller(
                self.left_hand_array, 
                self.right_hand_array, 
                self.dual_hand_data_lock, 
                self.dual_hand_state_array, 
                self.dual_hand_action_array
            )
            
            # Set initial pose
            self.set_initial_pose()
            
            self.initialized = True
            print("[INFO] Robot controller initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize robot: {e}")
            self.initialized = False
            
    def set_initial_pose(self):
        """Set robot to initial pose (zero position)"""
        print("[INFO] Setting robot to initial pose...")
        
        # Set arms to zero position
        self.current_left_arm = np.zeros(7)
        self.current_right_arm = np.zeros(7)
        dual_arm_pose = np.concatenate([self.current_left_arm, self.current_right_arm])
        
        # Set arm pose
        self.arm_ctrl.ctrl_dual_arm(dual_arm_pose, np.zeros(14))
        
        # Set hands to open position
        self.left_hand_array[:] = np.zeros(7)
        self.right_hand_array[:] = np.zeros(7)
        self.prev_left_hand_state = 0
        self.prev_right_hand_state = 0
        
        print("[INFO] Initial pose set. Waiting for robot to stabilize...")
        time.sleep(2.0)
        
    def execute_action(self, action: List[float], wait_for_gripper: bool = True):
        """
        Execute 16-dimensional action from model
        action[0:7] - left arm joint deltas
        action[7:14] - right arm joint deltas  
        action[14] - left hand state (0=open, 1=closed)
        action[15] - right hand state (0=open, 1=closed)
        """
        if not self.initialized:
            print("[WARN] Robot not initialized, skipping action")
            return
            
        try:
            # Parse action
            left_arm_delta = np.array(action[0:7])
            right_arm_delta = np.array(action[7:14])
            left_hand_state = action[14]  # 0=open, 1=closed
            right_hand_state = action[15]  # 0=open, 1=closed
            
            # Apply delta to current positions
            self.current_left_arm += left_arm_delta
            self.current_right_arm += right_arm_delta
            
            # Execute arm action
            dual_arm_action = np.concatenate([self.current_left_arm, self.current_right_arm])
            self.arm_ctrl.ctrl_dual_arm(dual_arm_action, np.zeros(14))
            
            # Execute hand action
            left_hand_pose = self._get_dex3_hand_pose(left_hand_state, 'left')
            right_hand_pose = self._get_dex3_hand_pose(right_hand_state, 'right')
            self.left_hand_array[:] = left_hand_pose
            self.right_hand_array[:] = right_hand_pose
            
            # Check if gripper state changed and wait if needed
            if wait_for_gripper:
                gripper_changed = (self.prev_left_hand_state != left_hand_state or 
                                 self.prev_right_hand_state != right_hand_state)
                if gripper_changed:
                    print(f"[INFO] Gripper state changed, waiting 2s...")
                    time.sleep(2.0)
                    self.prev_left_hand_state = left_hand_state
                    self.prev_right_hand_state = right_hand_state
            
        except Exception as e:
            print(f"[ERROR] Failed to execute action: {e}")
            
    def _get_dex3_hand_pose(self, state: float, hand_type: str) -> np.ndarray:
        """Convert binary state to dex3 hand pose"""
        if state < 0.5:  # Open
            return np.zeros(7)
        else:  # Closed
            if hand_type == 'left':
                return np.array([0, 1.05, 1.75, -1.57, -1.75, -1.57, -1.75])
            else:
                return np.array([0, -1.05, -1.75, 1.57, 1.75, 1.57, 1.75])
                
    def get_status(self) -> Dict:
        """Get current robot status"""
        return {
            'left_arm_pos': self.current_left_arm.tolist(),
            'right_arm_pos': self.current_right_arm.tolist(),
            'left_hand_state': 'closed' if np.any(self.left_hand_array[:]) else 'open',
            'right_hand_state': 'closed' if np.any(self.right_hand_array[:]) else 'open'
        }
        
    def reset_position(self):
        """Reset to initial position"""
        self.set_initial_pose()
        
    def cleanup(self):
        """Cleanup resources"""
        print("[INFO] Robot controller cleaned up")


def send_request(image_array: np.ndarray, instruction: str, server_url: str) -> np.ndarray:
    """
    Send image and instruction to inference server using json_numpy
    Returns action chunk as numpy array
    """
    payload = {
        "image": image_array,
        "instruction": instruction
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(server_url, headers=headers, data=json_numpy.dumps(payload))
        
        if response.status_code != 200:
            raise Exception(f"Server error: {response.status_code} - {response.text}")
        
        result = response.json()
        action_chunk = np.array(result)
        
        # Validate action chunk shape
        if action_chunk.ndim == 1:
            # Single action, reshape to (1, 16)
            action_chunk = action_chunk.reshape(1, -1)
        
        if action_chunk.shape[-1] != 16:
            raise ValueError(f"Expected 16-dimensional actions, got shape: {action_chunk.shape}")
        
        return action_chunk
        
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to server")


def run_closed_loop_control(
    server_url: str,
    task_instruction: str,
    frequency: float = 50.0,
    max_steps: int = 1000,
    display_status: bool = True,
    camera_config: Dict = None,
    chunk_size: int = 1
):
    """Main control loop for deployment with action chunking"""
    
    # Config
    camera_timeout = getattr(config, 'CAMERA_TIMEOUT', 10)
    step_timeout = getattr(config, 'STEP_TIMEOUT', 5.0)
    print_freq_every = getattr(config, 'PRINT_FREQUENCY_EVERY_N_STEPS', 10)
    wait_for_gripper = getattr(config, 'WAIT_FOR_GRIPPER', True)
    
    print("[INFO] Starting G1 closed-loop control with action chunking")
    print(f"  - Server: {server_url}")
    print(f"  - Task: {task_instruction}")
    print(f"  - Frequency: {frequency} Hz")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Chunk size: {chunk_size}")
    print("="*60)
    
    # Initialize camera
    camera_receiver = CameraReceiver(camera_config)
    camera_receiver.start()
    
    # Initialize robot
    robot_controller = G1DeployController(frequency=frequency)
    robot_controller.init()
    
    if not robot_controller.initialized:
        print("[ERROR] Failed to initialize robot controller")
        camera_receiver.stop()
        return
    
    # Wait for first frame
    print("[INFO] Waiting for camera...")
    wait_start = time.time()
    while camera_receiver.get_frame() is None:
        if time.time() - wait_start > camera_timeout:
            print(f"[ERROR] Camera timeout ({camera_timeout}s)")
            camera_receiver.stop()
            robot_controller.cleanup()
            return
        time.sleep(0.1)
    print("[INFO] Camera ready!")
    
    # Main control loop
    print(f"\n[INFO] Starting control loop...")
    print("Press Ctrl+C to stop\n")
    
    step = 0
    total_inference_count = 0
    
    try:
        while step < max_steps:
            loop_start = time.time()
            
            # Get camera frame
            current_image = camera_receiver.get_frame()
            if current_image is None:
                print(f"[WARN] No camera frame at step {step}")
                continue
            
            import cv2
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            
            # Get action chunk from model
            try:
                inference_start = time.time()
                action_chunk = send_request(current_image, task_instruction, server_url)
                inference_time = time.time() - inference_start
                total_inference_count += 1
            
                # # 只保留前4个动作
                # original_chunk_size = len(action_chunk)
                # action_chunk = action_chunk[:4]
                    
                if display_status:
                    print(f"\n[Inference {total_inference_count}] Time: {inference_time:.3f}s")
                    print(f"Received action chunk with {len(action_chunk)} actions")
                
                # Execute each action in the chunk
                for chunk_idx, action in enumerate(action_chunk):
                    if step >= max_steps:
                        break
                        
                    action_start = time.time()
                    
                    if display_status:
                        print(f"\nStep {step + 1}/{max_steps} (Chunk action {chunk_idx + 1}/{len(action_chunk)})")
                        print(f"Action:")
                        print(f"  - Left arm deltas: {np.array(action[:7]).round(3)}")
                        print(f"  - Right arm deltas: {np.array(action[7:14]).round(3)}")
                        print(f"  - Left hand: {'close' if action[14] >= 0.5 else 'open'}")
                        print(f"  - Right hand: {'close' if action[15] >= 0.5 else 'open'}")
                    
                    # Execute action
                    robot_controller.execute_action(action, wait_for_gripper=wait_for_gripper)
                    
                    if display_status and step % print_freq_every == 0:
                        status = robot_controller.get_status()
                        print(f"Current position:")
                        print(f"  - Left arm: {np.array(status['left_arm_pos']).round(2)}")
                        print(f"  - Right arm: {np.array(status['right_arm_pos']).round(2)}")
                        print(f"  - Left hand: {status['left_hand_state']}")
                        print(f"  - Right hand: {status['right_hand_state']}")
                    
                    # Maintain frequency
                    action_time = time.time() - action_start
                    sleep_time = max(0, (1.0 / frequency) - action_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    if display_status and step % print_freq_every == 0:
                        actual_freq = 1.0 / (time.time() - action_start)
                        print(f"  - Actual frequency: {actual_freq:.1f} Hz")
                    
                    step += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to get/execute action chunk: {e}")
                continue
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[INFO] Cleaning up...")
        robot_controller.cleanup()
        camera_receiver.stop()
    
    print("\n" + "="*60)
    print("[INFO] Control finished")
    print(f"[INFO] Total steps: {step}")
    print(f"[INFO] Total inferences: {total_inference_count}")
    print(f"[INFO] Camera frames: {camera_receiver.frame_count}")


def main():
    # Initialize logging
    init_logging()
    
    print("[INFO] G1 Robot Deployment with Action Chunking")
    print("="*60)
    print(f"Configuration loaded from: deploy_config.py")
    print(f"  - Server URL: {config.SERVER_URL}")
    print(f"  - Task: {config.TASK_INSTRUCTION}")
    print(f"  - Control frequency: {config.CONTROL_FREQUENCY} Hz")
    print(f"  - Max steps: {config.MAX_STEPS}")
    print(f"  - Camera FPS: {config.CAMERA_CONFIG['fps']}")
    
    # Get chunk size from config if available
    chunk_size = getattr(config, 'CHUNK_SIZE', 1)
    print(f"  - Chunk size: {chunk_size}")
    print("="*60)
    
    # Ask for user confirmation
    user_input = input("\nPress 's' to start deployment, 'r' to reset robot, or 'q' to quit: ")
    
    if user_input.lower() == 'q':
        print("Deployment cancelled.")
        return
    elif user_input.lower() == 'r':
        robot = G1DeployController()
        robot.init()
        robot.reset_position()
        print("[INFO] Robot reset to initial position")
        robot.cleanup()
        return
    elif user_input.lower() != 's':
        print("Invalid input. Exiting.")
        return
    
    # Run control loop
    run_closed_loop_control(
        server_url=config.SERVER_URL,
        task_instruction=config.TASK_INSTRUCTION,
        frequency=config.CONTROL_FREQUENCY,
        max_steps=config.MAX_STEPS,
        display_status=config.DISPLAY_STATUS,
        camera_config=config.CAMERA_CONFIG,
        chunk_size=chunk_size
    )

if __name__ == "__main__":
    main()