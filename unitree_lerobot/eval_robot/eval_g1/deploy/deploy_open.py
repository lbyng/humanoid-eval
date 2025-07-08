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
import cv2
import os

from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller
from lerobot.common.utils.utils import init_logging

import deploy_config as config


class ImageFolderReader:
    """Read images from a folder instead of camera"""
    def __init__(self, image_folder: str):
        self.image_folder = Path(image_folder)
        self.frame_count = 0
        self.current_index = 0
        
        # Get all image files sorted by name
        self.image_files = sorted([
            f for f in self.image_folder.glob("*.png")
            if f.stem.isdigit()
        ], key=lambda x: int(x.stem))
        
        if not self.image_files:
            self.image_files = sorted(self.image_folder.glob("*.jpg"))
        
        self.total_images = len(self.image_files)
        
        print(f"[INFO] Image folder reader initialized")
        print(f"  - Folder: {self.image_folder}")
        print(f"  - Total images: {self.total_images}")
        if self.total_images > 0:
            print(f"  - First image: {self.image_files[0].name}")
            print(f"  - Last image: {self.image_files[-1].name}")
        
    def start(self):
        """Compatibility method - does nothing for folder reader"""
        if self.total_images == 0:
            print("[ERROR] No images found in folder!")
            return
        print("[INFO] Image folder reader ready")
        
    def get_frame(self):
        """Get next image from folder"""
        if self.current_index >= self.total_images:
            print(f"[WARN] Reached end of images (index {self.current_index} >= {self.total_images})")
            return None
            
        image_path = self.image_files[self.current_index]
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[ERROR] Failed to read image: {image_path}")
            return None
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.frame_count += 8
        self.current_index += 8
        
        return image
    
    def reset(self):
        """Reset to first image"""
        self.current_index = 0
        print("[INFO] Reset to first image")
        
    def stop(self):
        """Compatibility method - does nothing for folder reader"""
        print(f"[INFO] Image folder reader stopped. Total frames read: {self.frame_count}")


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


def send_request(image_array: np.ndarray, instruction: str, server_url: str, timeout: float = 5.0) -> np.ndarray:
    """
    Send image and instruction to inference server using json_numpy
    Returns action chunk as numpy array
    """
    payload = {
        "image": image_array,  # Keep as numpy array, json_numpy will handle it
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
        
    except requests.exceptions.Timeout:
        raise Exception("Request timeout")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to server")


def run_offline_control(
    server_url: str,
    task_instruction: str,
    image_folder: str,
    frequency: float = 50.0,
    max_steps: int = 1000,
    display_status: bool = True,
    chunk_size: int = 1,
    display_images: bool = False
):
    """Main control loop for offline deployment with saved images"""
    
    # Config
    step_timeout = getattr(config, 'STEP_TIMEOUT', 5.0)
    print_freq_every = getattr(config, 'PRINT_FREQUENCY_EVERY_N_STEPS', 10)
    wait_for_gripper = getattr(config, 'WAIT_FOR_GRIPPER', True)
    
    print("[INFO] Starting G1 offline control with saved images")
    print(f"  - Server: {server_url}")
    print(f"  - Task: {task_instruction}")
    print(f"  - Image folder: {image_folder}")
    print(f"  - Frequency: {frequency} Hz")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Chunk size: {chunk_size}")
    print("="*60)
    
    # Initialize image reader
    image_reader = ImageFolderReader(image_folder)
    image_reader.start()
    
    if image_reader.total_images == 0:
        print("[ERROR] No images found in folder")
        return
    
    # Initialize robot
    robot_controller = G1DeployController(frequency=frequency)
    robot_controller.init()
    
    if not robot_controller.initialized:
        print("[ERROR] Failed to initialize robot controller")
        return
    
    # Main control loop
    print(f"\n[INFO] Starting control loop...")
    print("Press Ctrl+C to stop\n")
    
    if display_images:
        cv2.namedWindow("Current Image", cv2.WINDOW_NORMAL)
    
    step = 0
    total_inference_count = 0
    
    try:
        while step < max_steps and step < image_reader.total_images:
            loop_start = time.time()
            
            # Get next image from folder
            current_image = image_reader.get_frame()
            if current_image is None:
                print(f"[INFO] No more images available")
                break
            
            # Display image if requested
            if display_images:
                cv2.imshow("Current Image", cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            
            # Get action chunk from model
            try:
                inference_start = time.time()
                action_chunk = send_request(current_image, task_instruction, server_url)
                inference_time = time.time() - inference_start
                total_inference_count += 1
                
                if display_status:
                    print(f"\n[Inference {total_inference_count}] Image: {image_reader.image_files[image_reader.current_index-1].name}")
                    print(f"Inference time: {inference_time:.3f}s")
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
        image_reader.stop()
        if display_images:
            cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("[INFO] Control finished")
    print(f"[INFO] Total steps: {step}")
    print(f"[INFO] Total inferences: {total_inference_count}")
    print(f"[INFO] Images processed: {image_reader.frame_count}/{image_reader.total_images}")


def main():
    # Initialize logging
    init_logging()
    
    print("[INFO] G1 Robot Offline Deployment")
    print("="*60)
    print(f"Configuration loaded from: deploy_config.py")
    print(f"  - Server URL: {config.SERVER_URL}")
    print(f"  - Task: {config.TASK_INSTRUCTION}")
    print(f"  - Control frequency: {config.CONTROL_FREQUENCY} Hz")
    print(f"  - Max steps: {config.MAX_STEPS}")
    
    # Get chunk size from config if available
    chunk_size = getattr(config, 'CHUNK_SIZE', 1)
    print(f"  - Chunk size: {chunk_size}")
    
    # Get image folder path
    image_folder = getattr(config, 'IMAGE_FOLDER', None)
    if image_folder is None:
        image_folder = input("\nEnter path to image folder: ").strip()
    else:
        print(f"  - Image folder: {image_folder}")
    
    # Check if folder exists
    if not os.path.exists(image_folder):
        print(f"[ERROR] Image folder does not exist: {image_folder}")
        return
    
    # Display images option
    display_images = getattr(config, 'DISPLAY_IMAGES', False)
    print(f"  - Display images: {display_images}")
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
    
    # Run offline control loop
    run_offline_control(
        server_url=config.SERVER_URL,
        task_instruction=config.TASK_INSTRUCTION,
        image_folder=image_folder,
        frequency=config.CONTROL_FREQUENCY,
        max_steps=config.MAX_STEPS,
        display_status=config.DISPLAY_STATUS,
        chunk_size=chunk_size,
        display_images=display_images
    )


if __name__ == "__main__":
    main()