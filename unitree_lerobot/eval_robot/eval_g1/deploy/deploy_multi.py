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

from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller
from unitree_lerobot.eval_robot.eval_g1.image_server.image_client import ImageClient
from lerobot.common.utils.utils import init_logging

import deploy_multi_config as config

class CameraReceiver:
    """Camera interface for G1 head and wrist cameras"""
    def __init__(self, camera_config: Dict = None):
        # Camera config
        self.camera_config = camera_config
        self.frame_count = 0
        self.receiving = False
        
        # Head camera shape
        self.head_img_shape = (self.camera_config['head_camera_image_shape'][0], 
                               self.camera_config['head_camera_image_shape'][1], 3)
        
        # Wrist camera shape
        self.has_wrist = 'wrist_camera_type' in self.camera_config and self.camera_config['wrist_camera_type'] is not None
        if self.has_wrist:
            # For single wrist camera
            self.wrist_img_shape = (self.camera_config['wrist_camera_image_shape'][0],
                                   self.camera_config['wrist_camera_image_shape'][1], 3)
        
        # Initialize shared memory
        self.head_img_shm = None
        self.head_img_array = None
        self.wrist_img_shm = None
        self.wrist_img_array = None
        self.img_client = None
        self.image_receive_thread = None
        
        print(f"[INFO] Camera receiver initialized")
        print(f"  - Head camera shape: {self.head_img_shape}")
        if self.has_wrist:
            print(f"  - Wrist camera shape: {self.wrist_img_shape}")
        
    def start(self):
        """Start camera receiver"""
        try:
            # Create shared memory for head camera
            self.head_img_shm = shared_memory.SharedMemory(
                create=True, 
                size=np.prod(self.head_img_shape) * np.uint8().itemsize
            )
            self.head_img_array = np.ndarray(self.head_img_shape, dtype=np.uint8, 
                                           buffer=self.head_img_shm.buf)
            
            # Create shared memory for wrist camera
            if self.has_wrist:
                self.wrist_img_shm = shared_memory.SharedMemory(
                    create=True,
                    size=np.prod(self.wrist_img_shape) * np.uint8().itemsize
                )
                self.wrist_img_array = np.ndarray(self.wrist_img_shape, dtype=np.uint8,
                                                buffer=self.wrist_img_shm.buf)
            
            # Initialize image client with both cameras
            if self.has_wrist:
                self.img_client = ImageClient(
                    tv_img_shape=self.head_img_shape, 
                    tv_img_shm_name=self.head_img_shm.name,
                    wrist_img_shape=self.wrist_img_shape,
                    wrist_img_shm_name=self.wrist_img_shm.name
                )
            else:
                self.img_client = ImageClient(
                    tv_img_shape=self.head_img_shape, 
                    tv_img_shm_name=self.head_img_shm.name
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
            
    def get_head_frame(self):
        """Get current head camera frame"""
        if self.receiving and self.head_img_array is not None:
            return self.head_img_array.copy()
        return None
    
    def get_wrist_frame(self):
        """Get current wrist camera frame"""
        if self.receiving and self.has_wrist and self.wrist_img_array is not None:
            return self.wrist_img_array.copy()
        return None
    
    def stop(self):
        """Stop camera receiver"""
        print("[INFO] Stopping camera...")
        self.receiving = False
        
        # Clean up shared memory
        if self.head_img_shm is not None:
            self.head_img_shm.close()
            self.head_img_shm.unlink()
            
        if self.wrist_img_shm is not None:
            self.wrist_img_shm.close()
            self.wrist_img_shm.unlink()
            
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
        self.current_left_arm = [
                        -0.5737237982102243,
                        0.10833681482353297,
                        0.11715295640058615,
                        1.2739819628683753,
                        0.09773241798024906,
                        -0.8105767703215181,
                        0.10626869947361504
                    ]
        self.current_right_arm = [
                        -0.5779088152828018,
                        -0.2194551373009091,
                        -0.21519414603393777,
                        1.2949843913594083,
                        0.02554302428680151,
                        -0.7545965799635119,
                        0.0825105322549364
                    ]
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


def send_request(images: List[np.ndarray], instruction: str, server_url: str) -> np.ndarray:
    """
    Send images list to inference server
    """
    payload = {
        "image": images,  # List of numpy arrays
        "instruction": instruction
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(server_url, headers=headers, data=json_numpy.dumps(payload))
        
        if response.status_code != 200:
            raise Exception(f"Server error: {response.status_code} - {response.text}")
        
        result = response.json()
        action_chunk = np.array(result)
        
        # If single action, reshape to (1, 16)
        if action_chunk.ndim == 1:
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
    """Main control loop for deployment"""
    
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
    while camera_receiver.get_head_frame() is None:
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
            
            # Get camera frames
            head_image = camera_receiver.get_head_frame()
            if head_image is None:
                print(f"[WARN] No head camera frame at step {step}")
                continue
            
            # BRG -> RBG
            head_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB)
            
            # Build images list: [fix, head]
            images_list = []
            
            # Get wrist image if available
            if camera_receiver.has_wrist:
                wrist_image = camera_receiver.get_wrist_frame()
                if wrist_image is not None:
                    wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
                    images_list = [wrist_image, head_image]
                else:
                    print(f"[WARN] No wrist camera frame at step {step}, using head only")
                    images_list = [head_image]  # Head only
            else:
                images_list = [head_image]  # Head only
            
            # Increment frame count
            camera_receiver.frame_count += 1
            
            # Get action chunk from model
            try:
                inference_start = time.time()
                action_chunk = send_request(images_list, task_instruction, server_url)
                inference_time = time.time() - inference_start
                total_inference_count += 1
                
                if display_status:
                    print(f"\n[Inference {total_inference_count}] Time: {inference_time:.3f}s")
                    print(f"Received action chunk with {len(action_chunk)} actions")
                    print(f"  - Images sent: {len(images_list)} ({'wrist+head' if len(images_list) == 2 else 'head only'})")
                    for i, img in enumerate(images_list):
                        if len(images_list) == 2:
                            img_type = "wrist" if i == 0 else "head"
                        else:
                            img_type = "head"
                        print(f"  - {img_type} image shape: {img.shape}")
                
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
    
    print("[INFO] G1 Robot Deployment with Head and Wrist Cameras")
    print("="*60)
    print(f"Configuration loaded from: deploy_multi_config.py")
    print(f"  - Server URL: {config.SERVER_URL}")
    print(f"  - Task: {config.TASK_INSTRUCTION}")
    print(f"  - Control frequency: {config.CONTROL_FREQUENCY} Hz")
    print(f"  - Max steps: {config.MAX_STEPS}")
    print(f"  - Head camera FPS: {config.CAMERA_CONFIG['fps']}")
    
    # Check if wrist camera is configured
    if 'wrist_camera_type' in config.CAMERA_CONFIG:
        print(f"  - Wrist camera: Enabled")
        print(f"  - Wrist camera shape: {config.CAMERA_CONFIG['wrist_camera_image_shape']}")
        print(f"  - Image order: [wrist, head]")
    else:
        print(f"  - Wrist camera: Disabled")
        print(f"  - Image order: [head]")
    
    # Get chunk size from config
    chunk_size = getattr(config, 'CHUNK_SIZE', 1)
    print(f"  - Chunk size: {chunk_size}")
    print("="*60)
    
    # Confirmation
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