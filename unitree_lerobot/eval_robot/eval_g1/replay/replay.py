import json
import time
import numpy as np
import threading
from multiprocessing import Array, Lock
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional

from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from lerobot.common.utils.utils import init_logging

class RobotReplayController:
    def __init__(self, robot_config: dict, frequency: float = 50.0):
        self.robot_config = robot_config
        self.frequency = frequency
        self.arm_ctrl = None
        self.hand_ctrl = None
        self.gripper_ctrl = None
        
        # Initialize controllers
        self._init_controllers()
        
    def _init_controllers(self):
        """Initialize robot controllers"""
        # Initialize arm controller
        self.arm_ctrl = G1_29_ArmController()
        
        # Initialize hand controllers based on configuration
        if self.robot_config['hand_type'] == "dex3":
            self.left_hand_array = Array('d', 7, lock=True)
            self.right_hand_array = Array('d', 7, lock=True)
            self.dual_hand_data_lock = Lock()
            self.dual_hand_state_array = Array('d', 14, lock=False)
            self.dual_hand_action_array = Array('d', 14, lock=False)
            self.hand_ctrl = Dex3_1_Controller(
                self.left_hand_array, 
                self.right_hand_array, 
                self.dual_hand_data_lock, 
                self.dual_hand_state_array, 
                self.dual_hand_action_array
            )
            
        elif self.robot_config['hand_type'] == "gripper":
            self.left_hand_array = Array('d', 1, lock=True)
            self.right_hand_array = Array('d', 1, lock=True)
            self.dual_gripper_data_lock = Lock()
            self.dual_gripper_state_array = Array('d', 2, lock=False)
            self.dual_gripper_action_array = Array('d', 2, lock=False)
            self.gripper_ctrl = Gripper_Controller(
                self.left_hand_array, 
                self.right_hand_array, 
                self.dual_gripper_data_lock, 
                self.dual_gripper_state_array, 
                self.dual_gripper_action_array
            )
    
    def set_initial_pose(self, initial_states: dict):
        """Set robot to initial pose"""
        print("Setting robot to initial pose...")
        
        # Extract initial poses
        left_arm_pose = np.array(initial_states['left_arm']['qpos'])
        right_arm_pose = np.array(initial_states['right_arm']['qpos'])
        dual_arm_pose = np.concatenate([left_arm_pose, right_arm_pose])
        
        # Set arm pose
        self.arm_ctrl.ctrl_dual_arm(dual_arm_pose, np.zeros(14))
        
        # Set hand pose
        if self.robot_config['hand_type'] == "dex3":
            left_hand_pose = np.array(initial_states['left_hand']['qpos'])
            right_hand_pose = np.array(initial_states['right_hand']['qpos'])
            self.left_hand_array[:] = left_hand_pose
            self.right_hand_array[:] = right_hand_pose
            
        elif self.robot_config['hand_type'] == "gripper":
            left_hand_pose = initial_states['left_hand']['qpos'][0]  # Single value for gripper
            right_hand_pose = initial_states['right_hand']['qpos'][0]
            self.left_hand_array[:] = [left_hand_pose]
            self.right_hand_array[:] = [right_hand_pose]
        
        print("Initial pose set. Waiting for robot to stabilize...")
        time.sleep(2.0)
    
    def execute_action(self, action_data: dict):
        """Execute a single action"""
        # Extract arm actions
        left_arm_action = np.array(action_data['left_arm']['qpos'])
        right_arm_action = np.array(action_data['right_arm']['qpos'])
        dual_arm_action = np.concatenate([left_arm_action, right_arm_action])
        
        # Execute arm action
        self.arm_ctrl.ctrl_dual_arm(dual_arm_action, np.zeros(14))
        
        # Execute hand action
        if self.robot_config['hand_type'] == "dex3":
            left_hand_action = np.array(action_data['left_hand']['qpos'])
            right_hand_action = np.array(action_data['right_hand']['qpos'])
            self.left_hand_array[:] = left_hand_action
            self.right_hand_array[:] = right_hand_action
            
        elif self.robot_config['hand_type'] == "gripper":
            left_hand_action = action_data['left_hand']['qpos'][0]
            right_hand_action = action_data['right_hand']['qpos'][0]
            self.left_hand_array[:] = [left_hand_action]
            self.right_hand_array[:] = [right_hand_action]


def load_replay_data(json_file_path: str) -> dict:
    """Load replay data from JSON file"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {json_file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in file: {json_file_path}")
        raise


def replay_actions(
    replay_data: dict, 
    robot_config: dict, 
    frequency: float = 50.0,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    loop: bool = False
):
    """Replay robot actions from recorded data"""
    
    # Initialize robot controller
    controller = RobotReplayController(robot_config, frequency)
    
    # Get data sequence
    data_sequence = replay_data['data']
    if end_idx is None:
        end_idx = len(data_sequence)
    
    # Validate indices
    if start_idx >= len(data_sequence):
        logging.error(f"start_idx ({start_idx}) is greater than data length ({len(data_sequence)})")
        return
    
    if end_idx > len(data_sequence):
        end_idx = len(data_sequence)
        logging.warning(f"end_idx adjusted to data length: {end_idx}")
    
    # Display task information
    if 'text' in replay_data:
        print("=== Task Information ===")
        print(f"Goal: {replay_data['text'].get('goal', 'N/A')}")
        print(f"Description: {replay_data['text'].get('desc', 'N/A')}")
        print(f"Steps: {replay_data['text'].get('steps', 'N/A')}")
        print("========================")
    
    # Wait for user confirmation
    user_input = input("Please enter 's' to start replay (or 'q' to quit): ")
    if user_input.lower() == 'q':
        print("Replay cancelled.")
        return
    elif user_input.lower() != 's':
        print("Invalid input. Please restart and enter 's' to start.")
        return
    
    # Set initial pose
    initial_states = data_sequence[start_idx]['states']
    controller.set_initial_pose(initial_states)
    
    print(f"Starting replay from index {start_idx} to {end_idx-1}")
    print(f"Total actions to replay: {end_idx - start_idx}")
    print(f"Frequency: {frequency} Hz")
    print("Press Ctrl+C to stop replay\n")
    
    try:
        replay_count = 0
        while True:
            replay_count += 1
            print(f"=== Replay #{replay_count} ===")
            
            for i in range(start_idx, end_idx):
                action_data = data_sequence[i]['actions']
                
                # Skip if all actions are zero (common for first frame)
                if i == start_idx and all(
                    all(val == 0.0 for val in action_data[key]['qpos']) 
                    for key in action_data.keys() if action_data[key] is not None
                ):
                    print(f"Skipping frame {i} (zero actions)")
                    continue
                
                print(f"Executing action {i}/{end_idx-1}", end='\r')
                
                # Execute action
                controller.execute_action(action_data)
                
                # Sleep to maintain frequency
                time.sleep(1.0 / frequency)
            
            print(f"\nReplay #{replay_count} completed!")
            
            if not loop:
                break
            
            # Ask user if they want to continue
            user_input = input("Press 's' to replay again, 'q' to quit: ")
            if user_input.lower() == 'q':
                break
            elif user_input.lower() != 's':
                print("Invalid input. Stopping replay.")
                break
                
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    except Exception as e:
        logging.error(f"Error during replay: {e}")
        raise
    
    print("Replay finished.")


def main():
    parser = argparse.ArgumentParser(description="Replay robot actions from recorded data")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing replay data")
    parser.add_argument("--hand_type", type=str, default="dex3", choices=["dex3", "gripper"], 
                       help="Type of hand controller (default: dex3)")
    parser.add_argument("--frequency", type=float, default=30.0, 
                       help="Replay frequency in Hz (default: 50.0)")
    parser.add_argument("--start_idx", type=int, default=0, 
                       help="Starting index for replay (default: 0)")
    parser.add_argument("--end_idx", type=int, default=None, 
                       help="Ending index for replay (default: None, replay all)")
    parser.add_argument("--loop", action="store_true", 
                       help="Loop replay continuously")
    parser.add_argument("--arm_type", type=str, default="g1", 
                       help="Type of arm controller (default: g1)")
    
    args = parser.parse_args()
    
    # Initialize logging
    init_logging()
    
    # Robot configuration
    robot_config = {
        'arm_type': args.arm_type,
        'hand_type': args.hand_type,
    }
    
    # Load replay data
    print(f"Loading replay data from: {args.json_file}")
    replay_data = load_replay_data(args.json_file)
    
    print(f"Loaded {len(replay_data['data'])} frames")
    
    # Start replay
    replay_actions(
        replay_data=replay_data,
        robot_config=robot_config,
        frequency=args.frequency,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        loop=args.loop
    )


if __name__ == "__main__":
    main()