import json
import time
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from multiprocessing import Array, Lock

from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
from lerobot.common.utils.utils import init_logging


class ConvertedDataReplayController:
    def __init__(self, robot_config: dict, frequency: float = 50.0):
        self.robot_config = robot_config
        self.frequency = frequency
        self.arm_ctrl = None
        self.hand_ctrl = None
        self.gripper_ctrl = None
        
        # Track current arm positions for delta application
        self.current_left_arm = np.zeros(7)
        self.current_right_arm = np.zeros(7)
        
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
    
    def set_initial_pose(self):
        """Set robot to initial pose (zero position)"""
        print("Setting robot to initial pose...")
        
        # Set arms to zero position
        self.current_left_arm = np.zeros(7)
        self.current_right_arm = np.zeros(7)
        dual_arm_pose = np.concatenate([self.current_left_arm, self.current_right_arm])
        
        # Set arm pose
        self.arm_ctrl.ctrl_dual_arm(dual_arm_pose, np.zeros(14))
        
        # Set hands to open position
        if self.robot_config['hand_type'] == "dex3":
            self.left_hand_array[:] = np.zeros(7)
            self.right_hand_array[:] = np.zeros(7)
        elif self.robot_config['hand_type'] == "gripper":
            self.left_hand_array[:] = [0.0]
            self.right_hand_array[:] = [0.0]
        
        print("Initial pose set. Waiting for robot to stabilize...")
        time.sleep(2.0)
    
    def execute_action(self, raw_action: List[float]):
        """Execute action from converted data format"""
        # Parse action: left_arm(7) + right_arm(7) + left_hand(1) + right_hand(1)
        left_arm_delta = np.array(raw_action[0:7])
        right_arm_delta = np.array(raw_action[7:14])
        left_hand_state = raw_action[14]  # 0=open, 1=closed
        right_hand_state = raw_action[15]  # 0=open, 1=closed
        
        # Apply delta to current positions
        self.current_left_arm += left_arm_delta
        self.current_right_arm += right_arm_delta
        
        # Execute arm action
        dual_arm_action = np.concatenate([self.current_left_arm, self.current_right_arm])
        self.arm_ctrl.ctrl_dual_arm(dual_arm_action, np.zeros(14))
        
        # Execute hand action
        if self.robot_config['hand_type'] == "dex3":
            # Convert binary state to hand positions
            left_hand_pose = self._get_dex3_hand_pose(left_hand_state, 'left')
            right_hand_pose = self._get_dex3_hand_pose(right_hand_state, 'right')
            self.left_hand_array[:] = left_hand_pose
            self.right_hand_array[:] = right_hand_pose
            
        elif self.robot_config['hand_type'] == "gripper":
            # For gripper, use state directly
            self.left_hand_array[:] = [left_hand_state]
            self.right_hand_array[:] = [right_hand_state]
    
    def _get_dex3_hand_pose(self, state: float, hand_type: str) -> np.ndarray:
        """Convert binary state to dex3 hand pose"""
        if state == 0:  # Open
            return np.zeros(7)
        else:  # Closed
            if hand_type == 'left':
                return np.array([0, 1.05, 1.75, -1.57, -1.75, -1.57, -1.75])
            else:
                return np.array([0, -1.05 , -1.75, 1.57, 1.75, 1.57, 1.75])


def load_converted_data(json_file_path: str) -> dict:
    """Load converted data from JSON file"""
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


def group_episodes(data: List[dict]) -> Dict[str, List[dict]]:
    """Group data by episode ID"""
    episodes = {}
    for item in data:
        # Extract episode ID from image path (e.g., "0001_rgb/0001/0000.png" -> "0001")
        image_path = item['image']
        episode_id = image_path.split('_')[0]
        
        if episode_id not in episodes:
            episodes[episode_id] = []
        episodes[episode_id].append(item)
    
    # Sort items within each episode by image filename
    for episode_id in episodes:
        episodes[episode_id].sort(key=lambda x: x['image'].split('/')[-1])
    
    return episodes


def replay_episode(
    episode_data: List[dict],
    robot_config: dict,
    frequency: float = 50.0
):
    """Replay a single episode"""
    
    # Initialize robot controller
    controller = ConvertedDataReplayController(robot_config, frequency)
    
    # Display episode information
    print(f"\n=== Episode Information ===")
    print(f"Task: {episode_data[0]['task']}")
    print(f"Total frames: {len(episode_data)}")
    print(f"Frequency: {frequency} Hz")
    print("===========================\n")
    
    # Set initial pose
    controller.set_initial_pose()
    
    print("Starting replay...")
    print("Press Ctrl+C to stop replay\n")
    
    try:
        for i, item in enumerate(episode_data):
            # Parse raw action
            raw_action = json.loads(item['raw_action'])
            
            # Skip if all actions are zero (common for first frame)
            if i == 0 and all(val == 0.0 for val in raw_action):
                print(f"Skipping frame {i} (zero actions)")
                continue
            
            print(f"Executing frame {i}/{len(episode_data)-1}", end='\r')
            
            # Execute action
            controller.execute_action(raw_action)
            
            # Sleep to maintain frequency
            time.sleep(1.0 / frequency)
        
        print(f"\nReplay completed!")
        
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    except Exception as e:
        logging.error(f"Error during replay: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Replay robot actions from converted data")
    parser.add_argument("json_file", type=str, help="Path to the converted dataset JSON file")
    parser.add_argument("--hand_type", type=str, default="dex3", choices=["dex3", "gripper"], 
                       help="Type of hand controller (default: dex3)")
    parser.add_argument("--frequency", type=float, default=30.0, 
                       help="Replay frequency in Hz (default: 30.0)")
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
    
    # Load converted data
    print(f"Loading converted data from: {args.json_file}")
    data = load_converted_data(args.json_file)
    
    # Group data by episodes
    episodes = group_episodes(data)
    episode_ids = sorted(episodes.keys())
    
    print(f"\nFound {len(episodes)} episodes:")
    for i, episode_id in enumerate(episode_ids):
        episode = episodes[episode_id]
        print(f"  {i+1}. Episode {episode_id} - {len(episode)} frames - Task: {episode[0]['task']}")
    
    while True:
        # Let user select episode
        try:
            choice = input("\nEnter episode number to replay (or 'q' to quit): ")
            if choice.lower() == 'q':
                print("Exiting...")
                break
            
            episode_idx = int(choice) - 1
            if 0 <= episode_idx < len(episode_ids):
                selected_episode_id = episode_ids[episode_idx]
                selected_episode = episodes[selected_episode_id]
                
                print(f"\nSelected Episode {selected_episode_id}")
                
                # Replay the episode
                replay_episode(
                    episode_data=selected_episode,
                    robot_config=robot_config,
                    frequency=args.frequency
                )
                
                # Ask if user wants to replay another episode
                again = input("\nReplay another episode? (y/n): ")
                if again.lower() != 'y':
                    break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(episode_ids)}")
                
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except Exception as e:
            logging.error(f"Error: {e}")
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()