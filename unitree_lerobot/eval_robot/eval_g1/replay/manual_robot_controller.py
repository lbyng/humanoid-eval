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
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller
from lerobot.common.utils.utils import init_logging


class ManualRobotController:
    def __init__(self, robot_config: dict):
        self.robot_config = robot_config
        self.arm_ctrl = None
        self.hand_ctrl = None
        
        # Joint limits and defaults
        self.joint_limits = self._get_joint_limits()
        self.current_positions = self._get_default_positions()
        
        # Initialize controllers
        self._init_controllers()
        
    def _init_controllers(self):
        """Initialize robot controllers"""
        # Initialize arm controller
        self.arm_ctrl = G1_29_ArmController()
        
        # Initialize hand controller
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

    def _get_joint_limits(self):
        """Define joint limits for each part"""
        limits = {
            'left_arm': {
                'min': [-3.14, -2.0, -3.14, -2.0, -3.14, -2.0, -3.14],
                'max': [3.14, 2.0, 3.14, 2.0, 3.14, 2.0, 3.14]
            },
            'right_arm': {
                'min': [-3.14, -2.0, -3.14, -2.0, -3.14, -2.0, -3.14],
                'max': [3.14, 2.0, 3.14, 2.0, 3.14, 2.0, 3.14]
            }
        }
        
        limits['left_hand'] = {
            'min': [-1.05 , -0.724 ,   0  , -1.57 , -1.75 , -1.57  ,-1.75],
            'max': [1.05 ,  1.05  , 1.75 ,   0   ,  0    , 0     , 0]
        }
        limits['right_hand'] = {
            'min': [-1.05 , -1.05  , -1.75,    0  ,  0    ,   0   ,0],
            'max': [1.05 , 0.742  ,   0  ,  1.57 , 1.75  , 1.57  , 1.75]
        }
            
        return limits

    def _get_default_positions(self):
        """Get default/home positions for all joints"""
        positions = {
            'left_arm': [0.0] * 7,
            'right_arm': [0.0] * 7,
            'left_hand': [0.0] * 7,
            'right_hand': [0.0] * 7
        }
            
        return positions

    def validate_joint_values(self, part: str, values: List[float]) -> List[float]:
        """Validate and clamp joint values to limits"""
        limits = self.joint_limits[part]
        validated = []
        
        for i, val in enumerate(values):
            if i < len(limits['min']):
                clamped = max(limits['min'][i], min(limits['max'][i], val))
                validated.append(clamped)
                if clamped != val:
                    print(f"Warning: {part} joint {i} clamped from {val} to {clamped}")
            else:
                validated.append(val)
                
        return validated

    def set_joint_positions(self, positions: dict):
        """Set robot joint positions"""
        # Update current positions
        for part, values in positions.items():
            if part in self.current_positions:
                validated_values = self.validate_joint_values(part, values)
                self.current_positions[part] = validated_values

        # Set arm positions
        left_arm_pose = np.array(self.current_positions['left_arm'])
        right_arm_pose = np.array(self.current_positions['right_arm'])
        dual_arm_pose = np.concatenate([left_arm_pose, right_arm_pose])
        
        self.arm_ctrl.ctrl_dual_arm(dual_arm_pose, np.zeros(14))
        
        # Set hand positions
        self.left_hand_array[:] = self.current_positions['left_hand']
        self.right_hand_array[:] = self.current_positions['right_hand']

    def go_to_home_position(self):
        """Move robot to home position"""
        print("Moving to home position...")
        home_positions = self._get_default_positions()
        self.set_joint_positions(home_positions)
        time.sleep(1.0)
        print("Robot at home position")

    def print_current_positions(self):
        """Print current joint positions"""
        print("\n=== Current Joint Positions ===")
        for part, positions in self.current_positions.items():
            print(f"{part}: {[round(p, 3) for p in positions]}")
        print("==============================\n")

    def print_joint_info(self):
        """Print joint information and limits"""
        print("\n=== Joint Information ===")
        for part, limits in self.joint_limits.items():
            print(f"\n{part.upper()}:")
            print(f"  Number of joints: {len(limits['min'])}")
            print(f"  Min limits: {limits['min']}")
            print(f"  Max limits: {limits['max']}")
            print(f"  Current: {[round(p, 3) for p in self.current_positions[part]]}")
        print("========================\n")


def manual_control_interface(controller: ManualRobotController):
    """Interactive manual control interface"""
    print("=== Manual Robot Control Interface ===")
    print("Commands:")
    print("  'home' - Move to home position")
    print("  'show' - Show current joint positions")
    print("  'info' - Show joint information and limits")
    print("  'set <part> <values>' - Set joint values")
    print("    Example: set left_arm 0.1,0.2,0,0,0,0,0")
    print("    Example: set right_hand 0.5,0.5,0,0,0,0,0")
    print("  'quit' - Exit program")
    print("======================================\n")
    
    controller.go_to_home_position()
    
    while True:
        try:
            user_input = input("Enter command: ").strip().lower()
            
            if user_input == 'quit' or user_input == 'q':
                break
            elif user_input == 'home':
                controller.go_to_home_position()
            elif user_input == 'show':
                controller.print_current_positions()
            elif user_input == 'info':
                controller.print_joint_info()
            elif user_input.startswith('set '):
                try:
                    parts = user_input.split(' ', 2)
                    if len(parts) < 3:
                        print("Error: Invalid set command format")
                        continue
                    
                    part = parts[1]
                    values_str = parts[2]
                    
                    # Parse values
                    values = [float(x.strip()) for x in values_str.split(',')]
                    
                    # Validate part name
                    if part not in controller.current_positions:
                        print(f"Error: Unknown part '{part}'. Available parts: {list(controller.current_positions.keys())}")
                        continue
                    
                    # Check number of joints
                    expected_joints = len(controller.current_positions[part])
                    if len(values) != expected_joints:
                        print(f"Error: {part} requires {expected_joints} values, got {len(values)}")
                        continue
                    
                    # Set positions
                    positions = {part: values}
                    controller.set_joint_positions(positions)
                    print(f"Set {part} to: {[round(v, 3) for v in values]}")
                    
                except ValueError:
                    print("Error: Invalid number format in values")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Manual control session ended.")


def batch_control_from_file(controller: ManualRobotController, file_path: str):
    """Execute commands from a file"""
    try:
        with open(file_path, 'r') as f:
            commands = f.readlines()
        
        print(f"Executing {len(commands)} commands from {file_path}")
        
        for i, command in enumerate(commands):
            command = command.strip()
            if not command or command.startswith('#'):
                continue
                
            print(f"Executing command {i+1}: {command}")
            
            if command.lower() == 'home':
                controller.go_to_home_position()
            elif command.lower().startswith('set '):
                try:
                    parts = command.split(' ', 2)
                    part = parts[1]
                    values = [float(x.strip()) for x in parts[2].split(',')]
                    
                    positions = {part: values}
                    controller.set_joint_positions(positions)
                    print(f"Set {part} to: {[round(v, 3) for v in values]}")
                    
                except Exception as e:
                    print(f"Error executing command: {e}")
            
            time.sleep(0.5)  # Small delay between commands
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manual robot joint control")
    parser.add_argument("--arm_type", type=str, default="g1", 
                       help="Type of arm controller (default: g1)")
    parser.add_argument("--batch_file", type=str, default=None,
                       help="Execute commands from file instead of interactive mode")
    
    args = parser.parse_args()
    
    # Initialize logging
    init_logging()
    
    # Robot configuration
    robot_config = {
        'arm_type': args.arm_type,
    }
    
    # Initialize controller
    print("Initializing robot controller...")
    controller = ManualRobotController(robot_config)
    print("Robot controller initialized.")
    
    if args.batch_file:
        # Batch mode
        batch_control_from_file(controller, args.batch_file)
    else:
        # Interactive mode
        manual_control_interface(controller)


if __name__ == "__main__":
    main()