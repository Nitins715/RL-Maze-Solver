import numpy as np
# Assuming Preprocess.py is in the same directory or accessible via path
from Preprocess import Preprocess 

class MazeEnv:
    """
    Reinforcement Learning Environment for the preprocessed maze.
    Handles actions, state transitions, and reward calculation.
    """
    
    # Define Action Space: (dr, dc) for (row, col) change
    ACTION_SPACE = {
        0: (0, -1),  # LEFT (West)
        1: (0, 1),   # RIGHT (East)
        2: (-1, 0),  # UP (North)
        3: (1, 0)    # DOWN (South)
    }

    def __init__(self, preprocessor_instance):
        """Initializes the environment using the preprocessed grid."""
        if not isinstance(preprocessor_instance, Preprocess) or preprocessor_instance.nim is None:
            raise ValueError("Invalid Preprocess instance or grid not generated.")
            
        self.maze = preprocessor_instance.nim.copy()
        self.height, self.width = self.maze.shape
        self.start_coords = None
        self.goal_coords = None
        self.current_state = None

        # Extract start and goal coordinates from the Preprocess object
        if preprocessor_instance.loc and len(preprocessor_instance.loc) == 4:
            # self.loc is [sr, sc, gr, gc]
            self.start_coords = (preprocessor_instance.loc[0], preprocessor_instance.loc[1])
            self.goal_coords = (preprocessor_instance.loc[2], preprocessor_instance.loc[3])
        else:
            raise ValueError("Start/Goal coordinates not properly detected in the preprocessed maze.")

        # Reward values based on the grid contents
        # -1.0: Wall, 0.0: Path, 1.0: Goal, 2.0: Start (ignored for rewards)
        self.REWARD_GOAL = 10.0      # High positive reward for reaching the goal (1.0)
        self.REWARD_STEP = -0.04     # Small negative reward for each step (encourages efficiency)
        self.REWARD_WALL = -1.0      # Negative reward for hitting a wall (-1.0)
        
        print(f"Maze Environment initialized. Grid size: {self.height}x{self.width}")

    def reset(self):
        """
        Resets the agent to the starting position.
        Returns the initial state (row, col).
        """
        if self.start_coords is None:
            raise RuntimeError("Cannot reset: Start coordinates are unknown.")
            
        self.current_state = self.start_coords
        # The state for the agent is the index (row, col)
        return self.current_state

    def step(self, action_index):
        """
        Takes an action and returns the new state, reward, and done flag.

        Args:
            action_index (int): Index from ACTION_SPACE (0-3).

        Returns:
            tuple: (new_state, reward, done)
        """
        if action_index not in self.ACTION_SPACE:
            raise ValueError(f"Invalid action index: {action_index}")

        dr, dc = self.ACTION_SPACE[action_index]
        r, c = self.current_state

        # Calculate potential new coordinates
        new_r, new_c = r + dr, c + dc

        # 1. Check for boundaries
        if not (0 <= new_r < self.height and 0 <= new_c < self.width):
            # Hit boundary, stay in the same state, receive wall penalty
            reward = self.REWARD_WALL
            done = False
            new_state = self.current_state
            return new_state, reward, done

        # 2. Check for wall collision using the maze matrix value
        new_cell_value = self.maze[new_r, new_c]

        if new_cell_value == -1.0: # Wall collision
            reward = self.REWARD_WALL
            done = False
            new_state = self.current_state # Agent does not move
        else:
            # Valid movement (Path or Goal)
            self.current_state = (new_r, new_c)
            new_state = self.current_state
            
            if new_cell_value == 1.0: # Goal reached
                reward = self.REWARD_GOAL
                done = True
            else: # Path movement (value is 0.0 or 2.0)
                reward = self.REWARD_STEP
                done = False

        return new_state, reward, done
    
    def get_state_index(self, state):
        """Converts (row, col) state coordinates to a single linear index for Q-table."""
        r, c = state
        return r * self.width + c

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # 1. First, preprocess the image
        maze_path = "Mazes/20 by 20 maze.png" 
        p = Preprocess(maze_path) 
        p.generate(margin=0.005, pix=0) 
        
        # 2. Initialize the environment with the preprocessed data
        env = MazeEnv(p)
        
        # 3. Test the environment steps
        current_state = env.reset()
        print(f"Start State (r, c): {current_state}")
        
        # Example steps: Right (1), Down (3), Left (0), Up (2)
        actions = [1, 1, 3, 3, 0, 2] 
        print(f"Testing actions: {actions}")
        
        for action in actions:
            new_state, reward, done = env.step(action)
            print(f"  Action {action}: New State={new_state}, Reward={reward:.2f}, Done={done}")
            if done:
                print("Episode finished.")
                break
                
    except Exception as e:
        print(f"\n[Environment Test Error]: {e}")
