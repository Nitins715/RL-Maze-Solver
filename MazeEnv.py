import numpy as np

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
        if not hasattr(preprocessor_instance, 'nim') or preprocessor_instance.nim is None:
            raise ValueError("Invalid Preprocess instance or grid not generated.")
            
        self.maze = preprocessor_instance.nim.copy()
        self.height, self.width = self.maze.shape
        self.start_coords = None
        self.goal_coords = None
        self.current_state = None

        if preprocessor_instance.loc and len(preprocessor_instance.loc) == 4:
            self.start_coords = (preprocessor_instance.loc[0], preprocessor_instance.loc[1])
            self.goal_coords = (preprocessor_instance.loc[2], preprocessor_instance.loc[3])
        else:
            raise ValueError("Start/Goal coordinates not properly detected in the preprocessed maze.")

        self.REWARD_GOAL = 10.0      # High positive reward for reaching the goal (1.0)
        self.REWARD_STEP = -0.04     # Small negative reward for each step (encourages efficiency)
        self.REWARD_WALL = -1.0      # Negative reward for hitting a wall (-1.0)

    def reset(self):
        """Resets the agent to the starting position."""
        if self.start_coords is None:
            raise RuntimeError("Cannot reset: Start coordinates are unknown.")
            
        self.current_state = self.start_coords
        return self.current_state

    def step(self, action_index):
        """Takes an action and returns the new state, reward, and done flag."""
        if action_index not in self.ACTION_SPACE:
            raise ValueError(f"Invalid action index: {action_index}")

        dr, dc = self.ACTION_SPACE[action_index]
        r, c = self.current_state

        new_r, new_c = r + dr, c + dc

        # 1. Check for boundaries
        if not (0 <= new_r < self.height and 0 <= new_c < self.width):
            reward = self.REWARD_WALL
            done = False
            new_state = self.current_state
            return new_state, reward, done

        # 2. Check for wall collision
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
            else: # Path movement
                reward = self.REWARD_STEP
                done = False

        return new_state, reward, done
    
    def get_state_index(self, state):
        """Converts (r, c) state coordinates to a single linear index for Q-table."""
        r, c = state
        return r * self.width + c

    def heuristic(self, a, b):
        """
        Calculates the Euclidean distance between two points a and b (used for A*).
        This is an admissible (never overestimates) heuristic for grid movement.
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
