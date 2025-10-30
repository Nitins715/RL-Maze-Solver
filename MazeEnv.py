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

    def __init__(self, preprocessor_instance, start_coords=None, goal_coords=None):
        """
        Initializes the environment using the preprocessed grid.
        
        Args:
            preprocessor_instance: Instance of the Preprocess class.
            start_coords (tuple, optional): Custom (r, c) start coordinates. 
                                            If None, uses preprocessor's default.
            goal_coords (tuple, optional): Custom (r, c) goal coordinates.
                                           If None, uses preprocessor's default.
        """
        if not hasattr(preprocessor_instance, 'nim') or preprocessor_instance.nim is None:
            raise ValueError("Invalid Preprocess instance or grid not generated.")
            
        self.maze = preprocessor_instance.nim.copy()
        self.height, self.width = self.maze.shape
        self.current_state = None

        # --- NEW LOGIC TO HANDLE CUSTOM COORDINATES ---
        
        # 1. Get default coordinates from preprocessor
        default_start = None
        default_goal = None
        if preprocessor_instance.loc and len(preprocessor_instance.loc) == 4:
            default_start = (preprocessor_instance.loc[0], preprocessor_instance.loc[1])
            default_goal = (preprocessor_instance.loc[2], preprocessor_instance.loc[3])
        else:
            # Only raise error if custom points are ALSO not provided
            if start_coords is None or goal_coords is None:
                raise ValueError("Start/Goal coordinates not properly detected and custom coordinates not provided.")

        # 2. Set the final start and goal coordinates
        # Use custom coordinates if provided, otherwise use the detected defaults
        self.start_coords = start_coords if start_coords is not None else default_start
        self.goal_coords = goal_coords if goal_coords is not None else default_goal
        
        # Final check to ensure we have valid coordinates
        if self.start_coords is None or self.goal_coords is None:
             raise RuntimeError("Maze environment failed to initialize start or goal coordinates.")

        # --- END NEW LOGIC ---

        self.REWARD_GOAL = 10.0      
        self.REWARD_STEP = -0.04     
        self.REWARD_WALL = -1.0      

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
            
            # Check if the new state matches the goal coordinates (important for custom goals)
            if new_state == self.goal_coords: 
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
    