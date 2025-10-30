import numpy as np
from MazeEnv import MazeEnv

class SARSAAgent:
    """
    Implements a Tabular SARSA Agent (On-Policy).
    The update rule uses the Q-value of the action *actually taken* in the next state.
    """
    def __init__(self, env: MazeEnv, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.9995):
        """Initializes the SARSA agent with hyperparameters and the Q-table."""
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        
        self.num_states = env.height * env.width
        self.num_actions = len(env.ACTION_SPACE)
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

    def select_action(self, state):
        """
        Selects an action using the epsilon-greedy policy.
        """
        state_idx = self.env.get_state_index(state)

        if np.random.random() < self.epsilon:
            # Exploration: Choose a random action
            action = np.random.randint(self.num_actions)
        else:
            # Exploitation: Choose the action with the highest Q-value
            action = np.argmax(self.q_table[state_idx, :])
            
        return action

    def update_q_table(self, old_state, action, reward, new_state, next_action):
        """
        Performs the SARSA update rule.

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
        
        Args:
            next_action (int): The action that will be taken in the next state (s').
        """
        old_state_idx = self.env.get_state_index(old_state)
        new_state_idx = self.env.get_state_index(new_state)
        
        current_q = self.q_table[old_state_idx, action]
        
        # SARSA difference: Use Q(s', a') where a' is the action selected by the policy for s'.
        future_q = self.q_table[new_state_idx, next_action] 
        
        # Calculate the Temporal Difference (TD) Target
        td_target = reward + self.gamma * future_q
        
        # Calculate the TD Error
        td_error = td_target - current_q
        
        # Update the Q-value
        self.q_table[old_state_idx, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decreases the exploration rate over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def get_optimal_path(self):
        """
        Uses the trained Q-table (greedily) to find the shortest path from start to goal.
        """
        path = []
        current_state = self.env.reset()
        max_steps = self.env.height * self.env.width * 2 
        steps = 0
        
        while current_state != self.env.goal_coords and steps < max_steps:
            path.append(current_state)
            
            # Select the action greedily (no exploration)
            state_idx = self.env.get_state_index(current_state)
            action = np.argmax(self.q_table[state_idx, :])
            
            # Manually transition state using the environment's step logic
            dr, dc = self.env.ACTION_SPACE[action]
            r, c = current_state
            
            new_r, new_c = r + dr, c + dc
            
            # Check for movement validation
            if (0 <= new_r < self.env.height and 0 <= new_c < self.env.width) and \
               self.env.maze[new_r, new_c] != -1.0:
                current_state = (new_r, new_c)
            
            steps += 1

        if current_state == self.env.goal_coords:
            path.append(self.env.goal_coords)
            return path
        else:
            return None 
