import numpy as np

class MCAgent:
    """
    Monte Carlo (MC) Agent for the Maze Environment.
    An episode-based, model-free learning algorithm.
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.9997):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.num_actions = len(env.ACTION_SPACE)
        self.num_states = env.height * env.width
        
        # Initialize Q-table: rows=states, columns=actions
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        state_idx = self.env.get_state_index(state)
        
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.num_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            return np.argmax(self.q_table[state_idx, :])

    def update_q_table(self, trajectory):
        """
        Updates Q-table using the episode's full trajectory (First-Visit Monte Carlo).
        trajectory format: [(state, action, reward), ...]
        """
        
        # Dictionary to track which (state, action) pairs have already been updated in this episode
        sa_visited = set()
        
        # G is the return (discounted cumulative reward)
        G = 0 
        
        # Iterate backward through the trajectory
        # This allows us to calculate G_t efficiently
        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            state_idx = self.env.get_state_index(state)
            sa_pair = (state_idx, action)
            
            # Update the return G
            G = self.gamma * G + reward
            
            # First-Visit MC: Only update the Q-value if this (s, a) pair hasn't been visited yet in this episode
            if sa_pair not in sa_visited:
                # Update rule: Q(s, a) = Q(s, a) + alpha * (G_t - Q(s, a))
                old_q = self.q_table[state_idx, action]
                self.q_table[state_idx, action] = old_q + self.alpha * (G - old_q)
                sa_visited.add(sa_pair)


    def decay_epsilon(self):
        """Decreases the exploration rate."""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def get_optimal_path(self):
        """Extracts the optimal path from the final Q-table."""
        # Same logic as Q-Learning and SARSA
        path = [self.env.start_coords]
        current_state = self.env.start_coords
        
        # Use a max step limit to prevent infinite loops in cases of unlearned dead ends
        max_steps = self.env.height * self.env.width * 2
        
        while current_state != self.env.goal_coords and len(path) < max_steps:
            state_idx = self.env.get_state_index(current_state)
            
            # Select the best action greedily
            action = np.argmax(self.q_table[state_idx, :])
            
            # Take the step in the environment (read-only mode)
            dr, dc = self.env.ACTION_SPACE[action]
            r, c = current_state
            
            new_r, new_c = r + dr, c + dc
            
            new_state = (new_r, new_c)
            
            # Check if new state is valid and not a wall (shouldn't happen with perfect Q-table)
            if (0 <= new_r < self.env.height and 0 <= new_c < self.env.width 
                and self.env.maze[new_state] != -1.0):
                current_state = new_state
                path.append(current_state)
            else:
                # Agent got stuck or Q-table led to a bad state (should not happen if trained well)
                break 

        return path
