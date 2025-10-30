import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd # Used for clean comparison table
from Preprocess import Preprocess
from MazeEnv import MazeEnv
from QLearningAgent import QLearningAgent
from SARSAAgent import SARSAAgent

# --- Configuration ---
MAZE_FILE = "Mazes/20 by 20 maze.png" # <<< IMPORTANT: Update this path
NUM_EPISODES = 10000 # Increased episodes for better SARSA convergence
MAX_STEPS_PER_EPISODE = 1000 

# --- Hyperparameters (Shared) ---
ALPHA = 0.1     # Learning Rate
GAMMA = 0.95    # Discount Factor
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.9997 # Slightly slower decay for better exploration in both

def train_agent(env, agent, num_episodes, agent_name):
    """Generic function to train either Q-Learning or SARSA Agent."""
    print(f"\n--- Starting {agent_name} Training ({num_episodes} episodes) ---")
    start_time = time.time()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        current_state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        # SARSA requires selecting the first action a before the loop starts
        action = agent.select_action(current_state) 
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            old_state = current_state
            old_action = action
            
            # 1. Take a step in the environment
            new_state, reward, done = env.step(old_action)
            
            # 2. Select the next action (a') *before* the update (required by SARSA)
            next_action = agent.select_action(new_state)
            
            # 3. Update the Q-table
            if agent_name == 'Q-Learning':
                # Q-Learning update: depends only on max Q(s', a')
                agent.update_q_table(old_state, old_action, reward, new_state)
            else: # SARSA
                # SARSA update: depends on Q(s', a')
                agent.update_q_table(old_state, old_action, reward, new_state, next_action)
            
            current_state = new_state
            action = next_action # Move a' to a for the next iteration
            total_reward += reward
            steps += 1
            
        # Decay epsilon for the next episode
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        if (episode + 1) % (num_episodes // 5) == 0:
            avg_reward = np.mean(episode_rewards[-(num_episodes // 10):])
            print(f"  Episode {episode + 1}/{num_episodes}: Avg Reward={avg_reward:.2f}, Epsilon={agent.epsilon:.4f}")

    end_time = time.time()
    train_duration = end_time - start_time
    print(f"{agent_name} Training finished in {train_duration:.2f} seconds.")
    return train_duration, agent.get_optimal_path()

def visualize_path(maze_grid, path, algorithm_name, duration, path_length, plot_index):
    """Visualizes the maze grid and the found path for a specific agent."""
    H, W = maze_grid.shape
    
    plt.subplot(1, 2, plot_index) # Create side-by-side plots
    
    # Plot the base maze structure
    plt.imshow(maze_grid, aspect='equal', vmin=-1.0, vmax=2.0) 
    
    # Plot the path if found
    if path:
        path_rows = [r for r, c in path]
        path_cols = [c for r, c in path]
        
        # Draw the path
        plt.plot(path_cols, path_rows, color='red', linewidth=3, marker='o', 
                 markersize=5, markerfacecolor='yellow', markeredgecolor='red')

    # Add color bar and labels
    plt.title(f'{algorithm_name}\nPath Length: {path_length} | Time: {duration:.2f}s', fontsize=10)
    plt.xticks(np.arange(-.5, W, 1), minor=True)
    plt.yticks(np.arange(-.5, H, 1), minor=True)
    plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    plt.xticks([]), plt.yticks([])

def run_solver():
    try:
        # --- 1. Preprocessing ---
        print(f"--- Loading and Preprocessing Maze: {MAZE_FILE} ---")
        pre = Preprocess(MAZE_FILE) 
        pre.generate(margin=0.005, pix=0) 
        
        # --- 2. Environment Setup ---
        env = MazeEnv(pre)
        
        # --- 3. Agent Setup ---
        q_agent = QLearningAgent(env, alpha=ALPHA, gamma=GAMMA, 
                                 epsilon=INITIAL_EPSILON, min_epsilon=MIN_EPSILON, 
                                 epsilon_decay=EPSILON_DECAY)
                                 
        sarsa_agent = SARSAAgent(env, alpha=ALPHA, gamma=GAMMA, 
                                 epsilon=INITIAL_EPSILON, min_epsilon=MIN_EPSILON, 
                                 epsilon_decay=EPSILON_DECAY)

        results = {}

        # --- 4. Training Q-Learning ---
        q_duration, q_path = train_agent(env, q_agent, NUM_EPISODES, 'Q-Learning')
        q_path_length = len(q_path) - 1 if q_path else "Not Found"
        results['Q-Learning'] = {'Duration': q_duration, 'Path': q_path, 'Length': q_path_length}
        
        # --- 5. Training SARSA ---
        sarsa_duration, sarsa_path = train_agent(env, sarsa_agent, NUM_EPISODES, 'SARSA')
        sarsa_path_length = len(sarsa_path) - 1 if sarsa_path else "Not Found"
        results['SARSA'] = {'Duration': sarsa_duration, 'Path': sarsa_path, 'Length': sarsa_path_length}

        # --- 6. Comparison and Visualization ---
        
        # Create a Pandas DataFrame for clean printing of results
        df = pd.DataFrame({
            'Algorithm': ['Q-Learning', 'SARSA'],
            'Training Time (s)': [results['Q-Learning']['Duration'], results['SARSA']['Duration']],
            'Path Length (steps)': [results['Q-Learning']['Length'], results['SARSA']['Length']]
        }).set_index('Algorithm')
        
        print("\n" + "="*50)
        print("         🏁 Algorithm Comparison Results 🏁")
        print("="*50)
        # Using to_markdown for structured console output
        print(df.to_markdown(numalign="left", stralign="left"))
        print("="*50)

        # Matplotlib Visualization
        plt.figure(figsize=(18, 9))
        
        visualize_path(pre.nim, results['Q-Learning']['Path'], 'Q-Learning (Off-Policy)', 
                       results['Q-Learning']['Duration'], results['Q-Learning']['Length'], 1)
                       
        visualize_path(pre.nim, results['SARSA']['Path'], 'SARSA (On-Policy)', 
                       results['SARSA']['Duration'], results['SARSA']['Length'], 2)

        plt.suptitle(f"RL Maze Solver Comparison: {pre.h}x{pre.w} Grid", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}\nNOTE: Please ensure you have a maze image file at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_solver()
