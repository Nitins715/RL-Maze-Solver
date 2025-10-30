import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd 
from Preprocess import Preprocess
from MazeEnv import MazeEnv
from QLearningAgent import QLearningAgent
from SARSAAgent import SARSAAgent
# --- NEW IMPORT ---
from MCAgent import MCAgent
# ------------------
from PathfindingAlgorithms import PathfindingAlgorithms

# --- Configuration ---
MAZE_FILE = "Mazes/12 by 12 maze.png" # <<< IMPORTANT: Update this path
NUM_EPISODES = 10000 
MAX_STEPS_PER_EPISODE = 1000 

# --- RL Hyperparameters (Shared) ---
ALPHA = 0.1     
GAMMA = 0.95    
EPSILON_DECAY = 0.9999

# --- Global Result Storage ---
ALGO_RESULTS = {}
AGENTS = {
    'Q-Learning': None,
    'SARSA': None,
    'Monte Carlo': None # Placeholder for future agent
}

# ----------------------------------------------------------------------------------
## 🤖 Training Functions for Step-Based Agents (Q-Learning, SARSA)
# ----------------------------------------------------------------------------------

def train_step_rl_agent(env, agent, num_episodes, agent_name):
    """
    Generic training loop for step-based RL agents (Q-Learning and SARSA).
    Updates the Q-table after every single step.
    """
    print(f"\n--- Starting {agent_name} Training ({num_episodes} episodes) ---")
    start_time = time.time()
    
    for episode in range(num_episodes):
        current_state = env.reset()
        done = False
        steps = 0
        
        # SARSA requires selecting the first action a before the loop starts
        action = agent.select_action(current_state) 
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            old_state = current_state
            old_action = action
            
            new_state, reward, done = env.step(old_action)
            next_action = agent.select_action(new_state)
            
            # --- Update Rule Differentiator ---
            if agent_name == 'Q-Learning':
                # Off-Policy update: uses max Q(s', a')
                agent.update_q_table(old_state, old_action, reward, new_state)
            elif agent_name == 'SARSA':
                # On-Policy update: uses Q(s', a')
                agent.update_q_table(old_state, old_action, reward, new_state, next_action)
            
            current_state = new_state
            action = next_action
            steps += 1
            
        agent.decay_epsilon()
        
    train_duration = time.time() - start_time
    path = agent.get_optimal_path()
    path_length = len(path) - 1 if path else "Not Found"

    ALGO_RESULTS[agent_name] = {'Time': train_duration, 'Path': path, 'Length': path_length, 'Type': 'RL'}
    print(f"  {agent_name} finished in {train_duration:.4f}s. Path Length: {path_length}")

# ----------------------------------------------------------------------------------
## 💰 Training Function for Monte Carlo Agent
# ----------------------------------------------------------------------------------

def train_mc_agent(env, agent: MCAgent, num_episodes, agent_name='Monte Carlo'):
    """
    Monte Carlo-specific training loop (episode-based update).
    Collects the full trajectory before updating the Q-table.
    """
    print(f"\n--- Starting {agent_name} Training ({num_episodes} episodes) ---")
    start_time = time.time()
    
    for episode in range(num_episodes):
        current_state = env.reset()
        done = False
        trajectory = []
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            # 1. Select Action
            action = agent.select_action(current_state)
            
            # 2. Step in Env
            new_state, reward, done = env.step(action)
            
            # 3. Record Trajectory (state, action, reward)
            trajectory.append((current_state, action, reward)) 
            
            current_state = new_state
            steps += 1
        
        # 4. Update Q-table with the full episode trajectory
        if trajectory:
            agent.update_q_table(trajectory)
        
        # 5. Decay Epsilon
        agent.decay_epsilon()
            
    train_duration = time.time() - start_time
    path = agent.get_optimal_path()
    path_length = len(path) - 1 if path else "Not Found"

    ALGO_RESULTS[agent_name] = {'Time': train_duration, 'Path': path, 'Length': path_length, 'Type': 'RL'}
    print(f"  {agent_name} finished in {train_duration:.4f}s. Path Length: {path_length}")

# ----------------------------------------------------------------------------------
## 🗺️ Pathfinding Solver
# ----------------------------------------------------------------------------------

def solve_pathfinding(pathfinder: PathfindingAlgorithms):
    """Executes all pathfinding algorithms."""
    
    # 1. Dijkstra's Algorithm
    print("\n--- Starting Pathfinding Algorithms ---")
    d_time, d_path = pathfinder.dijkstra_search()
    d_len = len(d_path) - 1 if d_path else "Not Found"
    ALGO_RESULTS['Dijkstra’s'] = {'Time': d_time, 'Path': d_path, 'Length': d_len, 'Type': 'Pathfinding'}
    print(f"  Dijkstra's finished in {d_time:.6f}s. Path Length: {d_len}")

    # 2. A* Search
    a_time, a_path = pathfinder.a_star_search()
    a_len = len(a_path) - 1 if a_path else "Not Found"
    ALGO_RESULTS['A* Search'] = {'Time': a_time, 'Path': a_path, 'Length': a_len, 'Type': 'Pathfinding'}
    print(f"  A* Search finished in {a_time:.6f}s. Path Length: {a_len}")

    # 3. Bi-Directional Search
    b_time, b_path = pathfinder.bi_directional_search()
    b_len = len(b_path) - 1 if b_path else "Not Found"
    ALGO_RESULTS['Bi-Directional Search'] = {'Time': b_time, 'Path': b_path, 'Length': b_len, 'Type': 'Pathfinding'}
    print(f"  Bi-Directional Search finished in {b_time:.6f}s. Path Length: {b_len}")

# ----------------------------------------------------------------------------------
## 🖼️ Visualization and Main Runner
# ----------------------------------------------------------------------------------

def visualize_results(maze_grid):
    """Generates a visualization showing all solved paths."""
    
    # Sort results for consistent display order
    algorithms = sorted(ALGO_RESULTS.keys())
    num_algos = len(algorithms)
    
    # Determine grid size (e.g., 2 rows, 3 or 4 columns)
    rows = int(np.ceil(num_algos / 3))
    cols = min(3, num_algos)

    plt.figure(figsize=(6 * cols, 6 * rows))
    
    for i, algo_name in enumerate(algorithms):
        result = ALGO_RESULTS[algo_name]
        path = result['Path']
        path_length = result['Length']
        duration = result['Time']
        algo_type = result['Type']
        
        plt.subplot(rows, cols, i + 1)
        
        # Plot the base maze structure
        plt.imshow(maze_grid, aspect='equal', vmin=-1.0, vmax=2.0) 
        
        if path:
            path_rows = [r for r, c in path]
            path_cols = [c for r, c in path]
            
            # Draw the path (use different colors/styles if possible)
            color = 'red' if algo_type == 'Pathfinding' else 'blue'
            
            plt.plot(path_cols, path_rows, color=color, linewidth=2, marker='.', 
                     markersize=5, markerfacecolor='yellow', markeredgecolor='black')

        plt.title(f'{algo_name} ({algo_type}) \nLength: {path_length} | Time: {duration:.4f}s', fontsize=10)
        plt.xticks([]), plt.yticks([])
        plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.suptitle(f"Maze Solver Comparison ({num_algos} Algorithms)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_solver():
    try:
        # --- 1. Preprocessing and Environment Setup ---
        print(f"--- Loading and Preprocessing Maze: {MAZE_FILE} ---")
        pre = Preprocess(MAZE_FILE) 
        pre.generate(margin=0.005, pix=0) 
        env = MazeEnv(pre)
        pathfinder = PathfindingAlgorithms(env)
        
        # --- 2. Pathfinding Solver (Baseline) ---
        print("\n" + "="*50)
        print("Starting Classical Pathfinding Algorithms...")
        print("="*50)
        solve_pathfinding(pathfinder)

        # --- 3. RL Agent Training ---
        print("\n" + "="*50)
        print("Starting Reinforcement Learning Agents...")
        print("="*50)
        
        # Instantiate Agents
        q_agent = QLearningAgent(env, alpha=ALPHA, gamma=GAMMA, epsilon=1.0, min_epsilon=0.01, epsilon_decay=EPSILON_DECAY)
        sarsa_agent = SARSAAgent(env, alpha=ALPHA, gamma=GAMMA, epsilon=1.0, min_epsilon=0.01, epsilon_decay=EPSILON_DECAY)
        mc_agent = MCAgent(env, alpha=ALPHA, gamma=GAMMA, epsilon=1.0, min_epsilon=0.01, epsilon_decay=EPSILON_DECAY) 
        
        # Run Training for Step-Based Agents
        train_step_rl_agent(env, q_agent, NUM_EPISODES, 'Q-Learning')
        train_step_rl_agent(env, sarsa_agent, 2*NUM_EPISODES, 'SARSA')
        train_mc_agent(env, mc_agent, 3*NUM_EPISODES, 'Monte Carlo')
        
        # --- 4. Final Comparison and Visualization ---
        
        # Create a Pandas DataFrame for clean printing of results
        df = pd.DataFrame.from_dict(ALGO_RESULTS, orient='index')
        df.index.name = 'Algorithm'
        
        print("\n" + "="*50)
        print("         🏁 Comprehensive Algorithm Comparison 🏁")
        print("="*50)
        df_display = df.sort_values(by='Time', ascending=True).drop(columns=['Path'])
        df_display['Time (s)'] = df_display['Time'].apply(lambda x: f"{x:.6f}")
        df_display = df_display.drop(columns=['Time'])
        print(df_display.to_markdown(numalign="left", stralign="left"))
        print("="*50)

        visualize_results(pre.nim)

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}\nNOTE: Please ensure you have a maze image file at the specified path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_solver()