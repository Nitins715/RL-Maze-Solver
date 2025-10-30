import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd 
import os # Added for robust file path handling

from Preprocess import Preprocess
from MazeEnv import MazeEnv # Assuming this file has been updated
from QLearningAgent import QLearningAgent
from SARSAAgent import SARSAAgent
from MCAgent import MCAgent 
from PathfindingAlgorithms import PathfindingAlgorithms

# --- Configuration ---
# IMPORTANT: Use os.path.join for platform compatibility
MAZE_FILE = os.path.join("Mazes", "12 by 12 maze.png") 
NUM_EPISODES = 10000 
MAX_STEPS_PER_EPISODE = 1000 

# --- RL Hyperparameters (Shared) ---
ALPHA = 0.1     
GAMMA = 0.95    
EPSILON_DECAY = 0.9999

# --- Global Result Storage ---
ALGO_RESULTS = {}

# ----------------------------------------------------------------------------------
## 🖱️ User Input for Custom Points 
# ----------------------------------------------------------------------------------

def get_custom_points(maze_grid, env_height, env_width):
    """
    Displays the maze grid and allows the user to click to select 
    Start and Goal points. Converts pixel coordinates to grid coordinates.
    """
    
    plt.figure(figsize=(8, 8))
    plt.imshow(maze_grid, aspect='equal', vmin=-1.0, vmax=2.0)
    plt.title("Click to select START point, then click to select GOAL point.", fontsize=12)
    plt.xticks([]), plt.yticks([])
    
    print("\n\n--- WAITING FOR USER INPUT ---")
    print("Please click on the START point, then click on the GOAL point (or close the window to use defaults).")

    try:
        # Use ginput to get 2 mouse clicks
        coords = plt.ginput(2, timeout=60) 
    except Exception:
        # Catch exceptions if the window is closed abruptly
        coords = []
    
    plt.close() # Close the selection window immediately

    if len(coords) < 2:
        print("❌ Did not select two points. Using default Start/Goal.")
        return None, None

    # ginput returns (col, row) coordinates (x, y)
    start_x, start_y = coords[0]
    goal_x, goal_y = coords[1]

    # Convert pixel coordinates (float) to integer grid coordinates (row, col)
    start_r, start_c = int(round(start_y)), int(round(start_x))
    goal_r, goal_c = int(round(goal_y)), int(round(goal_x))
    
    # Check bounds and adjust if necessary
    start_r = np.clip(start_r, 0, env_height - 1)
    start_c = np.clip(start_c, 0, env_width - 1)
    goal_r = np.clip(goal_r, 0, env_height - 1)
    goal_c = np.clip(goal_c, 0, env_width - 1)

    print(f"✅ User Selected Start: ({start_r}, {start_c}), Goal: ({goal_r}, {goal_c})")
    return (start_r, start_c), (goal_r, goal_c)

# ----------------------------------------------------------------------------------
## 🤖 Training Functions (with Early Exit Logic)
# ----------------------------------------------------------------------------------

def train_step_rl_agent(env, agent, num_episodes, agent_name, optimal_benchmark_length):
    """
    Generic training loop for step-based RL agents (Q-Learning and SARSA). 
    Aborts early if the optimal path is found.
    """
    print(f"\n--- Starting {agent_name} Training ({num_episodes} episodes) ---")
    start_time = time.time()
    
    # Check frequency for early exit (e.g., every 100 episodes)
    check_frequency = max(1, num_episodes // 1000)
    
    for episode in range(num_episodes):
        current_state = env.reset()
        done = False
        steps = 0
        action = agent.select_action(current_state) 
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            old_state = current_state
            old_action = action
            
            new_state, reward, done = env.step(old_action)
            next_action = agent.select_action(new_state)
            
            # Update Rule Differentiator
            if agent_name == 'Q-Learning':
                agent.update_q_table(old_state, old_action, reward, new_state)
            elif agent_name == 'SARSA':
                agent.update_q_table(old_state, old_action, reward, new_state, next_action)
            
            current_state = new_state
            action = next_action
            steps += 1
            
        agent.decay_epsilon()
        
        # --- EARLY EXIT CHECK ---
        if optimal_benchmark_length != "Not Found" and (episode + 1) % check_frequency == 0:
            path = agent.get_optimal_path()
            path_length = len(path) - 1 if path else "Not Found"

            if path_length == optimal_benchmark_length:
                train_duration = time.time() - start_time
                ALGO_RESULTS[agent_name] = {'Time': train_duration, 'Path': path, 'Length': path_length, 'Type': 'RL'}
                print(f"✅ Success! {agent_name} ne optimal path ({path_length} steps) {episode+1} episodes mein hi dhoond liya.")
                return path
        # ------------------------
        
    train_duration = time.time() - start_time
    path = agent.get_optimal_path()
    path_length = len(path) - 1 if path else "Not Found"

    ALGO_RESULTS[agent_name] = {'Time': train_duration, 'Path': path, 'Length': path_length, 'Type': 'RL'}
    print(f"  {agent_name} finished in {train_duration:.4f}s. Path Length: {path_length}")
    return None 

def train_mc_agent(env, agent: MCAgent, num_episodes, agent_name, optimal_benchmark_length):
    """
    Monte Carlo-specific training loop (Episode-Based Update). Aborts early if 
    the optimal path is found.
    """
    print(f"\n--- Starting {agent_name} Training ({num_episodes} episodes) ---")
    start_time = time.time()
    
    # Check frequency for early exit
    check_frequency = max(1, num_episodes // 500) # Check Monte Carlo less frequently, as its path quality jumps
    
    for episode in range(num_episodes):
        current_state = env.reset()
        done = False
        trajectory = []
        steps = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = agent.select_action(current_state)
            new_state, reward, done = env.step(action)
            trajectory.append((current_state, action, reward)) 
            current_state = new_state
            steps += 1
        
        if trajectory:
            agent.update_q_table(trajectory)
        
        agent.decay_epsilon()
        
        # --- EARLY EXIT CHECK ---
        if optimal_benchmark_length != "Not Found" and (episode + 1) % check_frequency == 0:
            path = agent.get_optimal_path()
            path_length = len(path) - 1 if path else "Not Found"
            
            if path_length == optimal_benchmark_length:
                train_duration = time.time() - start_time
                ALGO_RESULTS[agent_name] = {'Time': train_duration, 'Path': path, 'Length': path_length, 'Type': 'RL'}
                print(f"✅ Success! {agent_name} ne optimal path ({path_length} steps) {episode+1} episodes mein hi dhoond liya.")
                return path
        # ------------------------
            
    train_duration = time.time() - start_time
    path = agent.get_optimal_path()
    path_length = len(path) - 1 if path else "Not Found"

    ALGO_RESULTS[agent_name] = {'Time': train_duration, 'Path': path, 'Length': path_length, 'Type': 'RL'}
    print(f"  {agent_name} finished in {train_duration:.4f}s. Path Length: {path_length}")
    return None 

# ----------------------------------------------------------------------------------
## 🗺️ Pathfinding Solver 
# ----------------------------------------------------------------------------------

def solve_pathfinding(pathfinder: PathfindingAlgorithms):
    """Executes all pathfinding algorithms."""
    
    print("\n--- Starting Classical Pathfinding Algorithms ---")
    d_time, d_path = pathfinder.dijkstra_search()
    d_len = len(d_path) - 1 if d_path else "Not Found"
    ALGO_RESULTS['Dijkstra’s'] = {'Time': d_time, 'Path': d_path, 'Length': d_len, 'Type': 'Pathfinding'}
    print(f"  Dijkstra's finished in {d_time:.6f}s. Path Length: {d_len}")

    a_time, a_path = pathfinder.a_star_search()
    a_len = len(a_path) - 1 if a_path else "Not Found"
    ALGO_RESULTS['A* Search'] = {'Time': a_time, 'Path': a_path, 'Length': a_len, 'Type': 'Pathfinding'}
    print(f"  A* Search finished in {a_time:.6f}s. Path Length: {a_len}")

    b_time, b_path = pathfinder.bi_directional_search()
    b_len = len(b_path) - 1 if b_path else "Not Found"
    ALGO_RESULTS['Bi-Directional Search'] = {'Time': b_time, 'Path': b_path, 'Length': b_len, 'Type': 'Pathfinding'}
    print(f"  Bi-Directional Search finished in {b_time:.6f}s. Path Length: {b_len}")


# ----------------------------------------------------------------------------------
## 🖼️ Visualization and Main Runner 
# ----------------------------------------------------------------------------------

def visualize_results(maze_grid, env): # <-- ENV IS NOW REQUIRED HERE
    """Generates a visualization showing all solved paths."""
    
    # Sort results for consistent display order
    algorithms = sorted(ALGO_RESULTS.keys())
    num_algos = len(algorithms)
    
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
        
        plt.imshow(maze_grid, aspect='equal', vmin=-1.0, vmax=2.0) 
        
        if path and path_length != "Not Found":
            path_rows = [r for r, c in path]
            path_cols = [c for r, c in path]
            
            color = 'red' if algo_type == 'Pathfinding' else 'blue'
            
            plt.plot(path_cols, path_rows, color=color, linewidth=2, marker='.', 
                     markersize=5, markerfacecolor='yellow', markeredgecolor='black')
        
        # Highlight custom Start and Goal points (FIXED ERROR)
        start_r, start_c = env.start_coords
        goal_r, goal_c = env.goal_coords
        plt.scatter(start_c, start_r, marker='D', color='lime', s=100, label='Start')
        plt.scatter(goal_c, goal_r, marker='*', color='gold', s=200, label='Goal')


        plt.title(f'{algo_name} ({algo_type}) \nLength: {path_length} | Time: {duration:.4f}s', fontsize=10)
        plt.xticks([]), plt.yticks([])
        plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.suptitle(f"Maze Solver Comparison ({num_algos} Algorithms) | S:{env.start_coords} -> G:{env.goal_coords}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def run_solver():
    global ALGO_RESULTS 
    try:
        # --- 1. Preprocessing and Environment Setup ---
        print(f"--- Loading and Preprocessing Maze: {MAZE_FILE} ---")
        pre = Preprocess(MAZE_FILE) 
        pre.generate(margin=0.005, pix=0) 
        
        # Initialize ENV to get dimensions and defaults for user input
        env_temp = MazeEnv(pre)
        
        # --- 2. Get Custom User Points ---
        custom_start, custom_goal = get_custom_points(pre.nim, env_temp.height, env_temp.width)
        
        # Re-initialize Environment with user's points (if provided)
        if custom_start and custom_goal:
            env = MazeEnv(pre, start_coords=custom_start, goal_coords=custom_goal)
        else:
            env = env_temp # Use defaults
            
        pathfinder = PathfindingAlgorithms(env)
        
        ALGO_RESULTS = {}
        
        # --- 3. Pathfinding Solver (Benchmark Set) ---
        print("\n" + "="*50)
        print(f"Starting Classical Pathfinding Algorithms for Benchmark (Start: {env.start_coords} | Goal: {env.goal_coords})...")
        print("="*50)
        solve_pathfinding(pathfinder)

        # Calculate Optimal Benchmark
        df_pathfinding = pd.DataFrame.from_dict(ALGO_RESULTS, orient='index')
        pathfinding_lengths = df_pathfinding[df_pathfinding['Type'] == 'Pathfinding']
        valid_lengths = pathfinding_lengths[pathfinding_lengths['Length'] != 'Not Found']['Length']
        
        if valid_lengths.empty:
            optimal_benchmark = "Not Found"
            print("⚠️ Pathfinding ne koi rasta nahi dhunda. RL agents ko bina benchmark ke poora run kiya jaaega.")
        else:
            optimal_benchmark = valid_lengths.min()
            print(f"⭐ Optimal Pathfinding Benchmark Set: {optimal_benchmark} steps.")
        
        
        # --- 4. RL Agent Training (Sequential Check with Early Exit) ---
        print("\n" + "="*50)
        print("Starting Reinforcement Learning Agents (Checking for Early Exit)...")
        print("="*50)
        
        # Instantiate Agents (using the final 'env')
        q_agent = QLearningAgent(env, alpha=ALPHA, gamma=GAMMA, epsilon=1.0, min_epsilon=0.01, epsilon_decay=EPSILON_DECAY)
        sarsa_agent = SARSAAgent(env, alpha=ALPHA, gamma=GAMMA, epsilon=1.0, min_epsilon=0.01, epsilon_decay=EPSILON_DECAY)
        mc_agent = MCAgent(env, alpha=ALPHA, gamma=GAMMA, epsilon=1.0, min_epsilon=0.01, epsilon_decay=EPSILON_DECAY) 
        
        if optimal_benchmark != "Not Found":
            # 1. Q-Learning Run
            found_path = train_step_rl_agent(env, q_agent, NUM_EPISODES, 'Q-Learning', optimal_benchmark)
            if found_path is not None:
                print("\n" + "="*50)
                print(f"🏁 **FINAL RESULT:** {q_agent.__class__.__name__} ne optimal path dhoond liya!")
                print(f"Path Length: {optimal_benchmark}")
                print(f"Path Coordinates: {' -> '.join([str(coord) for coord in found_path])}")
                visualize_results(pre.nim, env) # <-- FIXED CALL
                return
            
            # 2. SARSA Run
            found_path = train_step_rl_agent(env, sarsa_agent, 2*NUM_EPISODES, 'SARSA', optimal_benchmark)
            if found_path is not None:
                print("\n" + "="*50)
                print(f"🏁 **FINAL RESULT:** {sarsa_agent.__class__.__name__} ne optimal path dhoond liya!")
                print(f"Path Length: {optimal_benchmark}")
                print(f"Path Coordinates: {' -> '.join([str(coord) for coord in found_path])}")
                visualize_results(pre.nim, env) # <-- FIXED CALL
                return
            
            # 3. Monte Carlo Run
            found_path = train_mc_agent(env, mc_agent, 3*NUM_EPISODES, 'Monte Carlo', optimal_benchmark)
            if found_path is not None:
                print("\n" + "="*50)
                print(f"🏁 **FINAL RESULT:** {mc_agent.__class__.__name__} ne optimal path dhoond liya!")
                print(f"Path Length: {optimal_benchmark}")
                print(f"Path Coordinates: {' -> '.join([str(coord) for coord in found_path])}")
                visualize_results(pre.nim, env) # <-- FIXED CALL
                return
        else:
            # If no benchmark found, run all agents to completion without early exit check
            train_step_rl_agent(env, q_agent, NUM_EPISODES, 'Q-Learning', optimal_benchmark)
            train_step_rl_agent(env, sarsa_agent, 2*NUM_EPISODES, 'SARSA', optimal_benchmark)
            train_mc_agent(env, mc_agent, 3*NUM_EPISODES, 'Monte Carlo', optimal_benchmark)

        # --- 5. Final Comparison Table ---
        
        df = pd.DataFrame.from_dict(ALGO_RESULTS, orient='index')
        df.index.name = 'Algorithm'
        
        print("\n" + "="*50)
        print(f"         ❌ Koi Bhi RL Agent Optimal Path Tak Nahi Pahuncha (Benchmark: {optimal_benchmark}) ❌")
        print("         🏁 Comprehensive Algorithm Comparison 🏁")
        print("="*50)
        df_display = df.sort_values(by='Time', ascending=True).drop(columns=['Path'])
        df_display['Time (s)'] = df_display['Time'].apply(lambda x: f"{x:.6f}")
        df_display = df_display.drop(columns=['Time'])
        print(df_display.to_markdown(numalign="left", stralign="left"))
        print("="*50)

        visualize_results(pre.nim, env) # <-- FIXED CALL

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}\nNOTE: Please ensure you have a maze image file at the specified path: {MAZE_FILE}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_solver()