import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt 

# --- Import your core modules ---
# Ensure these files are accessible in the same directory or Python path
from ComparisonEngine import NUM_EPISODES
from Preprocess import Preprocess
from MazeEnv import MazeEnv
from QLearningAgent import QLearningAgent
from SARSAAgent import SARSAAgent
from MCAgent import MCAgent 
from PathfindingAlgorithms import PathfindingAlgorithms

# --- Configuration (Copied from ComparisonEngine for setup) ---
MAX_STEPS_PER_EPISODE = 1000 
ALPHA = 0.1     
GAMMA = 0.95    
EPSILON_DECAY = 0.9999
MAZE_DIR = "Mazes"

# --- Utility Functions (Adapted for Streamlit) ---

@st.cache_data
def load_maze_files():
    """Finds all PNG files in the Mazes directory."""
    if not os.path.exists(MAZE_DIR):
        return []
    return [f for f in os.listdir(MAZE_DIR) if f.endswith('.png')]

def get_algorithm_functions(env, alpha, gamma, epsilon_decay):
    """
    Initializes and returns the RL agents using UI-provided hyperparameters.
    """
    agents = {
        'Q-Learning': QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=1.0, min_epsilon=0.01, epsilon_decay=epsilon_decay),
        'SARSA': SARSAAgent(env, alpha=alpha, gamma=gamma, epsilon=1.0, min_epsilon=0.01, epsilon_decay=epsilon_decay),
        'Monte Carlo': MCAgent(env, alpha=alpha, gamma=gamma, epsilon=1.0, min_epsilon=0.01, epsilon_decay=epsilon_decay) 
    }
    pathfinder = PathfindingAlgorithms(env)
    
    return agents, pathfinder

def train_step_rl_agent_streamlit(env, agent, num_episodes, agent_name, optimal_benchmark_length, progress_bar, status_text):
    """
    Adapted training function for step-based RL agents with Streamlit updates.
    """
    start_time = time.time()
    check_frequency = max(1, num_episodes // 100)
    
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
            
            if agent_name == 'Q-Learning':
                agent.update_q_table(old_state, old_action, reward, new_state)
            elif agent_name == 'SARSA':
                agent.update_q_table(old_state, old_action, reward, new_state, next_action)
            
            current_state = new_state
            action = next_action
            steps += 1
            
        agent.decay_epsilon()
        
        # Streamlit update logic
        if (episode + 1) % check_frequency == 0:
            progress = (episode + 1) / num_episodes
            progress_bar.progress(progress)
            status_text.text(f"{agent_name} training: Episode {episode + 1}/{num_episodes} (Epsilon: {agent.epsilon:.4f})")

            # Early Exit Check 
            if optimal_benchmark_length != "Not Found":
                path = agent.get_optimal_path()
                path_length = len(path) - 1 if path else "Not Found"
                if path_length == optimal_benchmark_length:
                    return path, time.time() - start_time 
        
    path = agent.get_optimal_path()
    path_length = len(path) - 1 if path else "Not Found"
    return path, time.time() - start_time


def train_mc_agent_streamlit(env, agent: MCAgent, num_episodes, agent_name, optimal_benchmark_length, progress_bar, status_text):
    """
    Adapted training function for Monte Carlo agent with Streamlit updates.
    """
    start_time = time.time()
    check_frequency = max(1, num_episodes // 100)
    
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
        
        # Streamlit update logic
        if (episode + 1) % check_frequency == 0:
            progress = (episode + 1) / num_episodes
            progress_bar.progress(progress)
            status_text.text(f"{agent_name} training: Episode {episode + 1}/{num_episodes} (Epsilon: {agent.epsilon:.4f})")

            # Early Exit Check
            if optimal_benchmark_length != "Not Found":
                path = agent.get_optimal_path()
                path_length = len(path) - 1 if path else "Not Found"
                if path_length == optimal_benchmark_length:
                    return path, time.time() - start_time 

    path = agent.get_optimal_path()
    path_length = len(path) - 1 if path else "Not Found"
    return path, time.time() - start_time


def plot_path_on_maze(maze_grid, path, start_coords, goal_coords, title, path_type):
    """Generates a matplotlib figure for display in Streamlit."""
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(maze_grid, aspect='equal', vmin=-1.0, vmax=2.0)
    
    if path:
        path_rows = [r for r, c in path]
        path_cols = [c for r, c in path]
        
        color = 'red' if path_type == 'Pathfinding' else 'blue'
        
        ax.plot(path_cols, path_rows, color=color, linewidth=2, marker='.', 
                 markersize=5, markerfacecolor='yellow', markeredgecolor='black')
    
    # Highlight custom Start and Goal points
    start_r, start_c = start_coords
    goal_r, goal_c = goal_coords
    ax.scatter(start_c, start_r, marker='D', color='lime', s=100, label='Start', edgecolors='black')
    ax.scatter(goal_c, goal_r, marker='*', color='gold', s=200, label='Goal', edgecolors='black')

    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    st.pyplot(fig)

# ----------------------------------------------------------------------------------
## 🌐 Streamlit UI and Main Logic
# ----------------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="RL vs Pathfinding Maze Solver")
    st.title("🤖 RL vs Pathfinding Maze Solver Comparison")
    st.markdown("---")

    maze_files = load_maze_files()
    if not maze_files:
        st.error("❌ Maze files not found. Please ensure your maze images are in a folder named 'Mazes'.")
        return

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    selected_maze = st.sidebar.selectbox("1. Select Maze File", maze_files)
    
    # Initialize variables for scoping and sidebar input capture
    num_episodes = NUM_EPISODES
    alpha = ALPHA
    gamma = GAMMA

    col_rl_config, col_episodes = st.sidebar.columns(2)
    with col_episodes:
        num_episodes = st.number_input("Episodes (Q-L/SARSA Base)", value=NUM_EPISODES, min_value=1000, step=1000)
    with col_rl_config:
        alpha = st.number_input("Learning Rate (Alpha)", value=ALPHA, min_value=0.01, max_value=1.0, step=0.01, format="%.2f")
        gamma = st.number_input("Discount Factor (Gamma)", value=GAMMA, min_value=0.01, max_value=1.0, step=0.01, format="%.2f")
    
    st.sidebar.markdown("---")
    
    # --- Preprocessing & Environment Setup (Initial) ---
    st.info(f"Loading maze: {selected_maze}")
    maze_path = os.path.join(MAZE_DIR, selected_maze)
    pre = Preprocess(maze_path)
    try:
        pre.generate(margin=0.005, pix=0)
        env_temp = MazeEnv(pre)
    except Exception as e:
        st.error(f"Error during maze preprocessing/initial setup: {e}")
        return

    # --- Custom Point Selection ---
    st.sidebar.header("2. Custom Start/Goal")
    use_custom = st.sidebar.checkbox("Use Custom Start/Goal Points", False)
    
    custom_start, custom_goal = env_temp.start_coords, env_temp.goal_coords
    
    if use_custom:
        st.sidebar.markdown("Enter coordinates (Row, Col) based on the grid structure.")
        
        col_start, col_goal = st.sidebar.columns(2)
        with col_start:
            start_r = st.number_input("Start Row", value=env_temp.start_coords[0], min_value=0, max_value=env_temp.height-1)
            start_c = st.number_input("Start Col", value=env_temp.start_coords[1], min_value=0, max_value=env_temp.width-1)
            custom_start = (start_r, start_c)
        with col_goal:
            goal_r = st.number_input("Goal Row", value=env_temp.goal_coords[0], min_value=0, max_value=env_temp.height-1)
            goal_c = st.number_input("Goal Col", value=env_temp.goal_coords[1], min_value=0, max_value=env_temp.width-1)
            custom_goal = (goal_r, goal_c)

    # --- Final Environment Setup ---
    try:
        env = MazeEnv(pre, start_coords=custom_start, goal_coords=custom_goal)
    except Exception as e:
        st.error(f"Error setting up environment with coordinates: {e}")
        return
        
    st.sidebar.success(f"Path: S{env.start_coords} ➡️ G{env.goal_coords}")
    st.sidebar.markdown("---")

    # --- Main Execution Button ---
    if st.sidebar.button("🚀 RUN COMPARISON", type="primary"):
        # Initialize ALGO_RESULTS (using a local dict is safer for Streamlit reruns)
        ALGO_RESULTS = {}
        
        # --- 3. Pathfinding Solver (Benchmark Set) ---
        st.header("1. Pathfinding Algorithms (Benchmark)")
        pathfinding_container = st.container()
        
        with st.spinner("Running Pathfinding Algorithms..."):
            pathfinder = PathfindingAlgorithms(env)
            
            # 🟢 FIX: Capture the returned values (time, path)
            d_time, d_path = pathfinder.dijkstra_search()
            a_time, a_path = pathfinder.a_star_search()
            b_time, b_path = pathfinder.bi_directional_search()
            
            # Store results using the captured local variables
            ALGO_RESULTS['Dijkstra’s'] = {'Time': d_time, 'Path': d_path, 'Length': len(d_path) - 1 if d_path else 'Not Found', 'Type': 'Pathfinding'}
            ALGO_RESULTS['A* Search'] = {'Time': a_time, 'Path': a_path, 'Length': len(a_path) - 1 if a_path else 'Not Found', 'Type': 'Pathfinding'}
            ALGO_RESULTS['Bi-Directional Search'] = {'Time': b_time, 'Path': b_path, 'Length': len(b_path) - 1 if b_path else 'Not Found', 'Type': 'Pathfinding'}

        df_pathfinding = pd.DataFrame.from_dict(ALGO_RESULTS, orient='index')
        pathfinding_lengths = df_pathfinding[df_pathfinding['Type'] == 'Pathfinding']
        valid_lengths = pathfinding_lengths[pathfinding_lengths['Length'] != 'Not Found']['Length']
        
        optimal_benchmark = valid_lengths.min() if not valid_lengths.empty else "Not Found"
        
        pathfinding_container.success(f"Benchmark Optimal Path Length: **{optimal_benchmark}** steps.")
        
        
        # --- 4. RL Agent Training (Sequential Check with Early Exit) ---
        st.header("2. Reinforcement Learning Agents")
        progress_container = st.container()
        
        # 🟢 FIX: Pass the UI-defined local variables (alpha, gamma) to the function
        agents, _ = get_algorithm_functions(env, alpha, gamma, EPSILON_DECAY)
        
        rl_order = ['Q-Learning', 'SARSA', 'Monte Carlo']
        
        early_exit_path = None
        
        for name in rl_order:
            agent = agents[name]
            
            # Use the local variable 'num_episodes' from the sidebar input
            episodes = num_episodes if name in ['Q-Learning'] else num_episodes * 2
            if name == 'Monte Carlo': episodes = num_episodes * 3 
            
            with progress_container:
                st.subheader(f"Training: {name}")
                progress_bar = st.progress(0)
                status_text = st.empty()

            if name in ['Q-Learning', 'SARSA']:
                path, duration = train_step_rl_agent_streamlit(env, agent, episodes, name, optimal_benchmark, progress_bar, status_text)
            else:
                path, duration = train_mc_agent_streamlit(env, agent, episodes, name, optimal_benchmark, progress_bar, status_text)
            
            # Record result
            ALGO_RESULTS[name] = {'Time': duration, 'Path': path, 'Length': len(path) - 1 if path else 'Not Found', 'Type': 'RL'}
            
            status_text.success(f"✅ {name} finished in {duration:.4f}s. Path Length: {ALGO_RESULTS[name]['Length']}")
            progress_bar.progress(1.0)

            # Check for Early Exit *condition* if ALGO_RESULTS[name]['Length'] == optimal_benchmark:
            if not early_exit_path:
                early_exit_path = ALGO_RESULTS[name]['Path']
                st.success(f"🏆 {name} ACHIEVED OPTIMAL PATH! Stopping further RL training.")
                    # If you want to fully stop the loop here, uncomment the line below:
                break 

        # --- 5. Final Comparison and Validation ---
        st.header("3. Results and Comparison")
        
        df = pd.DataFrame.from_dict(ALGO_RESULTS, orient='index')
        
        # Validation
        if early_exit_path:
            st.markdown(f"## 🎉 Validation Successful!")
            st.markdown(f"The fastest optimal RL algorithm found the shortest path ({optimal_benchmark} steps).")
        elif optimal_benchmark == "Not Found":
            st.warning("⚠️ Pathfinding failed. Cannot guarantee RL optimality.")
        else:
            shortest_rl_length = df[df['Type'] == 'RL']['Length'].min()
            if shortest_rl_length == optimal_benchmark:
                 st.success("✅ One or more RL agents converged to the optimal path.")
            else:
                 st.error(f"❌ RL agents were sub-optimal. Fastest RL path: {shortest_rl_length} vs. Optimal: {optimal_benchmark}.")

        # Comparison Table
        st.subheader("Comprehensive Algorithm Comparison Table")
        df_display = df.sort_values(by='Time', ascending=True).drop(columns=['Path'])
        df_display['Time (s)'] = df_display['Time'].apply(lambda x: f"{x:.6f}")
        df_display = df_display.drop(columns=['Time'])
        st.dataframe(df_display)

        # Visualization
        st.subheader("Path Visualization")
        cols = st.columns(len(ALGO_RESULTS))
        
        for i, (name, result) in enumerate(ALGO_RESULTS.items()):
            with cols[i]:
                plot_path_on_maze(
                    env.maze, 
                    result['Path'], 
                    env.start_coords, 
                    env.goal_coords, 
                    f"{name} (L:{result['Length']})",
                    result['Type']
                )


if __name__ == "__main__":
    main()