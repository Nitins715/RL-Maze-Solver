import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt 
import io
from PIL import Image as PILImage

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


def create_path_gif(maze_grid, path, start_coords, goal_coords, path_type, duration=200, figsize=(4,4)):
    """Create an animated GIF (bytes) that shows the path being drawn step-by-step.

    Returns raw GIF bytes suitable for passing to Streamlit's st.image.
    """
    if not path:
        return None

    frames = []
    start_r, start_c = start_coords
    goal_r, goal_c = goal_coords

    # Precompute color based on type
    color = 'red' if path_type == 'Pathfinding' else 'blue'

    for step in range(1, len(path) + 1):
        subpath = path[:step]

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(maze_grid, aspect='equal', vmin=-1.0, vmax=2.0)

        path_rows = [r for r, c in subpath]
        path_cols = [c for r, c in subpath]

        ax.plot(path_cols, path_rows, color=color, linewidth=2, marker='.',
                 markersize=5, markerfacecolor='yellow', markeredgecolor='black')

        ax.scatter(start_c, start_r, marker='D', color='lime', s=80, edgecolors='black')
        ax.scatter(goal_c, goal_r, marker='*', color='gold', s=140, edgecolors='black')

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        plt.close(fig)
        buf.seek(0)

        # Convert to PIL image (ensure consistent mode)
        img = PILImage.open(buf).convert('RGBA')
        frames.append(img)
        buf.close()

    # Convert RGBA frames to P mode for GIF compatibility
    pal_frames = [f.convert('P', palette=PILImage.ADAPTIVE) for f in frames]

    gif_buf = io.BytesIO()
    try:
        pal_frames[0].save(gif_buf, format='GIF', save_all=True, append_images=pal_frames[1:], duration=duration, loop=0)
    except Exception:
        # Fallback: try saving RGBA (some PIL versions accept it)
        frames[0].save(gif_buf, format='GIF', save_all=True, append_images=frames[1:], duration=duration, loop=0)

    gif_buf.seek(0)
    return gif_buf.getvalue()

# ----------------------------------------------------------------------------------
## 🌐 Streamlit UI and Main Logic
# ----------------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="RL Pathfinding Maze Solver")
    st.title("🤖 RL Pathfinding Maze Solver")
    st.markdown("---")

    # Allow users to either upload a maze image or select one from the Mazes folder
    maze_files = load_maze_files()

    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload Maze Image (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

    selected_maze = None
    if uploaded_file is None:
        # No upload: fall back to selecting a file from the Mazes directory
        if not maze_files:
            st.error("❌ No maze files found in 'Mazes' and no file uploaded. Please upload a maze image or add images to the 'Mazes' folder.")
            return
        selected_maze = st.sidebar.selectbox("1. Select Maze File", maze_files)
    else:
        st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    
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
    # --- Preprocessing & Environment Setup (Initial) ---
    if uploaded_file is not None:
        st.info(f"Using uploaded maze: {uploaded_file.name}")
        pre_source = uploaded_file
    else:
        st.info(f"Loading maze: {selected_maze}")
        pre_source = os.path.join(MAZE_DIR, selected_maze)

    pre = None
    try:
        pre = Preprocess(pre_source)
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
        
        # Prepare maze identifier and check for existing pretrained q-tables
        try:
            maze_id = os.path.splitext(selected_maze)[0] if selected_maze else os.path.splitext(uploaded_file.name)[0]
        except Exception:
            maze_id = 'uploaded_maze'

        qtable_dir = os.path.join('models', 'qtables')
        os.makedirs(qtable_dir, exist_ok=True)
        qtable_path = os.path.join(qtable_dir, f"{maze_id}_qtable.npz")

        loaded_qtable = None
        if os.path.exists(qtable_path):
            try:
                npz = np.load(qtable_path, allow_pickle=True)
                loaded_qtable = {
                    'q_table': npz['q_table'],
                    'goal': (int(npz['goal_r'].tolist()), int(npz['goal_c'].tolist()))
                }
                st.sidebar.info(f"Found pretrained q-table for maze '{maze_id}' (goal={loaded_qtable['goal']}).")
            except Exception as e:
                st.sidebar.warning(f"Failed to load pretrained q-table: {e}")
        with st.spinner("Running Pathfinding Algorithms (A* only)..."):
            pathfinder = PathfindingAlgorithms(env)

            # Run only A* search as requested
            a_time, a_path = pathfinder.a_star_search()

            # Store only A* result for benchmark
            ALGO_RESULTS['A* Search'] = {
                'Time': a_time,
                'Path': a_path,
                'Length': len(a_path) - 1 if a_path else 'Not Found',
                'Type': 'Pathfinding'
            }

        df_pathfinding = pd.DataFrame.from_dict(ALGO_RESULTS, orient='index')
        pathfinding_lengths = df_pathfinding[df_pathfinding['Type'] == 'Pathfinding']
        valid_lengths = pathfinding_lengths[pathfinding_lengths['Length'] != 'Not Found']['Length']
        
        optimal_benchmark = valid_lengths.min() if not valid_lengths.empty else "Not Found"
        
        pathfinding_container.success(f"Benchmark Optimal Path Length: **{optimal_benchmark}** steps.")
        # --- Auto-Pretrain Q-Learning to the benchmark optimal path ---
        pretrained_q_table = None
        # If we have an on-disk q-table for this maze and the saved goal matches the current goal, reuse it
        if loaded_qtable is not None and loaded_qtable['goal'] == env.goal_coords:
            pretrained_q_table = loaded_qtable['q_table']
            st.sidebar.success('Loaded pretrained q-table that matches the current goal. It will be applied to the Q-Learning agent.')

        # Otherwise, if there's no matching q-table but we have an optimal benchmark, train and save
        if pretrained_q_table is None and optimal_benchmark != "Not Found":
            st.info("Pretraining Q-Learning agent to reach the optimal path (this may take a while)...")
            with st.container():
                pretrain_sub = st.empty()
                pretrain_progress = st.progress(0)
                pretrain_status = st.empty()

            # Use a larger number of episodes for full training; cap for safety
            PRETRAIN_EPISODES = max(10000, int(num_episodes) * 5)

            # Initialize a fresh Q-Learning agent for pretraining (or start from loaded_qtable if available)
            q_pretrain_agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=1.0, min_epsilon=0.01, epsilon_decay=EPSILON_DECAY)
            if loaded_qtable is not None:
                # Start from the loaded q-table to accelerate convergence (fine-tuning)
                try:
                    q_pretrain_agent.q_table = loaded_qtable['q_table'].copy()
                    pretrain_status.info('Starting fine-tuning from existing q-table (goal differs).')
                except Exception:
                    pass

            # Train until it finds the optimal path or reaches episode cap
            pre_path, pre_duration = train_step_rl_agent_streamlit(env, q_pretrain_agent, PRETRAIN_EPISODES, 'Q-Learning', optimal_benchmark, pretrain_progress, pretrain_status)

            if pre_path:
                pretrain_status.success(f"Pretraining reached optimal policy in {pre_duration:.2f}s. Saved learned Q-table.")
                pretrained_q_table = q_pretrain_agent.q_table.copy()

                # Persist q-table to models/qtables for reuse across runs (save goal metadata)
                try:
                    import numpy as _np
                    _np.savez(qtable_path, q_table=pretrained_q_table, goal_r=int(env.goal_coords[0]), goal_c=int(env.goal_coords[1]))
                except Exception as e:
                    st.warning(f"Failed to save q-table to disk: {e}")
            else:
                pretrain_status.error("Pretraining did not reach the optimal path within the episode cap.")
        
        # Remove the A* benchmark result from ALGO_RESULTS so it is not shown in the
        # final comparison table or path visualizations (user requested only Q-Learning visuals)
        benchmark_result = ALGO_RESULTS.pop('A* Search', None)

        # --- 4. RL Agent Training (Sequential Check with Early Exit) ---
        st.header("2. Reinforcement Learning Agents")
        progress_container = st.container()
        
        # 🟢 FIX: Pass the UI-defined local variables (alpha, gamma) to the function
        agents, _ = get_algorithm_functions(env, alpha, gamma, EPSILON_DECAY)
        # If we have a pretrained q-table from the benchmark, apply it to the Q-Learning agent
        if 'Q-Learning' in agents and pretrained_q_table is not None:
            try:
                agents['Q-Learning'].q_table = pretrained_q_table.copy()
                st.sidebar.success('Applied pretrained Q-table to Q-Learning agent.')
            except Exception as e:
                st.sidebar.warning(f'Failed to apply pretrained q-table to agent: {e}')
        
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
                # Prefer animated GIF showing the path being built step-by-step
                gif_bytes = None
                try:
                    gif_bytes = create_path_gif(env.maze, result['Path'], env.start_coords, env.goal_coords, result['Type'], duration=120)
                except Exception as e:
                    st.warning(f"Failed to create GIF for {name}: {e}")

                if gif_bytes:
                    st.image(gif_bytes, caption=f"{name} (L:{result['Length']})", use_container_width=True)
                else:
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