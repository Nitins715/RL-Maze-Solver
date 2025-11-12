# RL-Maze-Solver

A small research / demo project comparing classical pathfinding (A*) and tabular Reinforcement Learning (Q-Learning, SARSA, Monte Carlo) on maze images.

This repo contains a Streamlit UI that:
- Accepts a maze image (upload or choose from `Mazes/`).
- Preprocesses the image into a discrete grid for RL and pathfinding.
- Runs A* to compute an optimal benchmark path.
- Pretrains (or fine-tunes) a Q-Learning agent until it reaches the benchmark optimal path, saves the learned Q-table, and applies it to accelerate RL inference (especially useful when using custom start/goal points).
- Allows training/evaluation of RL agents and visualizes their paths alongside the pathfinding result.

## Quick overview of the code
- `app.py` — Streamlit front-end and orchestration (preprocessing, pathfinding, pretraining, RL training, visualization).
- `Preprocess.py` — Loads a maze image (PIL), binarizes and reduces it to a grid, detects start/goal openings and maps values to RL states.
- `MazeEnv.py` — Simple grid environment wrapper for RL; includes action space, reward shaping, state indexing, and an Euclidean heuristic used by A*.
- `PathfindingAlgorithms.py` — Implements classical algorithms (A*, Dijkstra, Bi-directional) used for benchmarking; the Streamlit UI uses A* by default.
- `QLearningAgent.py`, `SARSAAgent.py`, `MCAgent.py` — Tabular RL agents used for training and policy extraction.
- `models/qtables/` — Saved pre-trained Q-tables (.npz) are stored here by maze identifier and include goal metadata.

## How it works (high level)
1. User uploads a PNG/JPG or selects a maze from `Mazes/`.
2. The app runs `Preprocess.generate()` to produce a discrete grid (walls = -1.0, path = 0.0, start = 2.0, goal = 1.0).
3. An A* search runs on the grid to compute an optimal path and its length (benchmark).
4. The app attempts to load a pretrained q-table for that maze (`models/qtables/{maze_id}_qtable.npz`).
   - If a q-table exists and its saved goal matches the current goal, it is applied directly to the Q-Learning agent.
   - If a q-table exists but the goal differs, the q-table is used as initialization and fine-tuned against the current goal (faster than training from scratch).
   - If none exists, the app pretrains a Q-Learning agent until it reaches the benchmark (or a capped number of episodes) and saves the q-table alongside the goal coordinates.
5. The RL agents are then evaluated/trained (Q-Learning is seeded with the pretrained q-table if available), and final paths are displayed.

## Run locally (Windows PowerShell)
1. Create a virtual environment (recommended) and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install required packages (adjust versions as needed):

```powershell
pip install streamlit numpy pandas matplotlib pillow
```

3. Start the Streamlit app (from the project folder):

```powershell
python -m streamlit run app.py
```

4. Open the displayed URL (usually `http://localhost:8501`) in your browser. Upload a maze or pick one from `Mazes/`, then click **RUN COMPARISON**.

Notes:
- Pretraining can be time-consuming for larger mazes. The app caches and saves q-tables to `models/qtables/` so you only need to pretrain once per (maze_id, goal) pair.
- Q-tables are saved as `.npz` files containing `q_table`, `goal_r`, and `goal_c` metadata.

## File formats and identifiers
- `Mazes/` should contain PNG/JPG images of mazes.
- When uploading, the app uses the uploaded file's name as the maze identifier; for selected files it uses the filename (without extension) as the maze id.
- Saved q-tables: `models/qtables/{maze_id}_qtable.npz` with fields:
  - `q_table` — numpy array (num_states x num_actions)
  - `goal_r`, `goal_c` — stored goal row/col for quick matching

## Using custom start/goal
- Toggle **Use Custom Start/Goal Points** in the sidebar and enter coordinates (row, col).
- If the pretrained q-table's goal matches the environment goal, the pretrained policy will be applied and inference should be fast.
- If the goals differ, the app will fine-tune from any available q-table (if present) or pretrain from scratch.

## Suggested improvements (low-risk → higher effort)
- Add a `requirements.txt` or `pyproject.toml` to pin dependencies and simplify setup.
- Add an explicit UI toggle to choose behavior when a saved q-table exists for the same maze but with a different goal:
  - `Exact-match only` (use q-table only if goal matches), or
  - `Fine-tune if different` (initialize from saved q-table and fine-tune for new goal).
- Add an option to persist the Q-table under a user-specified name and to load any arbitrary q-table from disk.
- Provide progress checkpoints and ETA during long pretraining runs (e.g., log episodes-per-second and expected time).
- Add unit tests for preprocessing, pathfinding, and the environment wrapper (fast checks that small mazes produce expected results).
- Support different A* heuristics (Manhattan, Chebyshev) selectable in the UI.
- Add a CLI mode for batch processing many mazes (headless) and produce CSV summaries of runtime and path lengths.
- Visual improvements: interactive path stepping, heatmap of visitation counts, and per-step Q-values overlay.
- Performance: convert heavy loops to vectorized NumPy where possible; for very large mazes consider a sparse representation.

## Troubleshooting
- If preprocessing fails on upload, check the image is a clean maze (not extremely noisy). You may need to tweak `Preprocess.generate()` parameters (`margin`, `pix`, `div`) for unusual images.
- If the app runs out of memory or is very slow during pretraining, reduce `PRETRAIN_EPISODES` in `app.py` or use smaller mazes.

