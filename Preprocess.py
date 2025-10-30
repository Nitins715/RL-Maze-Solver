# http://www.mazegenerator.net/
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Preprocess: 
    """ 
    RL State Mapping:
    0: Path (Traversable, usually low/zero reward)
    -1: Wall (Obstacle, high negative reward)
    1: Goal/Exit (High positive reward)
    2: Start/Entry (or another distinct positive value for visualization)
    """

    def __init__(self, file_path):
        # 1. Image Loading and Grayscale Conversion
        try:
            self.im = Image.open(file_path).convert('L') 
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Maze image file not found at '{file_path}'")
            
        self.w, self.h = self.im.size
        self.nim = None
        # Stores the start and end coordinates: [sr, sc, gr, gc]
        self.loc = None 

    def generate(self, margin=0.005, pix=0, div=128):
        # 2. Binarization: Convert to binary image.
        binary = self.im.point(lambda p: p < div)

        # 3. Convert to Numpy array: Walls -> 1 (True), Paths -> 0 (False).
        self.nim = np.array(binary)
        
        # 4. Core transformation: Reduce high-res maze to minimal grid. (OPTIMIZED)
        self._reduce_matrix(margin)
        
        # 5. Map values to RL states: Walls (1) -> -1.0, Path (0) -> 0.0.
        self._invert_and_map()
        
        # 6. Trim the borders (if pix > 0).
        if pix > 0:
            self._trim(pix)
        
        # 7. Detect Start/Goal points.
        self.loc = self._detect_openings()
        
        # 8. Mark Start (2) and Goal (1).
        if self.loc and len(self.loc) == 4:
            self.nim[self.loc[0], self.loc[1]] = 2.0  # Start (Entry)
            self.nim[self.loc[2], self.loc[3]] = 1.0  # Goal (Exit)

    def _reduce_matrix(self, margin): 
        """Vectorized approach for collapsing redundant rows/cols."""
        # Determine the tolerance for differences
        if isinstance(margin, (tuple, list)):
            row_tol = self.w * margin[0]
            col_tol = self.h * margin[1]
        else:
            row_tol = self.w * margin
            col_tol = self.h * margin
            
        # --- Row Reduction ---
        row_diffs = np.sum(np.abs(np.diff(self.nim, axis=0)), axis=1)
        r_indices_to_keep = np.where(row_diffs > row_tol)[0] + 1
        r_indices = np.concatenate(([0], r_indices_to_keep))
        self.nim = self.nim[r_indices, :]
        self.h = self.nim.shape[0]

        # --- Column Reduction ---
        col_diffs = np.sum(np.abs(np.diff(self.nim, axis=1)), axis=0)
        c_indices_to_keep = np.where(col_diffs > col_tol)[0] + 1
        c_indices = np.concatenate(([0], c_indices_to_keep))
        self.nim = self.nim[:, c_indices]
        self.w = self.nim.shape[1]

    def _invert_and_map(self):
        """Maps boolean to RL states: Walls (1) -> -1.0, Path (0) -> 0.0."""
        self.nim = self.nim.astype(np.float32)
        self.nim[self.nim == 1] = -1.0
        
    def _trim(self, pix):
        """Removes 'pix' pixels from all four sides."""
        self.nim = self.nim[pix:-pix, pix:-pix]
        self.h -= 2 * pix
        self.w -= 2 * pix

    def _detect_openings(self):
        """Detects the first two open paths (0.0) on the perimeter."""
        l = [] # [start_r, start_c, goal_r, goal_c]

        # Check Top and Bottom rows
        for r in [0, self.h - 1]:
            open_cols = np.where(self.nim[r, :] == 0.0)[0]
            if open_cols.size > 0:
                l.extend([r, open_cols[0]]) 
                if len(l) == 4: return l

        # Check Left and Right columns (excluding corners)
        for c in [0, self.w - 1]:
            rows_to_check = np.arange(1, self.h - 1) 
            open_rows = rows_to_check[np.where(self.nim[rows_to_check, c] == 0.0)[0]]
            
            if open_rows.size > 0:
                l.extend([open_rows[0], c])
                if len(l) == 4: return l

        return l

    def show(self, aspect=1):
        """Displays the processed grid using matplotlib's default color scheme."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.nim, aspect=aspect, vmin=-1.0, vmax=2.0) 
        
        plt.colorbar(label='RL State Value (-1=Wall, 0=Path, 1=Goal, 2=Start)')
        plt.title('Preprocessed Maze Grid for RL (Default Matplotlib Colors)')
        plt.xticks([]), plt.yticks([])
        plt.grid(color='white', linewidth=0.5)
        plt.show()

    def __str__(self):
        return '\n'.join(' '.join(f"{val:3.0f}" for val in row) for row in self.nim)


if __name__ == "__main__":
    try:
        # NOTE: Update the file path as needed.
        p = Preprocess("Mazes/22 by 25 maze.png") 
        p.generate(margin=0.005, pix=0) 
        print(f"Original Size: {p.im.size} | RL Grid Size: {p.h}x{p.w}")
        print(f"Start/Goal Location (r, c): Start({p.loc[0]}, {p.loc[1]}), Goal({p.loc[2]}, {p.loc[3]})")
        p.show()
    except Exception as e:
        print(e)
