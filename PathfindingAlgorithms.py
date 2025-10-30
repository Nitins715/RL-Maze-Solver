import heapq
import time
import numpy as np
from MazeEnv import MazeEnv

class PathfindingAlgorithms:
    """
    Implements classical graph search algorithms (Dijkstra's, A*, Bi-Directional Search)
    for comparison against Reinforcement Learning agents.
    """
    
    def __init__(self, env: MazeEnv):
        self.env = env
        self.height = env.height
        self.width = env.width
        self.start = env.start_coords
        self.goal = env.goal_coords
        self.maze = env.maze

    def _reconstruct_path(self, came_from, current, start_node):
        """Reconstructs the path from the 'came_from' dictionary."""
        path = [current]
        # Use the start_node argument to ensure termination for both forward and backward searches
        while current != start_node:
            # Added a defensive check for KeyError, though not strictly needed if logic is perfect
            if current not in came_from:
                 # Should theoretically not happen, but prevents crash if graph is disjoint
                 return None 
            current = came_from[current]
            path.append(current)
        return path[::-1] # Reverse the path to go from start to end

    def dijkstra_search(self):
        """Finds the shortest path using Dijkstra's Algorithm (Guarantees optimality)."""
        start_time = time.time()
        
        priority_queue = [(0, self.start)]
        distances = {node: float('inf') for r in range(self.height) for c in range(self.width) for node in [(r, c)]}
        distances[self.start] = 0
        came_from = {}
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node == self.goal:
                duration = time.time() - start_time
                path = self._reconstruct_path(came_from, current_node, self.start)
                return duration, path

            r, c = current_node
            
            for action_index in self.env.ACTION_SPACE:
                dr, dc = self.env.ACTION_SPACE[action_index]
                neighbor = (r + dr, c + dc)
                
                # Check bounds and walls
                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width):
                    continue
                if self.maze[neighbor] == -1.0: 
                    continue
                
                new_distance = current_distance + 1
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    came_from[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))
        
        return time.time() - start_time, None

    def a_star_search(self):
        """Finds the shortest path using A* Search (Informed search)."""
        start_time = time.time()
        
        h_start = self.env.heuristic(self.start, self.goal)
        priority_queue = [(h_start, self.start)]
        
        g_score = {node: float('inf') for r in range(self.height) for c in range(self.width) for node in [(r, c)]}
        g_score[self.start] = 0
        
        came_from = {}
        
        while priority_queue:
            _, current_node = heapq.heappop(priority_queue) 

            if current_node == self.goal:
                duration = time.time() - start_time
                path = self._reconstruct_path(came_from, current_node, self.start)
                return duration, path

            r, c = current_node
            
            for action_index in self.env.ACTION_SPACE:
                dr, dc = self.env.ACTION_SPACE[action_index]
                neighbor = (r + dr, c + dc)
                
                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width) or self.maze[neighbor] == -1.0:
                    continue
                
                tentative_g_score = g_score[current_node] + 1
                
                if tentative_g_score < g_score.get(neighbor, float('inf')): # Use get for safety
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    h_score = self.env.heuristic(neighbor, self.goal)
                    f_score = tentative_g_score + h_score
                    heapq.heappush(priority_queue, (f_score, neighbor))
        
        return time.time() - start_time, None

    def bi_directional_search(self):
        """
        Performs a search simultaneously from the start and goal states.
        """
        start_time = time.time()

        # Queues: (distance, node)
        q_forward = [(0, self.start)]
        q_backward = [(0, self.goal)]

        # Distances
        dist_forward = {self.start: 0}
        dist_backward = {self.goal: 0}
        
        # Parents
        parent_forward = {self.start: None}
        parent_backward = {self.goal: None}
        
        meeting_point = None
        
        while q_forward and q_backward:
            
            # --- Forward Search Step ---
            current_dist_f, current_node_f = heapq.heappop(q_forward)
            
            # Check for intersection after popping
            if current_node_f in parent_backward:
                meeting_point = current_node_f
                break
            
            r, c = current_node_f
            for dr, dc in self.env.ACTION_SPACE.values():
                neighbor = (r + dr, c + dc)
                
                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width) or self.maze[neighbor] == -1.0:
                    continue

                new_dist_f = current_dist_f + 1
                
                # Check distance against current known shortest path
                if new_dist_f < dist_forward.get(neighbor, float('inf')):
                    dist_forward[neighbor] = new_dist_f
                    parent_forward[neighbor] = current_node_f
                    heapq.heappush(q_forward, (new_dist_f, neighbor))

            # --- Backward Search Step ---
            current_dist_b, current_node_b = heapq.heappop(q_backward)
            
            # Check for intersection after popping
            if current_node_b in parent_forward:
                meeting_point = current_node_b
                break

            r, c = current_node_b
            for dr, dc in self.env.ACTION_SPACE.values():
                neighbor = (r + dr, c + dc)
                
                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width) or self.maze[neighbor] == -1.0:
                    continue

                new_dist_b = current_dist_b + 1
                
                if new_dist_b < dist_backward.get(neighbor, float('inf')):
                    dist_backward[neighbor] = new_dist_b
                    parent_backward[neighbor] = current_node_b
                    heapq.heappush(q_backward, (new_dist_b, neighbor))

        # --- Path Reconstruction ---
        if meeting_point:
            duration = time.time() - start_time
            
            # Reconstruct forward path (Start to Meeting Point)
            path_f = self._reconstruct_path(parent_forward, meeting_point, self.start)
            
            # Reconstruct backward path (Goal back to Meeting Point)
            path_b = self._reconstruct_path(parent_backward, meeting_point, self.goal)
            
            if path_f is None or path_b is None:
                # Should not happen, but defensive check for faulty reconstruction
                return duration, None 

            # Combine: path_f (excluding meeting point) + path_b (reversed, including meeting point)
            # The meeting point is included in path_b[::-1]
            full_path = path_f[:-1] + path_b[::-1] 
            return duration, full_path
        
        return time.time() - start_time, None
