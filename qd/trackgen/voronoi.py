# python version of src/trackGen/voronoiTrackGenerator.js gives output with same results


import numpy as np
from scipy.spatial import Voronoi
import random
import collections
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set

# Constant
NUMBER_OF_VORONOI_SITES = 100
from alea import prng_alea

class DisjointSet:
    def __init__(self, elements):
        # elements are expected to be indices (integers)
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, element):
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return

        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1

class VoronoiDiagramHelper:
    @staticmethod
    def get_neighbor_indices(site_idx, vor: Voronoi) -> List[int]:
        """
        Returns indices of sites adjacent to the given site_idx,
        ordered clockwise from west (180°) to match JS rhill-voronoi halfedge order.
        """
        import math
        
        # Get all neighbors from ridge_points (adjacency info)
        neighbors = []
        for (site_a, site_b), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            if -1 in (v1, v2):  # Skip infinite edges
                continue
            if site_a == site_idx:
                neighbors.append(site_b)
            elif site_b == site_idx:
                neighbors.append(site_a)
        
        if not neighbors:
            return []
        
        # Order neighbors clockwise from west (180°) to match JS rhill-voronoi ordering
        cell_center = vor.points[site_idx]
        
        def get_angle_from_west(neighbor_idx):
            """Get angle from cell center to neighbor, normalized to start from west (180°) going clockwise."""
            neighbor_pt = vor.points[neighbor_idx]
            dx = neighbor_pt[0] - cell_center[0]
            dy = neighbor_pt[1] - cell_center[1]
            angle = math.atan2(dy, dx)  # Returns angle in radians, range [-pi, pi]
            # Convert to clockwise from west: west=180°, going down to -180°
            # atan2 returns: east=0, north=90°(pi/2), west=180°(pi), south=-90°(-pi/2)
            # JS order is clockwise from west, so we want: 180° -> 90° -> 0° -> -90° -> -180°
            # This is simply descending order of the angle
            return -angle  # Negate to get clockwise order, then sort ascending
        
        # Sort by angle (ascending after negation = clockwise from east)
        # Then rotate to start from west
        neighbors_with_angles = [(n, get_angle_from_west(n)) for n in neighbors]
        neighbors_with_angles.sort(key=lambda x: x[1])
        sorted_neighbors = [n for n, _ in neighbors_with_angles]
        
        # Find the neighbor closest to west (180°) to use as starting point
        def get_raw_angle(neighbor_idx):
            neighbor_pt = vor.points[neighbor_idx]
            dx = neighbor_pt[0] - cell_center[0]
            dy = neighbor_pt[1] - cell_center[1]
            return math.atan2(dy, dx)
        
        # Find where to rotate: we want to start from the neighbor with angle closest to 180° (pi)
        raw_angles = [(i, get_raw_angle(n)) for i, n in enumerate(sorted_neighbors)]
        # Find the one closest to pi (west)
        closest_to_west_idx = min(range(len(raw_angles)), 
                                   key=lambda i: abs(raw_angles[i][1] - math.pi))
        
        # Rotate the list to start from that neighbor
        sorted_neighbors = sorted_neighbors[closest_to_west_idx:] + sorted_neighbors[:closest_to_west_idx]
        
        return sorted_neighbors

    @staticmethod
    def find_site_index_by_coords(x, y, vor: Voronoi) -> Optional[int]:
        # Using numpy for fast lookup
        target = np.array([x, y])
        # Calculate distances to all points
        dists = np.linalg.norm(vor.points - target, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < 1e-5: # Epsilon tolerance
            return int(min_idx)
        return None

class PathFinder:
    @staticmethod
    def find_shortest_path(start_idx, end_idx, vor: Voronoi):
        queue = collections.deque([[start_idx]])
        visited = {start_idx}

        while queue:
            path = queue.popleft()
            current_idx = path[-1]

            if current_idx == end_idx:
                return path

            for neighbor in VoronoiDiagramHelper.get_neighbor_indices(current_idx, vor):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        return None

class VoronoiTrackGenerator:
    def __init__(self, bbox: Dict, seed, track_size, data_set=None, selected_voronoi_sites=None):
        self.bbox = bbox
        self.track_size = track_size
        self.random_gen = prng_alea(seed)
        
        # 1. Prepare Data
        if data_set and len(data_set) > 0:
            self.data_set = np.array([[p['x'], p['y']] for p in data_set])
        else:
            self.data_set = self.generate_points()

        # 2. Compute Voronoi
        self.vor = Voronoi(self.data_set)
        
        self.patch_path = []
        
        # 3. Select Cells (Sites)
        if selected_voronoi_sites and len(selected_voronoi_sites) > 0:
            self.selected_indices = self.sites_from_input(selected_voronoi_sites)
        else:
            self.selected_indices = self.select_cells_for_track(track_size)

        # 4. Find the outline (Track Edges)
        self.track_edges = self.find_track_edges()

    def generate_points(self):
        # Match JS: x = randomGen() * bbox.xr, y = randomGen() * bbox.yb
        points = []
        for _ in range(NUMBER_OF_VORONOI_SITES):
            px = self.random_gen() * self.bbox['xr']
            py = self.random_gen() * self.bbox['yb']
            points.append([px, py])
            
        return np.array(points)

    def sites_from_input(self, points):
        selected_indices = []
        for p in points:
            idx = VoronoiDiagramHelper.find_site_index_by_coords(p['x'], p['y'], self.vor)
            if idx is not None:
                selected_indices.append(idx)
        
        return self.ensure_connected_cells(list(set(selected_indices)))

    def ensure_connected_cells(self, selected_indices):
        current_selection = list(selected_indices)
        disjoint_set = DisjointSet(current_selection)

        for idx in current_selection:
            neighbors = VoronoiDiagramHelper.get_neighbor_indices(idx, self.vor)
            for neighbor in neighbors:
                if neighbor in current_selection:
                    disjoint_set.union(idx, neighbor)

        components = collections.defaultdict(list)
        for idx in current_selection:
            root = disjoint_set.find(idx)
            components[root].append(idx)

        if len(components) > 1:
            component_roots = list(components.keys())
            for i in range(1, len(component_roots)):
                start_node = component_roots[i-1]
                end_node = component_roots[i]
                
                path = PathFinder.find_shortest_path(start_node, end_node, self.vor)
                if path:
                    for cell_idx in path:
                        if cell_idx not in current_selection:
                            current_selection.append(cell_idx)
                            pt = self.vor.points[cell_idx]
                            self.patch_path.append({'x': pt[0], 'y': pt[1]})
                            
        return current_selection

    def select_cells_for_track(self, num_cells):
        center_x = (self.bbox['xr'] + self.bbox['xl']) / 2
        center_y = (self.bbox['yb'] + self.bbox['yt']) / 2
        
        dists = np.sum((self.vor.points - np.array([center_x, center_y]))**2, axis=1)
        start_idx = int(np.argmin(dists))

        selected_indices = []
        current_idx = start_idx
        
        while len(selected_indices) < num_cells and current_idx is not None:
            if current_idx not in selected_indices:
                selected_indices.append(current_idx)
            
            current_idx = self.get_next_cell(current_idx)
            
        return selected_indices

    def get_next_cell(self, cell_idx):
        neighbors = VoronoiDiagramHelper.get_neighbor_indices(cell_idx, self.vor)
        if neighbors:
            return neighbors[int(self.random_gen() * len(neighbors))]
        return None

    def find_track_edges(self):
        # Collect edges in the same order as JS: iterate through selected cells and their polygon edges
        # This matches the JS: selectedCells.flatMap(cell => cell.halfedges.map(halfedge => halfedge.edge))
        all_edges_in_order = []
        edge_count = collections.defaultdict(int)
        
        for site_idx in self.selected_indices:
            region_idx = self.vor.point_region[site_idx]
            region_vertices = self.vor.regions[region_idx]
            
            if -1 in region_vertices or len(region_vertices) == 0:
                continue
            
            n_verts = len(region_vertices)
            for i in range(n_verts):
                v1 = region_vertices[i]
                v2 = region_vertices[(i + 1) % n_verts]
                # Store edge in consistent format: (v1, v2) with actual vertex coords for vb matching
                edge_key = tuple(sorted((v1, v2)))
                # Store the edge with its 'vb' being v2 (end of the edge)
                all_edges_in_order.append((edge_key, v1, v2))
                edge_count[edge_key] += 1

        # External edges are those that appear exactly once (on the boundary)
        external_edge_keys = {edge for edge, count in edge_count.items() if count == 1}
        
        # Build external_edges list preserving order from all_edges_in_order
        # This matches JS which uses edge object identity - first occurrence wins
        seen = set()
        external_edges = []
        for (edge_key, v1, v2) in all_edges_in_order:
            if edge_key in external_edge_keys and edge_key not in seen:
                seen.add(edge_key)
                # Store with 'vb' endpoint (v2) to match JS currentEdge.vb
                external_edges.append((edge_key, v2))

        if not external_edges:
            return []

        # Build adjacency for edge traversal
        adj = collections.defaultdict(list)
        for (edge_key, vb) in external_edges:
            adj[edge_key[0]].append((edge_key, vb))
            adj[edge_key[1]].append((edge_key, vb))

        track_points = []
        
        # Match JS: start from middle of external_edges array
        start_idx = len(external_edges) // 2
        current_edge_key, current_vb = external_edges[start_idx]
        current_vertex_idx = current_vb  # Start from vb of the middle edge
        
        remaining_edge_keys = {ek for (ek, _) in external_edges}
        
        while remaining_edge_keys:
            pt = self.vor.vertices[current_vertex_idx]
            track_points.append({'x': pt[0], 'y': pt[1]})
            
            if current_edge_key in remaining_edge_keys:
                remaining_edge_keys.remove(current_edge_key)
            
            # Find next edge connected to current vertex
            next_edge = None
            for (edge_key, vb) in adj[current_vertex_idx]:
                if edge_key in remaining_edge_keys:
                    next_edge = (edge_key, vb)
                    break
            
            if not next_edge:
                break
                
            current_edge_key, _ = next_edge
            # Move to the other vertex of this edge
            current_vertex_idx = current_edge_key[0] if current_edge_key[1] == current_vertex_idx else current_edge_key[1]

        return track_points

    def to_json(self):
        # 1. Reconstruct 'cells' (Polygons)
        diagram_cells = []
        for i, point in enumerate(self.vor.points):
            region_idx = self.vor.point_region[i]
            region_verts_indices = self.vor.regions[region_idx]
            
            # Skip infinite (-1) or empty regions
            if -1 in region_verts_indices or len(region_verts_indices) == 0:
                continue
                
            vertices = self.vor.vertices[region_verts_indices]
            
            # Mock the 'halfedges' structure
            halfedges_mock = []
            for k in range(len(vertices)):
                v_start = vertices[k]
                v_end = vertices[(k + 1) % len(vertices)]
                halfedges_mock.append({
                    'edge': {
                        'va': {'x': v_start[0], 'y': v_start[1]},
                        'vb': {'x': v_end[0],   'y': v_end[1]}
                    }
                })

            diagram_cells.append({
                'site': {'x': point[0], 'y': point[1], 'voronoiId': i},
                'halfedges': halfedges_mock
            })

        # 2. Reconstruct 'edges' (Grid Lines)
        # The frontend iterates over diagram.edges to draw lines
        diagram_edges = []
        for v_pair in self.vor.ridge_vertices:
            # v_pair is [index_a, index_b]
            # If -1 is present, the edge goes to infinity (skip for now)
            if -1 in v_pair:
                continue
                
            v_a = self.vor.vertices[v_pair[0]]
            v_b = self.vor.vertices[v_pair[1]]
            
            diagram_edges.append({
                'va': {'x': v_a[0], 'y': v_a[1]},
                'vb': {'x': v_b[0], 'y': v_b[1]}
            })

        return {
            'bbox': self.bbox,
            'dataSet': [{'x': p[0], 'y': p[1]} for p in self.data_set],
            # Reconstruct the specific object structure the frontend expects
            'diagram': {
                'cells': diagram_cells,
                'edges': diagram_edges  # <--- This fixes the TypeError
            },
            'trackEdges': self.track_edges,
            'trackSize': self.track_size,
            'selectedCells': [
                {'x': self.vor.points[i][0], 'y': self.vor.points[i][1]} 
                for i in self.selected_indices
            ],
            'patchPath': self.patch_path
        }
