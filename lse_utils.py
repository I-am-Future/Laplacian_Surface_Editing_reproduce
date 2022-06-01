import open3d as o3d
import numpy as np
from typing import Tuple

class PointcloudGraph():
    def __init__(self, n: int, pos: np.ndarray) -> None:
        self.adjacency_matrix = np.zeros((n, n), dtype=np.uint8)
        self.adjacency_distance = np.zeros((n, n), dtype=np.float32)
        assert pos.shape == (n, 3)
        self.pos = pos.astype(np.float32)

    def set_node_pos(self, nodeid: int, pos: np.ndarray) -> None:
        ''' Set a single node's position. '''
        assert pos.shape == (3, ) 
        self.pos[nodeid] = pos
        
    def add_edge(self, node1: int, node2: int) -> None:
        ''' Add an edge between `node1` and `node2`. '''
        self.adjacency_matrix[node1, node2] = 1
        self.adjacency_matrix[node2, node1] = 1
        dis = np.linalg.norm(self.pos[node1] - self.pos[node2]) ** 2
        self.adjacency_distance[node1, node2] = dis
        self.adjacency_distance[node2, node1] = dis

    def sub_graph_(self, index: list) -> None:
        ''' Change this graph into a sub graph with `index` nodes. '''
        self.adjacency_matrix = self.adjacency_matrix[index, :]
        self.adjacency_matrix = self.adjacency_matrix[:, index]
        self.adjacency_distance = self.adjacency_distance[index, :]
        self.adjacency_distance = self.adjacency_distance[:, index]
        self.pos = self.pos[index, :]

    def a_star(self, start: int, end: int) -> list:
        ''' Find a short path from start to end. '''
        from_pos = {}
        op_queue = [] # priority queue [ (nodeid, walk_so_far, expected) ]
        op_queue.append( (start, 0, self.__get_manhattan_dist(start, end)) )
        has_visited = set()
        while len(op_queue) != 0:
            current = min(op_queue, key= lambda x: x[1]+x[2])
            op_queue.remove(current)
            current_id = current[0]
            has_visited.add(current_id)
            if current_id == end:
                route = []
                while current_id != start:
                    current_id = from_pos[current_id]
                    route.append(current_id)
                route.reverse()
                return route

            for new_id in self.__find_neighbors(current_id):
                if (new_id not in has_visited):
                    op_queue.append( (new_id, current[1]+self.adjacency_distance[current_id, 
                            new_id], self.__get_manhattan_dist(new_id, end)) )
                    from_pos[new_id] = current_id

    def __get_manhattan_dist(self, node1: int, node2: int) -> float:
        ''' Utility function for a_star. '''
        pos1, pos2 = self.pos[node1], self.pos[node2]
        return float(np.sum(np.abs(pos1 - pos2)))

    def __find_neighbors(self, current_id: int) -> list:
        ''' Utility function for a_star. '''
        row = self.adjacency_matrix[current_id]
        return np.where(row==1)[0].tolist()

    def dilate_boundary(self, boundaries: list, step: int) -> list:
        ''' Dilate the width of the boundary. '''
        new_pts = set(boundaries)
        for nodeid in boundaries.copy():
            for pt in self.__find_neighbors(nodeid):
                new_pts.add(pt)
        return list(new_pts)

    def edit_selection(self, boundaries: list, handle: int) -> list:
        ''' Get the edit part of the graph. '''
        queue = [handle]
        visited = set()
        while len(queue) != 0:
            current = queue.pop(0)
            visited.add(current)
            if current not in boundaries:
                for nodeid in self.__find_neighbors(current):
                    if nodeid not in visited:
                        queue.append(nodeid)
        return list(visited)



def find_neighbors(mat: np.ndarray, current_id: int) -> list:
    ''' Utility function for a_star. '''
    row = mat[current_id]
    return np.where(row==1)[0].tolist()


def pick_points(pcd) -> None:
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def display_inlier_outlier(cloud, ind: Tuple[list, np.ndarray]) -> None:
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


