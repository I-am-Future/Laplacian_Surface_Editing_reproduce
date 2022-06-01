# the implementation took reference to https://github.com/luost26/laplacian-surface-editing

import numpy as np
import open3d as o3d
import scipy.sparse.linalg
from lse_utils import *
from scipy.sparse import coo_matrix
from typing import Tuple


def LSE(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh, 
    handle_dx: float, 
    handle_dy: float, 
    handle_dz: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    ''' Laplacian Surface Editing algorithm's wrapper function. 
        Return new [pointcloud, point indices] that are changed. 
    '''

    ## Construct the graph
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    print('Mesh vertices size:', vertices.shape[0])
    print('Mesh faces size:', faces.shape[0])
    graph = PointcloudGraph(vertices.shape[0], vertices)
    for face in faces:
        graph.add_edge(face[0], face[1])
        graph.add_edge(face[0], face[2])
        graph.add_edge(face[1], face[2])
    indices = np.sum(graph.adjacency_matrix>0, axis=1) != 0
    graph.sub_graph_( indices )
    print('New mesh vertices size:', graph.adjacency_matrix.shape[0])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pc_index = np.where(indices == True)[0]
    point_cloud = point_cloud.select_by_index( pc_index )

    ## Ask the user for boundary
    boundary_ctrl_pts = pick_points(point_cloud)
    # boundary_ctrl_pts = [241, 2381, 8082, 4272, 4746]
    boundary_pts = []
    for i in range(len(boundary_ctrl_pts)):
        p1 = boundary_ctrl_pts[i]
        p2 = boundary_ctrl_pts[ (i+1)%len(boundary_ctrl_pts) ]
        path = graph.a_star(p1, p2)
        boundary_pts += path
    boundary_pts = graph.dilate_boundary(boundary_pts, 1)
    display_inlier_outlier(point_cloud, boundary_pts)

    ## Ask the user for handle point
    handle_pt = pick_points(point_cloud)[-1]
    # handle_pt = 4104
    editable_pts = graph.edit_selection(boundary_pts, handle_pt)
    # print(editable_pts)
    display_inlier_outlier(point_cloud, editable_pts)

    ## reconstruct the graph with boundary points and edit points and handle pts
    new_graph_indices = editable_pts + boundary_pts + [handle_pt]
    # print(new_graph_indices)
    glo2sub = {}
    sub2glo = {}  # remember sub nodeids and global nodeid relationship
    n = len(new_graph_indices)
    for i, nid in enumerate(new_graph_indices):
        glo2sub[nid] = i
        sub2glo[i] = nid
    subgraph_adjmat = graph.adjacency_matrix[new_graph_indices, :][:, new_graph_indices]

    Laplacian = np.eye(n) - np.diag(1/np.sum(subgraph_adjmat, axis=1)) @ subgraph_adjmat
    V = np.asarray(point_cloud.points)[new_graph_indices]
    Delta = Laplacian @ V

    ## Construct the Linear System
    A1 = np.zeros((3*n, 3*n))
    A1[0*n:1*n, 0*n:1*n] = (-1) * Laplacian
    A1[1*n:2*n, 1*n:2*n] = (-1) * Laplacian
    A1[2*n:3*n, 2*n:3*n] = (-1) * Laplacian

    ### construct A1 (main editable nodes) node by node
    for nid in range(n):
        neighbor = find_neighbors(subgraph_adjmat, nid)
        k = np.array([nid] + neighbor)
        K_pts = V[k]

        A_i = np.zeros((3*len(k), 7))
        for j in range(len(k)):
            A_i[j, :] =          np.array([K_pts[j, 0], 0          , K_pts[j, 2], -K_pts[j, 1], 1, 0, 0])
            A_i[j+len(k), :] =   np.array([K_pts[j, 1], -K_pts[j, 2],          0,  K_pts[j, 0], 0, 1, 0])
            A_i[j+2*len(k), :] = np.array([K_pts[j, 2], K_pts[j, 1] ,-K_pts[j, 0],           0, 0, 0, 1])

        pinv_A_i = np.linalg.pinv(A_i)
        s = pinv_A_i[0]
        h = pinv_A_i[1:4]

        T_delta = np.vstack([
            Delta[nid,0]*s    - Delta[nid,1]*h[2] + Delta[nid,2]*h[1],
            Delta[nid,0]*h[2] + Delta[nid,1]*s    - Delta[nid,2]*h[0],
           -Delta[nid,0]*h[1] + Delta[nid,1]*h[0] + Delta[nid,2]*s   ,
        ])

        A1[nid, np.hstack([k, k+n, k+2*n])] += T_delta[0]
        A1[nid+n, np.hstack([k, k+n, k+2*n])] += T_delta[1]
        A1[nid+2*n, np.hstack([k, k+n, k+2*n])] += T_delta[2]

    b1 = np.zeros((3*n, 1))

    ### Construct A2 (boundary node)
    A2 = np.zeros((3*len(boundary_pts), 3*n))
    b2 = np.zeros((3*len(boundary_pts), 1))
    boundary_pts_local = [glo2sub[pt] for pt in boundary_pts]
    for idx, pt in enumerate(boundary_pts_local):
        A2[3*idx, pt + 0*n] = 1
        A2[3*idx+1, pt + 1*n] = 1
        A2[3*idx+2, pt + 2*n] = 1
        b2[3*idx] = V[pt, 0]
        b2[3*idx+1] = V[pt, 1]
        b2[3*idx+2] = V[pt, 2]

    ### Construct A3 (handle point)
    A3 = np.zeros((3, 3*n))
    b3 = np.zeros((3, 1))
    A3[0, glo2sub[handle_pt]] = 1
    A3[1, glo2sub[handle_pt]+n] = 1
    A3[2, glo2sub[handle_pt]+2*n] = 1
    b3[0] = V[glo2sub[handle_pt], 0] + handle_dx
    b3[1] = V[glo2sub[handle_pt], 1] + handle_dy 
    b3[2] = V[glo2sub[handle_pt], 2] + handle_dz

    A = np.vstack([A1, A2, A3])
    A = coo_matrix(A)
    b = np.vstack([b1, b2, b3])
    V_p = scipy.sparse.linalg.lsqr(A, b)[0]

    ## Change into answer
    new_pts = np.asarray(point_cloud.points).copy()
    for i in range(n):
        new_pts[sub2glo[i], 0] = V_p[i]
        new_pts[sub2glo[i], 1] = V_p[i + n]
        new_pts[sub2glo[i], 2] = V_p[i + 2*n]

    # print(new_pts.shape)

    return new_pts, new_graph_indices
 

