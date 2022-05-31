import numpy as np
from lse_utils import *
import open3d as o3d

def LSE_core(mesh):
    ''' Laplacian Surface Editing algorithm's core function. '''

    print(np.asarray(mesh.triangles).shape)
    print(np.asarray(mesh.vertices).shape)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    res = pick_points(point_cloud)
    print(res)
    # o3d.visualization.draw_geometries([mesh])
    # print(mesh)
    # print('Vertices:')
    # print(np.asarray(mesh.vertices))
    # print('Triangles:')
    # print(np.asarray(mesh.triangles))

def LSE(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh, 
    # boundaries: list
    ):
    ''' Laplacian Surface Editing algorithm's wrapper function. '''

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
    print(graph.adjacency_matrix.shape)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pc_index = np.where(indices == True)[0]
    point_cloud = point_cloud.select_by_index( pc_index )

    ## Ask the user for boundary
    # boundary_pts = pick_points(point_cloud)
    boundary_ctrl_pts = [240, 2550, 8063, 4988]
    print(boundary_ctrl_pts)
    boundary_pts = []
    for i in range(len(boundary_ctrl_pts)):
        p1 = boundary_ctrl_pts[i]
        p2 = boundary_ctrl_pts[ (i+1)%len(boundary_ctrl_pts) ]
        path = graph.a_star(p1, p2)
        boundary_pts += path
    boundary_pts = graph.dilate_boundary(boundary_pts, 1)
    print(boundary_pts)
    # display_inlier_outlier(point_cloud, boundary_pts)

    ## Ask the user for handle point
    # handle_pt = pick_points(point_cloud)[-1]
    handle_pt = 4104
    editable_pts = graph.edit_selection(boundary_pts, handle_pt)
    # print(editable_pts)
    # display_inlier_outlier(point_cloud, editable_pts)

    ## reconstruct the graph with boundary points and edit points and handle pts
    new_graph_indices = editable_pts + boundary_pts + [handle_pt]
    # print(new_graph_indices)
    glo2sub = {}
    sub2glo = {}  # remember sub nodeids and global nodeid relationship
    n = len(new_graph_indices)
    print(n)
    for i, nid in enumerate(new_graph_indices):
        glo2sub[nid] = i
        sub2glo[i] = nid
    subgraph_adjmat = graph.adjacency_matrix[new_graph_indices, :]
    subgraph_adjmat = subgraph_adjmat[:, new_graph_indices]

    Laplacian = np.eye(n) - np.diag(1/np.sum(subgraph_adjmat, axis=1)) @ subgraph_adjmat
    # print(Laplacian)
    print(Laplacian.shape)
    V = np.asarray(point_cloud.points)[new_graph_indices]
    Delta = Laplacian @ V
    # print(Delta)
    print(Delta.shape)

    ## Construct the Linear System
    A1 = np.zeros((3*n, 3*n))
    A1[0:n, 0:n] = -Laplacian
    A1[n:2*n, n:2*n] = -Laplacian
    A1[2*n:3*n, 2*n:3*n] = -Laplacian

    # construct A1 one node by one node
    # for nid in range(n):


    b1 = np.zeros((3*n, 3))



mesh = o3d.io.read_triangle_mesh('meshes/bun_zipper_res2.ply')

LSE(mesh)
