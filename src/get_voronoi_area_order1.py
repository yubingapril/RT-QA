### Author: Zhang Yipeng. Email: yipeng001@e.ntu.edu.sg

import numpy as np
from scipy.spatial import ConvexHull,HalfspaceIntersection
import os 
from joblib import Parallel,delayed

def find_convex_hull_boundary(points):
    """Given a set of points, return which points lie on the convex hull boundary"""
    hull = ConvexHull(points)
    boundary_indices = np.unique(hull.vertices)
    return boundary_indices



def circumcenter(a, b, c, d):
    """Compute the center of the circumscribed sphere for four points in 3D space"""
    A = np.vstack((b - a, c - a, d - a))
    bvec = np.array([np.dot(b-a, b-a),
                     np.dot(c-a, c-a),
                     np.dot(d-a, d-a)])


    circ_center = np.linalg.solve(2 * A, bvec) + a
    return circ_center


def centroid_and_normal(p1, p2, p3):
    """Given three points, compute the centroid and the direction perpendicular to the plane"""
    centroid = (p1 + p2 + p3) / 3
    normal_vector = np.cross(p2 - p1, p3 - p1)
    return centroid, normal_vector


def point_plane_position(point, plane_eq):
    """
    Determine the position of a point relative to a plane (on the plane, on one side, or on the opposite side).
    Returns:
    - Positive value: point is on one side of the plane
    - Negative value: point is on the opposite side of the plane
    - Zero: point lies on the plane
    """
    a, b, c, d = plane_eq
    return a * point[0] + b * point[1] + c * point[2] + d


def line_plane_intersection(p1, p2, plane_eq):
    """
    Compute the intersection point of the line segment (p1, p2) with a plane.
    Returns the coordinates of the intersection point.
    """
    u = point_plane_position(p1, plane_eq)
    v = point_plane_position(p2, plane_eq)
    if (u == 0) and (v == 0):
        return ['Both On Plane']
    t = u / (u - v)  # Calculate the interpolation coefficient t
    intersection_point = p1 + t * (p2 - p1)
    return intersection_point


from shapely.geometry import Polygon, Point

def calculate_voronoi_area(voronoi_points, edge, radius):
    """
    Compute the area of intersection between the convex hull formed by Voronoi points
    and a circle centered at the projection of a given point onto the plane.

    Parameters:
        voronoi_points: ndarray
            3D coordinates of Voronoi points (N, 3).
        edge: list of lists
            List of point indices, e.g., [[0, 1], [1, 2]].
        radius: float
            Radius of the circle.

    Returns:
        float
            Area of the intersection region.
    """


    def sort_points_by_norm(voronoi_points):
        norms = np.linalg.norm(voronoi_points, axis=1)
        sorted_indices = np.argsort(norms)
        sorted_points = voronoi_points[sorted_indices]
        return sorted_points

    global coords
    # Step 1: Fit the plane defined by the Voronoi points
    voronoi_points = sort_points_by_norm(np.array(voronoi_points))
    origin_point = voronoi_points[0]
    p1, p2, p3 = voronoi_points[:3]
    v1 = p2 - p1
    v2 = p3 - p1
    normal_vector = np.cross(v1, v2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Construct a local 2D coordinate system
    basis_x = np.cross(normal_vector, [1, 0, 0])
    if np.allclose(basis_x, 0):
        basis_x = np.cross(normal_vector, [0, 1, 0])
    basis_x = basis_x / np.linalg.norm(basis_x)
    basis_y = np.cross(normal_vector, basis_x)
    basis_y = basis_y / np.linalg.norm(basis_y)

    # Step 2: Project Voronoi points onto the fitted plane to obtain 2D coordinates
    def project_to_plane(points, basis_x, basis_y, origin):
        local_coords = np.dot(points - origin, np.vstack([basis_x, basis_y]).T)
        return local_coords

    plane_points_2d = project_to_plane(voronoi_points, basis_x, basis_y, origin_point)

    # Step 3: Extract all points from edge indices and project them onto the plane
    unique_indices = set(index for pair in edge for index in pair)
    relevant_coords = np.array([coords[i] for i in unique_indices])
    projected_coords = project_to_plane(relevant_coords, basis_x, basis_y, origin_point)

    # Step 4: Calculate the distance from points to the plane
    def point_to_plane_distance(point, normal, point_on_plane):
        normal = normal / np.linalg.norm(normal)
        return np.abs(np.dot(point - point_on_plane, normal))

    distances = [point_to_plane_distance(coords[i], normal_vector, origin_point) for i in unique_indices]

    # Step 5: Construct the convex hull of the Voronoi points
    hull = ConvexHull(plane_points_2d)
    hull_polygon = Polygon(plane_points_2d[hull.vertices])

    # Step 6: Construct a circle and compute the intersection area
    circles = []
    for coord_2d, distance in zip(projected_coords, distances):
        if radius ** 2 - distance ** 2 > 0:
            effective_radius = np.sqrt(radius ** 2 - distance ** 2)
        else:
            effective_radius = 0

        circle = Point(coord_2d).buffer(effective_radius)
        circles.append(circle)

    if not circles:
        return 0.0
    intersection = hull_polygon
    for circle in circles:
        intersection = intersection.intersection(circle)
    intersection_area = intersection.area

    return intersection_area


def rhomboid_reader(filename):
    r_fslices_set = []
    current_slice = 0
    max_dim = 0
    with open(filename+'_fslices.txt', 'r') as file:
        for line in file:
            line = line.strip()  

          
            if line.startswith('Slice'):
                current_slice = int(line.split(' ')[1].replace(':', ''))
            elif line:  
                tuple_data = eval(line) 
                r_fslices_set.append([tuple_data[0], current_slice, tuple_data[1]])
    return r_fslices_set


import numpy as np
from scipy.spatial import ConvexHull

def get_on_S_indices(simplex):
    sets = [set(sublist) for sublist in simplex]
    union_set = set.union(*sets)
    intersection_set = set.intersection(*sets)
    on_s_set = union_set - intersection_set
    on_s_list = list(on_s_set)
    return on_s_list

def get_in_S_indices(simplex):
    sets = [set(sublist) for sublist in simplex]
    intersection_set = set.intersection(*sets)
    in_s_list = list(intersection_set)
    return in_s_list

def get_voronoi_area(file,file_coor,file_slice,out_file, input_max_order, radius):
    global coords
    coords = np.loadtxt(file_coor)

    IDs=[]
    with open(file,'r') as f:
            for line in f:
                columns=line.split()
                if columns:
                    IDs.append(columns[0])  

    r_fslices_set = []
    with open(file_slice, 'r') as file:
        for line in file:
            line = line.strip()  
            if line.startswith('Slice'):
                current_slice = int(line.split(' ')[1].replace(':', ''))
            elif line: 
                tuple_data = eval(line)
                r_fslices_set.append([tuple_data[0], current_slice, tuple_data[1]])

    max_order = r_fslices_set[-1][1]
    if max_order < input_max_order:
        print('The input max order is bigger than what we have in the fslice file')
        max_order = input_max_order
    else:
        max_order = input_max_order

    vertices = []
    edges = []
    triangles = []
    tetrahedrons = []
    for i in range(max_order):
        vertices.append([])
        edges.append([])
        triangles.append([])
        tetrahedrons.append([])
    for line in r_fslices_set:
        simplex = line[0]
        order = int(line[1])
        if order == max_order:
            if len(simplex) == 1:
                vertices[order-1].append(simplex)
            if len(simplex) == 2:
                edges[order-1].append(simplex)
            if len(simplex) == 3:
                triangles[order-1].append(simplex)
            if len(simplex) >= 4:
                tetrahedrons[order-1].append(simplex)

    dual_area = dict()
    is_area_boundary = dict()
    for i in [max_order-1]:
        dual_coords = dict()
        for tetrahedron in tetrahedrons[i]:
            on_s_list = get_on_S_indices(tetrahedron)
            if len(on_s_list) == 4:
                v1 = coords[on_s_list[0]]
                v2 = coords[on_s_list[1]]
                v3 = coords[on_s_list[2]]
                v4 = coords[on_s_list[3]]
                dual_coord = circumcenter(v1, v2, v3, v4)
                dual_coords[str(tetrahedron)] = dual_coord

            else:
                print('coords are not in general position')

        dual_edges = dict()
        for triangle in triangles[i]:
            on_s_list = get_on_S_indices(triangle)
            if len(on_s_list) == 3:
                triangle_set = {tuple(t) for t in triangle}
                nodes_in_edge = []
                for tetrahedron in tetrahedrons[i]:
                    tetrahedron_set = {tuple(t) for t in tetrahedron}
                    if triangle_set.issubset(tetrahedron_set):
                        nodes_in_edge.append(dual_coords[str(tetrahedron)])
                if len(nodes_in_edge) == 2:
                    dual_edges[str(triangle)] = nodes_in_edge

                if len(nodes_in_edge) == 1:
                    centroid, backup_normal = centroid_and_normal(coords[on_s_list[0]], coords[on_s_list[1]], coords[on_s_list[2]])
                    if np.linalg.norm(centroid - nodes_in_edge[0]) > 0:
                        normal = (centroid - nodes_in_edge[0])/np.linalg.norm(centroid - nodes_in_edge[0])
                    else:
                        normal = backup_normal
        # 10000 below is a big number to make sure nodes_in_edge[0] + 10000*normal is outside of given boundary,
        # please change it to a suitable number if it's not large enough
                    inside_points = get_in_S_indices(triangle)
                    if (len(inside_points) == 0):
                        nodes_in_edge.append(nodes_in_edge[0] + 10000 * normal)
                    else:
                        if (np.linalg.norm(nodes_in_edge[0] + 10000 * normal - coords[on_s_list[0]]) >= np.linalg.norm(nodes_in_edge[0] + 10000 * normal - coords[inside_points[0]])):
                            nodes_in_edge.append(nodes_in_edge[0] + 10000 * normal)
                        else:
                            nodes_in_edge.append(nodes_in_edge[0] - 10000 * normal)
                    dual_edges[str(triangle)] = nodes_in_edge

            else:
                print('Something is wrong about triangle:',triangle)
        
        log_entries=[]
        for edge in edges[i]:
            e1=edge[0][0];e2=edge[1][0]
            ##skip edge with same chain
            if IDs[e1][2]==IDs[e2][2]:
                continue

            dual_bound_edges = []
            on_s_list = get_on_S_indices(edge)
            if len(on_s_list) == 2:
                edge_set = {tuple(t) for t in edge}
                for triangle in triangles[i]:
                    triangle_set = {tuple(t) for t in triangle}
                    if edge_set.issubset(triangle_set):
                        dual_bound_edges.append(dual_edges[str(triangle)])

                inner_voronoi_points = set()
                for start, end in dual_bound_edges:
                    inner_voronoi_points.add(tuple(start))
                    inner_voronoi_points.add(tuple(end))
                inner_voronoi_points = list(inner_voronoi_points)
                inner_voronoi_points = [np.array(point) for point in inner_voronoi_points]
                area = calculate_voronoi_area(inner_voronoi_points, edge, radius)
                log_entries.append(f"{IDs[e1]}\t{IDs[e2]} {area}\n")
                is_area_boundary[str(edge)] = False
            else:
                print('something weird about edge:', edge)
        with open(out_file, 'w') as log_file:
            log_file.writelines(log_entries)
    return dual_area




def cal_area(model_name,atom_coor_dir,slice_dir,out_dir):
    try:
        os.makedirs("./tmp",exist_ok=True)
        
        file = os.path.join(atom_coor_dir,model_name+'.txt')
        file_coor=f"./tmp/{model_name}.txt"
        commend="awk '{print $2,$3,$4}' "+file+">"+file_coor
        os.system(commend)
        file_slice=os.path.join(slice_dir,model_name+'.txt')
        out_file=os.path.join(out_dir,model_name+'.txt')
        max_order = 1
        radius = 3.4

        get_voronoi_area(file,file_coor,file_slice,out_file, max_order, radius)
        os.remove(file_coor)
    except Exception as e:
        print(f"error in {model_name}: {e}")

def batch_cal_area_order1(model_list,atom_coor_dir,slice_dir,out_dir,n_jobs):
    Parallel(n_jobs=n_jobs)(
        delayed(cal_area)(model,atom_coor_dir,slice_dir,out_dir) for model in model_list
    )




