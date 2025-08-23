import numpy as np



def calculate_dis(A,B):
    AB = B - A
    dis = np.linalg.norm(AB)
    return dis



def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    return np.arccos(cos_theta)

def compute_triangle_angles(A, B, C):
    # Vectors for angles at A
    AB = B - A
    AC = C - A
    angle_A = angle_between_vectors(AB, AC)

    # Vectors for angles at B
    BA = A - B
    BC = C - B
    angle_B = angle_between_vectors(BA, BC)

    # Vectors for angles at C
    CA = A - C
    CB = B - C
    angle_C = angle_between_vectors(CA, CB)

    return angle_A, angle_B, angle_C  # in radians



def calculate_triangle_properties_1(point1, point2, point3):
    # point1, point2, point3 = point1.cpu().numpy(), point2.cpu().numpy(), point3.cpu().numpy()
    pps = []
    # Calculate centroid coordinates
    centroid = np.mean([point1, point2, point3], axis=0)
    distances = [np.linalg.norm(point - centroid) for point in [point1, point2, point3]]
    pps.extend(distances)
    # Calculate the side lengths of the triangle
    a = np.linalg.norm(point2 - point1)
    b = np.linalg.norm(point3 - point2)
    c = np.linalg.norm(point1 - point3)
    pps.extend([a,b,c])
    
    A,B,C = compute_triangle_angles(point1, point2, point3)
    pps.extend([A,B,C])
    if a > 0 and b > 0 and c > 0 and (a + b > c) and (a + c > b) and (b + c > a):
        s = 0.5 * (a + b + c)  
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        pps.append(area)
    else:
        pps.append(0)

    return(pps)



def calculate_dihedral_angle(A, B, C, D):
    AB = B - A
    BC = C - B
    CD = D - C

    normal1 = np.cross(AB, BC)
    normal2 = np.cross(BC, CD)

    normal1 /= np.linalg.norm(normal1)
    normal2 /= np.linalg.norm(normal2)

    cosine_angle = np.dot(normal1, normal2)
    sine_angle = np.dot(np.cross(normal1, normal2), BC / np.linalg.norm(BC))

    angle_rad = np.arctan2(sine_angle, cosine_angle)
    if angle_rad < 0:
        angle_rad += np.pi

    return angle_rad


def calculate_dihedral_angle_from_planes(p1, p2, p3, p4, p5, p6):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    p4 = np.array(p4, dtype=float)
    p5 = np.array(p5, dtype=float)
    p6 = np.array(p6, dtype=float)

    # Calculate the normal vector of plane ABC
    v1 = p2 - p1
    v2 = p3 - p1
    n1 = np.cross(v1, v2)
    n1 /= np.linalg.norm(n1)
    
    # Calculate the normal vector of plane DEF
    v3 = p5 - p4
    v4 = p6 - p4
    n2 = np.cross(v3, v4)
    n2 /= np.linalg.norm(n2)
    
    # Calculate the cosine and sine of the dihedral angle
    cos_phi = np.dot(n1, n2)
    sin_phi = np.dot(v1, n2) * np.linalg.norm(np.cross(n1, n2))
    
    # Calculate the dihedral angle
    phi = np.arctan2(sin_phi, cos_phi)
    
    # Adjust the angle range to be between 0 and π
    if phi < 0:
        phi += np.pi
    
    return phi


