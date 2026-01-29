import numpy as np
from .cam2px import cam2px
from cam_class import camera

def correlate(cam1: camera, cam2: camera, pts1_cam1, pts2_cam2):
    """
    Correlates points from Cam2 to lines formed by points from Cam1.
    Assumes pts are shape (3, N) and (3, M).

    NOTE: this function is fine with pts2_cam being shorter than pts1_cam
    or vice versa. But it will only return a list as long as the union.
    AKA: 3 pt1's, 2 pt2's, you get 2 matching indices.
    """
    
    # 1) Obtain camera 1's position relative to camera 2 (in cam 2 frame)
    r_cam22cam1_g = cam1.r_o2cam_g - cam2.r_o2cam_g
    
    # Rotate global vector into Cam 2 frame
    # Note: Ensure r_cam22cam1_g is shape (3,1) for correct broadcasting if needed
    if r_cam22cam1_g.ndim == 1: r_cam22cam1_g = r_cam22cam1_g[:, np.newaxis]
    r_cam22cam1_cam2 = np.transpose(cam2.c_cam2g) @ r_cam22cam1_g

    # 2) Transform pts1 into Cam2 frame
    # P_cam2 = R_1to2 @ P_cam1 + T_1to2
    R_1to2 = np.transpose(cam2.c_cam2g) @ cam1.c_cam2g
    pts1_cam2 = (R_1to2 @ pts1_cam1) + r_cam22cam1_cam2

    # 3) Project everything into pixel space
    # px1_cam2: The pts1 projected onto Cam2 image (Shape: 2, N)
    # px_epipole: Cam1 origin projected onto Cam2 image (Shape: 2, 1)
    px1_cam2 = cam2px(cam2, pts1_cam2)
    px_epipole = cam2px(cam2, r_cam22cam1_cam2) # This is the "Epipole"
    px2_cam2 = cam2px(cam2, pts2_cam2)          # Shape: 2, M

    # 4) Calculate Epipolar Lines in Standard Form: ax + by + c = 0
    # The line passes through px1_cam2 (point P) and px_epipole (point E)
    # Vector V = P - E
    diff = px1_cam2 - px_epipole 
    
    # Normal vector (a, b) is orthogonal to V = (dx, dy) -> (-dy, dx)
    # This avoids calculating slope 'm' and handles vertical lines
    a = -diff[1, :]       # Shape (N,)
    b =  diff[0, :]       # Shape (N,)
    
    # Solve for c: ax + by + c = 0  ->  c = -(ax + by)
    # We use the epipole coordinates (x_e, y_e) to solve for c
    c = -(a * px_epipole[0] + b * px_epipole[1]) # Shape (N,)

    # 5) Calculate distances from every point (Cam2) to every line (Cam1)
    # We broadcast to create a matrix of distances: (N_lines, M_points)
    
    # Normalize a, b to make distance calculation Euclidean
    norm_factor = np.sqrt(a**2 + b**2)
    
    # Reshape for broadcasting
    # A, B, C: (N, 1)
    # X, Y (from px2): (1, M)
    A = a[:, np.newaxis] / norm_factor[:, np.newaxis]
    B = b[:, np.newaxis] / norm_factor[:, np.newaxis]
    C = c[:, np.newaxis] / norm_factor[:, np.newaxis]
    
    X = px2_cam2[0, :][np.newaxis, :]
    Y = px2_cam2[1, :][np.newaxis, :]

    # Distance Matrix: |Ax + By + C|
    dists = np.abs(A * X + B * Y + C) # Shape (N_pts1, M_pts2)

    # 6) Find the closest line (index of pts1) for each point in pts2
    # argmin along axis 0 finds the row index (pts1 index) for each column (pts2 point)
    closest_pt1_idx = np.argmin(dists, axis=0)

    return closest_pt1_idx