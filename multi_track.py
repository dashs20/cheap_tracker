import numpy as np
from cam_math import *
from cam_class import camera
from numpy.typing import NDArray

"This function takes in lit pixels from OpenCV, and spits out 3D points!"
"It can track an arbitrarily large number of points, but it'll probably start sucking"
def multi_track(px1: NDArray, px2: NDArray, cam1: camera, cam2: camera):
    # 1) Offset pixels
    px1_off = offset_pixels(cam1.res, px1)
    px2_off = offset_pixels(cam2.res, px2)

    # 2) Transform pixels to 3D unit vectors
    u1 = px2cam_unit(cam1, px1_off)
    u2 = px2cam_unit(cam2, px2_off)

    # 3) Correlate via epipolar constraint
    pt2_pt1_partner_indices = correlate(cam1, cam2, u1, u2)
    
    # 4) Sort matched indices and triangulate pairs
    matched_u1 = u1[:, pt2_pt1_partner_indices]
    n_pairs = u2.shape[1]
    pts_g = np.zeros((3, n_pairs))
    
    for i_pair in range(n_pairs):
        v1_vec = matched_u1[:, i_pair]
        v2_vec = u2[:, i_pair]
        
        # Pass the specific vectors for this pair
        point_3d, error = locate(v1_vec, v2_vec, cam1, cam2)
        pts_g[:, i_pair] = point_3d.flatten()

    # 5) Return solved points
    return pts_g