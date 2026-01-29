import numpy as np
from .cam_class import camera

def locate(v1, v2, cam1: camera, cam2: camera):
    vc1_g = cam1.c_cam2g @ v1
    vc2_g = cam2.c_cam2g @ v2
    A = np.array([
        [1, 0, 0, -vc1_g[0], 0],
        [0, 1, 0, -vc1_g[1], 0],
        [0, 0, 1, -vc1_g[2], 0],
        [1, 0, 0, 0, -vc2_g[0]],
        [0, 1, 0, 0, -vc2_g[1]],
        [0, 0, 1, 0, -vc2_g[2]]
    ])
    b = np.concatenate((cam1.r_o2cam_g, cam2.r_o2cam_g))
    res, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    error = np.sqrt(residuals[0]) if residuals.size > 0 else 0.0
    return res[0:3], error