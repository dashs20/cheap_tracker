import numpy as np
from .cam_class import camera

def px2cam_unit(cam: camera, px):
    """
    Converts Screen-Centered Pixels (2D) to Unit Vectors in Camera Frame (3D).
    Input `px` must assume (0,0) is the center of the image.
    """
    # 0) Safety: Ensure inputs are float and res is shaped (2, 1)
    px = px.astype(np.float64)
    res = np.array(cam.res, dtype=np.float64).reshape(2, 1)
    
    # 2) Calculate Inverse Focal Lengths (Scale Factors)
    tan_half_fov = np.tan(np.deg2rad(cam.fovh_deg / 2.0))
    
    # sp = [1/fx, 1/fy]
    sp = np.array([
        [2.0 * tan_half_fov / res[0, 0]],       
        [2.0 * cam.AR * tan_half_fov / res[1, 0]]   
    ])

    # 3) Apply Scaling to get Normalized Image Coordinates (x/z, y/z)
    vs_2d = px * sp

    # 4) Stack with Z = 1
    ones = np.ones((1, px.shape[1]))
    vs = np.vstack((vs_2d, ones)) 

    # 5) Normalize to Unit Vectors
    return vs