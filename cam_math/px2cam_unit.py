import numpy as np


def px2cam_unit(fovh_deg, res, px, AR):
    """
    Converts Screen-Centered Pixels (2D) to Unit Vectors in Camera Frame (3D).
    Input `px` must assume (0,0) is the center of the image.
    """
    # 0) Safety: Ensure inputs are float and res is shaped (2, 1)
    px = px.astype(np.float64)
    res = np.array(res, dtype=np.float64).reshape(2, 1)

    # 1) Center points
    # SKIPPED! Inputs are already centered.
    
    # 2) Calculate Inverse Focal Lengths (Scale Factors)
    tan_half_fov = np.tan(np.deg2rad(fovh_deg / 2.0))
    
    # sp = [1/fx, 1/fy]
    sp = np.array([
        [2.0 * tan_half_fov / res[0, 0]],       
        [2.0 * AR * tan_half_fov / res[1, 0]]   
    ])

    # 3) Apply Scaling to get Normalized Image Coordinates (x/z, y/z)
    vs_2d = px * sp

    # 4) Stack with Z = 1
    ones = np.ones((1, px.shape[1]))
    vs_unnormalized = np.vstack((vs_2d, ones)) 

    # 5) Normalize to Unit Vectors
    magnitudes = np.linalg.norm(vs_unnormalized, axis=0)
    return vs_unnormalized / magnitudes