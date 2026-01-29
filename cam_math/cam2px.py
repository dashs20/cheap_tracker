import numpy as np
from .cam_class import camera

def cam2px(cam: camera, pts):
    """
    Transform from camera frame (3D) to Screen-Centered Pixels (2D).
    Origin (0,0) is the center of the image.
    +X is Right, +Y is Down (OpenCV convention).
    """
    # Ensure pts is a numpy array
    pts = np.array(pts)

    # Handle 1D array input
    if pts.ndim == 1:
        pts = pts[:, np.newaxis]

    # 1) Calculate Focal Lengths (sx, sy) in Pixels
    # sx = focal length (horizontal)
    sx = cam.res[0] / (2 * np.tan(np.deg2rad(cam.fovh_deg) / 2))
    
    # sy = focal length (vertical)
    # Note: If pixels are square, sy should equal sx. 
    # If using AR to force a specific aspect, we use the formula below:
    sy = cam.res[1] / (2 * np.tan(np.deg2rad(cam.fovh_deg) / 2) * cam.AR)

    # 2) Project: Divide by Z and scale by focal length
    # Note: We do NOT add res/2 here. (0,0) stays at center.
    u = sx * (pts[0, :] / pts[2, :])
    v = sy * (pts[1, :] / pts[2, :])

    return np.vstack((u, v))