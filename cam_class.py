import numpy as np
from numpy.typing import NDArray

class camera:
    """
    Camera object; no actual functions exist.
    """

    def __init__(self, r_o2cam_g: NDArray, c_cam2g: NDArray, fovh_deg: float, res: NDArray, AR: float):
        """
        Initializes the cam with the provided attributes.
        """
        self.r_o2cam_g = r_o2cam_g # vector from global origin to camera pinhole.
        self.c_cam2g = c_cam2g # column DCM rotating from camera frame to global frame.
        self.fovh_deg = fovh_deg # camera horizontal field of view, in degrees
        self.res = res # camera sensor resolution (width,height)
        self.AR = AR # aspect ratio (width/height)