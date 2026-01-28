import numpy as np

def offset_pixels(res, px):
    """
    Shifts pixels to screen-center.
    Inputs:
      res: (width, height)
      px:  OpenCV points (N, 2) OR Your Math Format (2, N)
    
    Returns:
      (2, N) array centered at (0,0) ready for your math pipeline.
    """
    px = np.array(px)
    res = np.array(res)
    
    # CASE 1: Standard OpenCV format (N, 2) -> Transpose it first
    if px.shape[-1] == 2 and px.ndim == 2 and px.shape[0] != 2:
        px = px.T
        
    # CASE 2: Already in your format (2, N)
    
    # Perform subtraction (reshape res to 2x1 to allow broadcasting against 2xN)
    return px - res.reshape(2, 1) / 2