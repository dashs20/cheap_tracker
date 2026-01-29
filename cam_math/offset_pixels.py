import numpy as np

def offset_pixels(res, px):
    """
    Shifts pixels to screen-center.
    Inputs:
      res: (width, height)
      px:  OpenCV points (N, 2)
    
    Returns:
      (2, N) array centered at (0,0)
    """
    px = np.array(px)
    
    # 1) Force Transpose: OpenCV gives (N, 2), we want (2, N)
    # We check if it is NOT (2, N) or simply always transpose if we trust the source.
    # Safe bet: If the last dim is 2, it's likely (N, 2).
    if px.shape[-1] == 2:
        px = px.T
        
    # 2) Reshape 'res' for broadcasting against (2, N)
    center_offset = np.array(res).reshape(2, 1) / 2.0
    
    return px - center_offset