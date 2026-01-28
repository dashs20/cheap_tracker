import numpy as np

def dist_pt2line(point, m, b):
    """
    Calculates the perpendicular distance from a point to a line y = mx + b.
    
    Args:
        point (tuple or list): The (x, y) coordinates of the point.
        m (float): Slope of the line.
        b (float): y-intercept of the line.
    
    Returns:
        float: The perpendicular distance.
    """
    x0, y0 = point
    numerator = abs(m * x0 - y0 + b)
    denominator = np.sqrt(m**2 + 1)
    
    return numerator / denominator