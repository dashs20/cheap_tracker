import numpy as np
import matplotlib.pyplot as plt

def datum(dcm, position, ax=None, axis_length=1.0, label_suffix=""):
    """
    Plots a 3D datum (coordinate frame) defined by a DCM and a position.

    Parameters:
    - dcm (np.array): 3x3 Direction Cosine Matrix. 
                      Assumes columns are the X, Y, Z unit vectors of the datum 
                      expressed in the reference frame.
    - position (np.array): [x, y, z] position of the datum's origin.
    - ax (matplotlib.axes._subplots.Axes3DSubplot): Existing 3D axis. 
                                                     If None, a new figure is created.
    - axis_length (float): The length of the plotted axis arrows.
    - label_suffix (str): Optional text to append to axis labels (e.g. "_body").
    """
    
    # Ensure inputs are numpy arrays
    dcm = np.array(dcm)
    pos = np.array(position).flatten()
    
    if dcm.shape != (3, 3):
        raise ValueError("DCM must be a 3x3 matrix")
    if pos.shape != (3,):
        raise ValueError("Position must be a 3-element vector")

    # Create axis if not provided
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Extract the unit vectors (columns of the DCM)
    # col 0 = X axis, col 1 = Y axis, col 2 = Z axis
    u = dcm[:, 0] * axis_length
    v = dcm[:, 1] * axis_length
    w = dcm[:, 2] * axis_length

    # Origins for the arrows (all start at 'position')
    x_orig, y_orig, z_orig = pos[0], pos[1], pos[2]

    # Plot X axis (Red)
    ax.quiver(x_orig, y_orig, z_orig, u[0], u[1], u[2], 
              color='r', label=f'X{label_suffix}', linewidth=2)
    
    # Plot Y axis (Green)
    ax.quiver(x_orig, y_orig, z_orig, v[0], v[1], v[2], 
              color='g', label=f'Y{label_suffix}', linewidth=2)
    
    # Plot Z axis (Blue)
    ax.quiver(x_orig, y_orig, z_orig, w[0], w[1], w[2], 
              color='b', label=f'Z{label_suffix}', linewidth=2)

    # Plot the origin point
    ax.scatter(x_orig, y_orig, z_orig, color='k', s=20)

    # We add a slight offset (1.1x) to put the text just beyond the arrow tip
    ax.text(x_orig + u[0]*1.1, y_orig + u[1]*1.1, z_orig + u[2]*1.1, 
            f"X{label_suffix}", color='r', fontweight='bold')
    
    ax.text(x_orig + v[0]*1.1, y_orig + v[1]*1.1, z_orig + v[2]*1.1, 
            f"Y{label_suffix}", color='g', fontweight='bold')
    
    ax.text(x_orig + w[0]*1.1, y_orig + w[1]*1.1, z_orig + w[2]*1.1, 
            f"Z{label_suffix}", color='b', fontweight='bold')

    return ax