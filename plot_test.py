from plotting_tools import *
from cam_math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CAMERA A ---
r_o2cam1_g = np.array([ 1.9462,-0.9882, 3.0655])
c_cam12g   = np.array([[-0.0658,-0.7498,-0.6584],
                        [-0.9978, 0.0555, 0.0365],
                        [ 0.0091, 0.6594,-0.7517]])

# --- CAMERA B ---
r_o2cam2_g = np.array([ 0.2248,-2.3525, 2.4353])
c_cam22g   = np.array([[-0.9954, 0.0935, 0.022 ],
                        [ 0.0818, 0.7057, 0.7037],
                        [ 0.0503, 0.7023,-0.7101]])

# Define point to be tracked
pt_g = np.array([0,0,0])

# get camera 1 origin in camera 2 frame. Also get pt_g in camrea 2 frame
r_cam22cam1_g = r_o2cam2_g - r_o2cam1_g
r_cam22cam1_cam2 = np.transpose(c_cam22g) @ r_cam22cam1_g

r_cam22pt_g = r_o2cam2_g - pt_g
r_cam22pt_cam2 = np.transpose(c_cam22g) @ r_cam22pt_g

# get both points in cam 2's pixel frame (assume 640x360)
r_cam22cam1_px = cam2px(110,np.array([640,360]),r_cam22cam1_cam2,9/16)
r_cam22pt_px = cam2px(110,np.array([640,360]),r_cam22pt_cam2,9/16)

# --- PLOTTING ---
fig = plt.figure(figsize=(16, 8))

# === 1. 3D Plot (Left) ===
ax = fig.add_subplot(121, projection='3d')

# Plot Cameras
datum(c_cam12g, r_o2cam1_g, ax, label_suffix="Cam 1")
datum(c_cam22g, r_o2cam2_g, ax, label_suffix="Cam 2")

# Plot Global Frame
datum(np.eye(3), np.zeros((1,3)), ax, label_suffix="Global")

# plot a line from camera 1 to point
ax.plot3D(np.array([r_o2cam1_g[0],pt_g[0]]),
          np.array([r_o2cam1_g[1],pt_g[1]]),
          np.array([r_o2cam1_g[2],pt_g[2]]))

axis_equal(ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D Global View")

# === 2. 2D Plot (Right) ===
ax2 = fig.add_subplot(122)

# Ensure points are flat for plotting
p1 = np.array(r_cam22cam1_px).flatten()
p2 = np.array(r_cam22pt_px).flatten()

# Plot line and points
ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', label='Projected Line')
ax2.plot(p1[0], p1[1], 'bo') 
ax2.plot(p2[0], p2[1], 'ko', label='Target Point') 

# Text Labels
ax2.text(p1[0], p1[1], ' Cam1 (Proj)', fontsize=9)
ax2.text(p2[0], p2[1], ' Point (Proj)', fontsize=9)

# Generate Box (640x360 centered at origin)
rect = patches.Rectangle((-320, -180), 640, 360, 
                         linewidth=2, edgecolor='r', facecolor='none', label='Sensor Bounds')
ax2.add_patch(rect)

# --- 2D DATUM (Center Axes) ---
# Red Arrow (u / X axis) - Points Right
ax2.arrow(0, 0, 40, 0, head_width=10, head_length=10, fc='r', ec='r', zorder=10)
ax2.text(50, 5, 'u', color='r', fontweight='bold')

# Green Arrow (v / Y axis) - Points "Down" relative to image coords
# Note: We still draw it to +40, but since we flip the axis limits below, +40 will appear BELOW 0.
ax2.arrow(0, 0, 0, 40, head_width=10, head_length=10, fc='g', ec='g', zorder=10)
ax2.text(5, 50, 'v', color='g', fontweight='bold')

# 2D Plot Styling
ax2.set_xlabel('u (pixels)')
ax2.set_ylabel('v (pixels)')
ax2.set_title("Camera 2 Sensor View")
ax2.legend(loc='upper right')
ax2.grid(True)
ax2.set_aspect('equal')

# --- FLIP Y AXIS HERE ---
# Top limit is now negative (visually top), Bottom limit is positive (visually bottom)
ax2.set_xlim(-400, 400)
ax2.set_ylim(250, -250) 

plt.show()