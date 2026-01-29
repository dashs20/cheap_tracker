import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from plotting_tools import *

# Import your custom modules
from cam_math import *
from cam_class import camera

# --- 1. SETUP SCENE ---
res = np.array([640, 360])
fovh_deg = 110
AR = 9/16

# The "God View" Point (Ground Truth)
# We will see if your math can find this exact point.
TRUE_PT_G = np.array([1.5, -1.0, 1.0]) 

# CAM 1
cam1 = camera(np.array([ 0.5661,-2.3785, 2.5925]),
              np.array([[-0.9917, 0.1237, 0.0359],
                        [ 0.1147, 0.7216, 0.6827],
                        [ 0.0585, 0.6812,-0.7298]]),
              fovh_deg, res, AR)

# CAM 2
cam2 = camera(np.array([ 2.1788,-1.201 , 2.9993]),
              np.array([[-0.1162,-0.7595,-0.64  ],
                        [-0.9925, 0.065 , 0.1031],
                        [-0.0367, 0.6472,-0.7614]]),
              fovh_deg, res, AR)

# --- 2. GENERATE FAKE SENSOR DATA ---
def project_point_to_cam(cam, pt_g):
    vec_g = pt_g - cam.r_o2cam_g
    pt_c = cam.c_cam2g.T @ vec_g 
    
    # cam2px returns CENTERED coordinates (0,0 is middle)
    centered_px = cam2px(cam, pt_c.reshape(3,1))
    
    # FIX: Shift to Top-Left Origin (0,0 is top-left)
    # This mimics what the hardware actually gives us
    top_left_px = centered_px + (cam.res.reshape(2,1) / 2.0)
    
    return top_left_px

# Generate the "perfect" pixels the cameras would see
px1_raw = project_point_to_cam(cam1, TRUE_PT_G) # Shape (2,1)
px2_raw = project_point_to_cam(cam2, TRUE_PT_G) # Shape (2,1)

# Transpose to match OpenCV format (N, 2) just to be authentic to your pipeline inputs
px1_sim = px1_raw.T 
px2_sim = px2_raw.T 

print(f"--- SIMULATION ---")
print(f"True Point: {TRUE_PT_G}")
print(f"Cam 1 sees pixel: {np.round(px1_sim.flatten(), 1)}")
print(f"Cam 2 sees pixel: {np.round(px2_sim.flatten(), 1)}")
print("-" * 30)


# --- 3. RUN YOUR PIPELINE (The "Test") ---

# A) Offset Pixels
px1_off = offset_pixels(res, px1_sim)
px2_off = offset_pixels(res, px2_sim)

# B) Get Unit Vectors
u1 = px2cam_unit(cam1, px1_off)
u2 = px2cam_unit(cam2, px2_off)

# C) Correlate (Should return index 0 since we only have 1 point)
indices = correlate(cam1, cam2, u1, u2)
print(f"Correlate matched index 0 with index: {indices[0]}")

# D) Triangulate
v1_vec = u1[:, 0]
v2_vec = u2[:, 0] # Use indices here in real life
calc_pt, error = locate(v1_vec, v2_vec, cam1, cam2)

print(f"Calculated: {np.round(calc_pt, 4)}")
print(f"Error: {np.linalg.norm(calc_pt - TRUE_PT_G):.5f} meters")


# --- 4. VISUALIZATION ---
plt.ioff()
fig = plt.figure(figsize=(14, 7))

# === Left: 3D Scene ===
ax3d = fig.add_subplot(121, projection='3d')
ax3d.set_title(f"3D Math Verification\nError: {error:.4f}m")
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')

# Plot Cameras
ax3d.scatter(*cam1.r_o2cam_g, c='r', marker='^', s=100, label='Cam 1')
ax3d.scatter(*cam2.r_o2cam_g, c='b', marker='^', s=100, label='Cam 2')

# Plot Truth vs Calculated
ax3d.scatter(*TRUE_PT_G, c='g', marker='x', s=100, label='Ground Truth')
ax3d.scatter(*calc_pt, c='k', marker='o', s=50, label='Calculated')

# Plot Rays
ax3d.plot([cam1.r_o2cam_g[0], calc_pt[0]], 
          [cam1.r_o2cam_g[1], calc_pt[1]], 
          [cam1.r_o2cam_g[2], calc_pt[2]], 'r--', alpha=0.3)

ax3d.plot([cam2.r_o2cam_g[0], calc_pt[0]], 
          [cam2.r_o2cam_g[1], calc_pt[1]], 
          [cam2.r_o2cam_g[2], calc_pt[2]], 'b--', alpha=0.3)

ax3d.legend()
axis_equal(ax3d) # Assuming you have this helper, otherwise standard matplotlib aspect

# === Right: Cam 2 Sensor View ===
ax2 = fig.add_subplot(122)
ax2.set_title("Cam 2 Sensor: Simulated vs Expected")
ax2.set_xlim(-res[0]/2, res[0]/2)
ax2.set_ylim(res[1]/2, -res[1]/2) # Flip Y
ax2.grid(True)
ax2.set_aspect('equal')

# Draw Sensor Rect
rect = patches.Rectangle((-res[0]/2, -res[1]/2), res[0], res[1], 
                         linewidth=2, edgecolor='k', facecolor='none')
ax2.add_patch(rect)

# Draw what the pipeline "saw" (Blue)
p_seen = px2_off.flatten()
ax2.plot(p_seen[0], p_seen[1], 'bo', label='Pipeline Input (Offset)')

# Draw the Epipolar Line from Cam 1
# Project Cam1 Origin into Cam2
vec_g = cam1.r_o2cam_g - cam2.r_o2cam_g
vec_c2 = cam2.c_cam2g.T @ vec_g.flatten() # Transpose for Global->Cam
epipole = cam2px(cam2, vec_c2.reshape(3,1)).flatten()

ax2.plot([epipole[0], p_seen[0]], [epipole[1], p_seen[1]], 'g--', alpha=0.5, label='Epipolar Line')
ax2.legend()

plt.show()