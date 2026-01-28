import cv2
import numpy as np
from cam_math import *

# dash math stuff
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


# opencv stuff
# --- 1. SETUP (Run Once) ---
# Configure the blob detector
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 80     # Brightness floor (0-255)
params.filterByArea = True
params.minArea = 2           # Minimum pixel area to count as a "spot"
params.filterByCircularity = False # Set True if you only want perfect circles
params.minDistBetweenBlobs = 10

detector = cv2.SimpleBlobDetector_create(params)

# Initialize Cameras (Indexes 0 and 1 are standard for USB cams)
cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Check if cameras opened
if not cam1.isOpened() or not cam2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# --- 2. LOOP (Run Every Frame) ---
print("Tracking started. Press 'q' to quit.")

while True:
    # Read frames
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    
    if not ret1 or not ret2: break

    # Convert to grayscale (OpenCV requires this for detection)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect blobs (Returns list of KeyPoint objects)
    keypoints1 = detector.detect(gray1)
    keypoints2 = detector.detect(gray2)

    # Convert to MATLAB-style Nx2 NumPy arrays
    # If no points found, return an empty 0x2 array to prevent errors later
    pts1 = np.array([kp.pt for kp in keypoints1]) if keypoints1 else np.empty((0, 2))
    pts2 = np.array([kp.pt for kp in keypoints2]) if keypoints2 else np.empty((0, 2))

    # --- YOUR LOGIC GOES HERE ---

    # 1) get unit vectors in camera frame
    v1 = get_cam2pt_hat(55,np.array([640,360]),pts1,9/16)
    v2 = get_cam2pt_hat(55,np.array([640,360]),pts2,9/16)

    # 2) correlate the pts
    
    # Visualization (Optional: Draw red circles on found spots)
    # Using drawKeypoints is the built-in debug view
    vis1 = cv2.drawKeypoints(frame1, keypoints1, np.array([]), (0,0,255), 
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("Camera 1 View", vis1)
    
    # Quit logic
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam1.release()
cam2.release()
cv2.destroyAllWindows()