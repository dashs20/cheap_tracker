import cv2
import numpy as np
from multi_track import multi_track
from cam_class import camera
import sys

# --- 1. SETUP MATH OBJECTS ---
res = np.array([640, 360])
fovh_deg = 55
AR = 9/16

# CAM 1 (Math Object)
cam1 = camera(np.array([ 0.5661,-2.3785, 2.5925]),
              np.array([[-0.9917, 0.1237, 0.0359],
                        [ 0.1147, 0.7216, 0.6827],
                        [ 0.0585, 0.6812,-0.7298]]),
              fovh_deg, res, AR)

# CAM 2 (Math Object)
cam2 = camera(np.array([ 2.1788,-1.201 , 2.9993]),
              np.array([[-0.1162,-0.7595,-0.64  ],
                        [-0.9925, 0.065 , 0.1031],
                        [-0.0367, 0.6472,-0.7614]]),
              fovh_deg, res, AR)

# --- 2. SETUP HARDWARE (OPENCV) ---
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Configure Hardware
for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras.")
    sys.exit()

# Configure Blob Detector (TUNED FOR ROBUSTNESS)
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 50
params.maxThreshold = 255
params.thresholdStep = 10
params.filterByArea = True
params.minArea = 5
params.maxArea = 5000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.filterByColor = True
params.blobColor = 255
params.minDistBetweenBlobs = 10
detector = cv2.SimpleBlobDetector_create(params)

print("Tracking started. Press 'q' to quit.")

# --- 3. MAIN LOOP ---
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2: 
        print("Frame capture failed")
        break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    keypoints1 = detector.detect(gray1)
    keypoints2 = detector.detect(gray2)

    # --- FORMAT DATA ---
    px1 = np.array([kp.pt for kp in keypoints1], dtype=np.float32) if keypoints1 else np.empty((0, 2), dtype=np.float32)
    px2 = np.array([kp.pt for kp in keypoints2], dtype=np.float32) if keypoints2 else np.empty((0, 2), dtype=np.float32)

    # --- MATH ---
    if len(px1) > 0 and len(px2) > 0:
        pts_g = multi_track(px1,px2,cam1,cam2)

        # Print up to 2 points
        msg = ""
        n_pairs = pts_g.shape[1]
        count_to_print = min(n_pairs, 2)
        for i in range(count_to_print):
            # Print X, Y, Z rounded to 3 decimals
            msg += f"Pt{i}: {np.round(pts_g[:, i], 3)}  "
        print(msg)

    # --- VIZ ---
    vis1 = cv2.drawKeypoints(frame1, keypoints1, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis2 = cv2.drawKeypoints(frame2, keypoints2, None, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("Cam 1", vis1)
    cv2.imshow("Cam 2", vis2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()