import cv2
import numpy as np

# --- CONFIGURATION ---
CAM_A_ID = 0
CAM_B_ID = 1

# Camera Intrinsics
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
HORI_FOV = 55 # Degrees
RES_PX = np.array([FRAME_WIDTH, FRAME_HEIGHT])

# Tuning for the "Snap" feature
SNAP_THRESHOLD = 100  # Brightness threshold for finding dots
SEARCH_WINDOW = 40    # Size of the box to search around your click (px)

# STEP 0: GROUND TRUTH (The "Goobagob" Datum)
# Your arbitrary 3D points in METERS
GROUND_TRUTH_POINTS = np.array([
    [.4303,   0.0,    0.0],   # Point 1
    [.3481,   -.2529, 0.0],   # Point 2
    [-.1330,  -.4092, 0.0],   # Point 3
    [-.3481,  .2529,  0.0]    # Point 4
], dtype=np.float32)

# --- GUI TOOLS ---
clicked_points = []
current_frozen_frame = None # Global to hold the frame for processing

def find_centroid_in_roi(frame, center_x, center_y, window_size):
    """
    Searches a small window around (center_x, center_y) for the brightest blob.
    Returns the global sub-pixel (x, y) of the centroid.
    """
    h, w = frame.shape[:2]
    
    # Define ROI boundaries (Clamp to image edges)
    x1 = max(0, center_x - window_size // 2)
    y1 = max(0, center_y - window_size // 2)
    x2 = min(w, center_x + window_size // 2)
    y2 = min(h, center_y + window_size // 2)
    
    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    
    # Convert to Gray & Threshold
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
        
    _, mask = cv2.threshold(gray, SNAP_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None # No blob found
    
    # Find the largest blob in the ROI
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    
    if M["m00"] == 0:
        return None
        
    # Centroid relative to ROI
    cx_roi = M["m10"] / M["m00"]
    cy_roi = M["m01"] / M["m00"]
    
    # Convert back to Global Image Coordinates
    cx_global = x1 + cx_roi
    cy_global = y1 + cy_roi
    
    return (cx_global, cy_global)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Attempt to snap to the dot
        snapped_pt = find_centroid_in_roi(current_frozen_frame, x, y, SEARCH_WINDOW)
        
        if snapped_pt:
            print(f"Click at ({x},{y}) -> Snapped to ({snapped_pt[0]:.2f}, {snapped_pt[1]:.2f})")
            clicked_points.append(snapped_pt)
        else:
            print(f"Click at ({x},{y}) -> No dot found! Using raw click.")
            clicked_points.append((float(x), float(y)))

def capture_and_click(cam_id, window_name):
    global clicked_points, current_frozen_frame
    clicked_points = []
    
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Force low exposure for calibration
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0) 
    
    print(f"\n--- {window_name} ---")
    print("Press SPACE to freeze frame and start clicking.")
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            current_frozen_frame = frame.copy()
            break
            
    cap.release()
    cv2.destroyWindow(window_name)
    
    cv2.namedWindow("CLICKING_MODE")
    cv2.setMouseCallback("CLICKING_MODE", mouse_callback)
    
    print(f"Click points 1 through {len(GROUND_TRUTH_POINTS)}.")
    
    while len(clicked_points) < len(GROUND_TRUTH_POINTS):
        display = current_frozen_frame.copy()
        
        # Draw search box around mouse (visual aid)
        # Note: We can't easily track mouse move in this simple loop without more callbacks
        # but we can draw the captured points.
        
        for i, pt in enumerate(clicked_points):
            # Draw crosshair
            pt_int = (int(pt[0]), int(pt[1]))
            cv2.line(display, (pt_int[0]-10, pt_int[1]), (pt_int[0]+10, pt_int[1]), (0,255,0), 1)
            cv2.line(display, (pt_int[0], pt_int[1]-10), (pt_int[0], pt_int[1]+10), (0,255,0), 1)
            cv2.circle(display, pt_int, 2, (0, 0, 255), -1)
            
            cv2.putText(display, f"Pt {i+1}", (pt_int[0]+10, pt_int[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow("CLICKING_MODE", display)
        cv2.waitKey(50)
        
    cv2.destroyWindow("CLICKING_MODE")
    return np.array(clicked_points, dtype=np.float32)

# --- THE MAGIC (OpenCV PnP) ---
def get_camera_state(pixels, name):
    print(f"\nSolving for {name}...")
    
    # 1. Build Camera Matrix (K)
    f_px = (FRAME_WIDTH / 2.0) / np.tan(np.deg2rad(HORI_FOV / 2.0))
    K = np.array([
        [f_px, 0, FRAME_WIDTH/2],
        [0, f_px, FRAME_HEIGHT/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 2. Solve PnP
    success, rvec_w2c, tvec_w2c = cv2.solvePnP(
        GROUND_TRUTH_POINTS, 
        pixels, 
        K, 
        np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        print("!! SOLVER FAILED !! Check your points.")
        return None, None

    # 3. Convert to GNC Format
    R_w2c, _ = cv2.Rodrigues(rvec_w2c)
    R_c2w = R_w2c.T
    pos_cam_global = -R_c2w @ tvec_w2c
    
    return pos_cam_global.flatten(), R_c2w

# --- MAIN ---
if __name__ == "__main__":
    
    # 1. Capture Pixels
    px_A = capture_and_click(CAM_A_ID, "Camera A")
    px_B = capture_and_click(CAM_B_ID, "Camera B")
    
    # 2. Solve Math
    pos_A, R_A = get_camera_state(px_A, "Camera A")
    pos_B, R_B = get_camera_state(px_B, "Camera B")
    
    # 3. Print Results
    print("\n" + "="*30)
    print("   CALIBRATION RESULTS")
    print("="*30)
    
    np.set_printoptions(precision=4, suppress=True)
    
    print("\n# --- CAMERA A ---")
    print(f"r_o2cam1_g = np.array({np.array2string(pos_A, separator=',')})")
    print(f"c_cam12g   = np.array({np.array2string(R_A, separator=',')})")
    
    print("\n# --- CAMERA B ---")
    print(f"r_o2cam2_g = np.array({np.array2string(pos_B, separator=',')})")
    print(f"c_cam22g   = np.array({np.array2string(R_B, separator=',')})")
    print("\n" + "="*30)