import cv2
import numpy as np

CAM_ID = 0 # Check if this is your camera index

def nothing(x):
    pass

cap = cv2.VideoCapture(CAM_ID)

# FORCE MANUAL EXPOSURE (Keep your current dark settings)
# You might need to tweak these depending on your driver
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6.0) 

cv2.namedWindow('Bright Spot Tuner')
cv2.createTrackbar('Threshold', 'Bright Spot Tuner', 40, 255, nothing)

print("Adjust Threshold until only the dots remain.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Convert to Grayscale (We don't care about color anymore)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Get Slider Value
    thresh_val = cv2.getTrackbarPos('Threshold', 'Bright Spot Tuner')
    
    # 3. Apply Threshold
    # Any pixel brighter than 'thresh_val' becomes 255 (White). All others 0 (Black).
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    
    # 4. GNC Safety: minimal cleanup
    # WARNING: Your dots are small. A standard 'erode' might delete them.
    # We will use no erosion, or a very tiny one if noise is bad.
    # mask = cv2.erode(mask, None, iterations=1) # Uncomment only if you see "sparkles"
    
    # 5. Visualize
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    display = frame.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter for tiny noise (1 pixel) but keep your small markers (likely 2-10 pixels)
        if area > 2: 
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(display, (int(x), int(y)), int(radius)+2, (0, 255, 0), 1)

    # Stack images
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((display, mask_bgr))
    
    cv2.imshow('Bright Spot Tuner', stacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()