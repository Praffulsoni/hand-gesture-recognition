import cv2

print("=== STARTING WEB CAM SLOT SCAN ===")
for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Index {index}: WORKING | Resolution: {int(width)}x{int(height)}")
        cap.release()
    else:
        print(f"Index {index}: Closed/Unavailable")
print("=== SCAN COMPLETE ===")