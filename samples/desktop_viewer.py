"""
Native Desktop Screen Viewer for SmartGuard Surveillance System
Displays the real-time AI surveillance video stream in a dedicated Windows popup window.
"""
import cv2
import sys

STREAM_URL = "http://127.0.0.1:8000/video_feed"

print(f"Connecting to live surveillance stream: {STREAM_URL}")
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("[ERROR] Could not connect to video stream. Ensure surveillance server is running!")
    sys.exit(1)

cv2.namedWindow("SmartGuard AI Mission Control - Video Screen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SmartGuard AI Mission Control - Video Screen", 960, 720)

print("\n[+] Displaying video on screen! Press 'q' or close the window to exit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Reconnecting to stream...")
        cap = cv2.VideoCapture(STREAM_URL)
        continue

    cv2.imshow("SmartGuard AI Mission Control - Video Screen", frame)
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
