import cv2
import pika
import base64
import json
import time
import requests # Cài thêm thư viện requests

cap = cv2.VideoCapture(0)
API_URL = "http://localhost:8000/api/upload-frame"

print("[INFO] Bắt đầu gửi frame qua API...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Nén ảnh thành jpg và chuyển sang Base64
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    payload = {
        "camera_id": "BTL-THPTTTNT-36",
        "image_base64": jpg_as_text
    }

    # Bắn qua Backend API
    try:
        requests.post(API_URL, json=payload)
    except Exception as e:
        print("Lỗi kết nối API:", e)

    cv2.imshow("Camera Producer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
