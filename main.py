import pika
import json
import cv2
import numpy as np
import base64
import psycopg2
import io
import torch
import collections
import torchvision.models as models
import torchvision.transforms as transforms
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. KHỞI TẠO CÁC MÔ HÌNH AI VÀ KẾT NỐI CƠ SỞ DỮ LIỆU
# ==========================================
print("[INFO] Đang khởi tạo hệ thống và tải các mô hình AI...")

# A. Mô hình nhận diện phương tiện và theo dõi
vehicle_detector = YOLO('result_train(pt)/vehicle_model.pt')  

# B. Mô hình nhận diện biển số 
lp_detector = YOLO('result_train(pt)/lp_model.pt')  
ocr_detector = YOLO('result_train(pt)/ocr_model.pt')  

# C. Mô hình trích xuất Vector đặc trưng
resnet = models.resnet18(pretrained=True)
resnet.eval() 
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#Kết nối đến các cơ sở dữ liệu
# Kết nối Postgres
pg_conn = psycopg2.connect(dsn="dbname=lpr_db user=admin password=admin123 host=localhost")

# Kết nối MinIO
minio_client = Minio("localhost:9000", access_key="admin", secret_key="admin123", secure=False)

# Kết nối Qdrant
qdrant_client = QdrantClient("localhost", port=6333)


def extract_vector(image_crop):
    input_tensor = preprocess(image_crop).unsqueeze(0)
    with torch.no_grad():
        output = resnet(input_tensor)
    return output.numpy()[0] 


def sort_characters(boxes, classes, names_dict):
    if len(boxes) == 0:
        return ""
    chars = []
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        char_name = names_dict[int(cls)]
        chars.append({'char': char_name, 'cx': cx, 'cy': cy, 'h': y2 - y1})

    chars.sort(key=lambda x: x['cy'])
    avg_h = sum(c['h'] for c in chars) / len(chars)

    line_1, line_2 = [], []
    min_cy = chars[0]['cy']
    for c in chars:
        if c['cy'] - min_cy > avg_h * 0.5:
            line_2.append(c)
        else:
            line_1.append(c)

    line_1.sort(key=lambda x: x['cx'])
    line_2.sort(key=lambda x: x['cx'])

    result = "".join([c['char'] for c in line_1])
    if len(line_2) > 0:
        result += "-" + "".join([c['char'] for c in line_2])
    return result

#Hàm hỗ trợ lưu trữ
def save_to_minio(img_np, bucket, filename):
    """Lưu ảnh vào MinIO"""
    success, encoded_img = cv2.imencode('.jpg', img_np)
    if success:
        data = io.BytesIO(encoded_img.tobytes())
        minio_client.put_object(bucket, filename, data, len(encoded_img.tobytes()), content_type="image/jpeg")
        return f"http://localhost:9000/{bucket}/{filename}"
    return None

def check_violation_and_log(plate_text, img_url, lp_url):
    """Kiểm tra vi phạm và lưu vào Postgres"""
    cursor = pg_conn.cursor()
    cursor.execute("SELECT id FROM phat_nguoi WHERE bien_so = %s", (plate_text,))
    violation = cursor.fetchone()
    has_violation = True if violation else False
    
    query = """INSERT INTO lich_su_camera (bien_so, link_anh_goc, link_anh_bien_so, co_vi_pham) 
               VALUES (%s, %s, %s, %s)"""
    cursor.execute(query, (plate_text, img_url, lp_url, has_violation))
    pg_conn.commit()
    cursor.close()
    return has_violation

# Các biến trạng thái để theo dõi xe
plate_buffer = {}           # Giỏ chứa danh sách các lần đọc biển số
vehicle_images = {}         # Lưu ảnh cắt xe rõ nhất trong RAM
plate_images = {}           # Lưu ảnh cắt biển số rõ nhất trong RAM
active_ids_last_frame = set() # Ghi nhận các xe xuất hiện trong khung hình TRƯỚC
logged_ids = set() # Lưu mã ID của những chiếc xe đã được lưu vào Database thành công.


def callback(ch, method, properties, body):
    global active_ids_last_frame, logged_ids

    # A. Giải mã ảnh nhận được từ RabbitMQ
    # 1. Giải mã JSON
    data = json.loads(body)
    # Lấy thêm camera_id
    camera_id = data.get('camera_id', 'Unknown')

    # 2. Giải mã ảnh Base64 thành Numpy Array cho OpenCV
    img_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Tạo một set trống để ghi nhận các xe xuất hiện trong khung hình HIỆN TẠI
    active_ids_this_frame = set()

    #B.
    # BƯỚC 1: TÌM XE VÀ GÁN ID (TRACKING)
    # [ĐÃ SỬA]: Xóa bỏ classes=[...] để bắt mọi loại xe, hạ conf=0.2 để nhạy hơn với xe chạy nhanh
    track_results = vehicle_detector.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.2, verbose=False)
    
    if track_results[0].boxes.id is not None:
        boxes = track_results[0].boxes.xyxy.cpu().numpy()
        track_ids = track_results[0].boxes.id.int().cpu().tolist()
        
        # [ĐÃ THÊM]: Lấy thêm danh sách ID loại xe (0, 1, 2...) để tí in ra màn hình
        class_ids = track_results[0].boxes.cls.int().cpu().tolist()

        # Cập nhật danh sách các xe đang đứng trước camera
        active_ids_this_frame = set(track_ids)

        # [ĐÃ SỬA]: Nhét thêm cls_id vào vòng lặp
        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            vx1, vy1, vx2, vy2 = map(int, box)
            
            # Cắt ảnh chiếc xe
            vehicle_img = frame[vy1:vy2, vx1:vx2]
            if vehicle_img.size == 0: continue

            # Vẽ khung nhận diện Phương tiện
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            
            # [ĐÃ SỬA]: In thẳng số loại xe (Class) và ID xe lên đầu khung đỏ
            cv2.putText(frame, f"Class: {cls_id} | ID: {track_id}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # [ĐÃ THÊM] Lưu lại khung hình có chứa xe này vào RAM (luôn cập nhật ảnh mới nhất)
            vehicle_images[track_id] = frame.copy()
            
            # BƯỚC 2: TÌM BIỂN SỐ 
            lp_results = lp_detector.predict(vehicle_img, conf=0.5, verbose=False)

            for result in lp_results:
                for lp_box in result.boxes:
                    px1, py1, px2, py2 = map(int, lp_box.xyxy[0])
                    
                    # Cắt ảnh biển số
                    plate_img = vehicle_img[py1:py2, px1:px2]
                    if plate_img.size == 0: continue

                    # BƯỚC 3: ĐỌC OCR 
                    ocr_results = ocr_detector.predict(plate_img, conf=0.4, verbose=False)

                    for ocr_res in ocr_results:
                        char_boxes = ocr_res.boxes.xyxy.cpu().numpy()
                        char_classes = ocr_res.boxes.cls.cpu().numpy()

                        plate_text = sort_characters(char_boxes, char_classes, ocr_detector.names)

                        # Chuyển đổi hệ tọa độ để vẽ lên frame gốc
                        abs_px1, abs_py1 = vx1 + px1, vy1 + py1
                        abs_px2, abs_py2 = vx1 + px2, vy1 + py2

                        # Vẽ khung Biển số 
                        cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{plate_text}", (abs_px1, abs_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # ----------------------------------------------------
                        # LOGIC 1: GOM DỮ LIỆU ĐỌC ĐƯỢC VÀO GIỎ NHÁP
                        # ----------------------------------------------------
                        if track_id not in logged_ids:
                            if track_id not in plate_buffer:
                                plate_buffer[track_id] = []
                            # Thêm kết quả vào giỏ nếu OCR có đọc ra chữ
                            if plate_text.strip() != "":
                                plate_buffer[track_id].append(plate_text)

    # ----------------------------------------------------
    # LOGIC 2: CHỐT KẾT QUẢ KHI XE ĐI KHUẤT
    # ----------------------------------------------------
    # Tìm những xe có ở frame trước nhưng frame này không còn (Đã đi ra khỏi camera)
    left_ids = active_ids_last_frame - active_ids_this_frame

    for l_id in left_ids:
        # Nếu xe rời đi có dữ liệu trong giỏ nháp
        if l_id in plate_buffer and len(plate_buffer[l_id]) > 0:
            
            # Kiểm đếm xem biển số nào xuất hiện nhiều nhất
            counter = collections.Counter(plate_buffer[l_id])
            best_plate = counter.most_common(1)[0][0] 
            
            timestamp = datetime.now().strftime('%H%M%S')
            filename = f"{best_plate}_{timestamp}.jpg"
            
            # 1. Lưu ảnh vào MinIO (Lấy ảnh cuối cùng trong RAM làm Best Frame)
            full_url = save_to_minio(vehicle_images.get(l_id, frame), "full-frames", filename)
            lp_url = save_to_minio(plate_images.get(l_id, frame), "plates", f"lp_{filename}")
            
            # 2. Trích xuất và lưu Vector vào Qdrant
            try:
                vector = extract_vector(vehicle_images.get(l_id, frame))
                qdrant_client.upsert(
                    collection_name="plates",
                    points=[PointStruct(id=l_id, vector=vector.tolist(), payload={"plate": best_plate})]
                )
            except Exception as e:
                print(f"[ERROR] Qdrant Error: {e}")
            
            # 3. Check vi phạm và lưu Postgres
            is_violated = check_violation_and_log(best_plate, full_url, lp_url)
            
            print(f"[SUCCESS] Đã xử lý xe ID {l_id}: {best_plate} | Vi phạm: {is_violated}")

            # 4. Dọn dẹp bộ nhớ RAM cho xe này
            logged_ids.add(l_id)
            if l_id in plate_buffer: del plate_buffer[l_id]
            if l_id in vehicle_images: del vehicle_images[l_id]
            if l_id in plate_images: del plate_images[l_id]
    # Hiển thị video AI đang nhận diện
    cv2.imshow("He Thong AI Nhan Dien Bien So", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    # Cập nhật lại lịch sử các ID cho vòng lặp tiếp theo
    active_ids_last_frame = active_ids_this_frame
    # Báo cho RabbitMQ biết là đã xử lý xong ảnh này, gửi ảnh tiếp theo đi!
    ch.basic_ack(delivery_tag=method.delivery_tag)
# ==========================================
# 4. KẾT NỐI RABBITMQ VÀ CHẠY
# ==========================================
# Tạo thông tin đăng nhập
credentials = pika.PlainCredentials('admin', 'admin123')

rabbitmq_conn = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', credentials=credentials)
)
channel = rabbitmq_conn.channel()
channel.queue_declare(queue='camera_frames', durable=True)

# Giới hạn mỗi lần chỉ nhận 1 ảnh để tránh quá tải RAM
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='camera_frames', on_message_callback=callback)

print("[INFO] Worker AI đang lắng nghe dữ liệu từ RabbitMQ...")
channel.start_consuming()