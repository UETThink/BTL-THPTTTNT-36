import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import csv
import collections
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. KHỞI TẠO CÁC MÔ HÌNH AI
# ==========================================
print("[INFO] Đang tải các mô hình AI...")

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


# ==========================================
# 2. MỞ LUỒNG VIDEO & XỬ LÝ REAL-TIME
# ==========================================
cap = cv2.VideoCapture(0)

# KHO LƯU TRỮ TẠM THỜI CHO THUẬT TOÁN "BEST FRAME"
plate_buffer = {}           # Giỏ chứa danh sách các lần đọc biển số của mỗi xe
active_ids_last_frame = set() # Ghi nhớ ID xe của khung hình trước
logged_ids = set()          # Nhớ ID đã ghi vào CSV để không ghi trùng

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Tạo một set trống để ghi nhận các xe xuất hiện trong khung hình HIỆN TẠI
    active_ids_this_frame = set()

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
            
            # LƯU KẾT QUẢ CUỐI CÙNG VÀO CSV
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("log_bien_so.csv", mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([now, l_id, best_plate])
            print(f"[SUCCESS] Đã lưu biển số {best_plate} của xe ID: {l_id}")

            # Đánh dấu xe này đã lưu để không xử lý lại, đồng thời xóa giỏ nháp đi
            logged_ids.add(l_id)
            del plate_buffer[l_id]

    # Cập nhật lại lịch sử các ID cho vòng lặp tiếp theo
    active_ids_last_frame = active_ids_this_frame

    # Hiển thị
    cv2.imshow('He Thong ALPR UET - Street Level', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()