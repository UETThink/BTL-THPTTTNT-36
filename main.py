import cv2
import math
from ultralytics import YOLO

# 1. Khởi tạo 2 mô hình
lp_detector = YOLO('result_train(pt)/lp_model.pt')  # Mô hình tìm biển số
ocr_detector = YOLO('result_train(pt)/ocr_model.pt')  # Mô hình đọc từng ký tự


def sort_characters(boxes, classes, names_dict):
    """
    Thuật toán sắp xếp các ký tự nhận diện được.
    Xử lý được cả biển số 1 dòng (ô tô dài) và 2 dòng (xe máy, ô tô vuông).
    """
    if len(boxes) == 0:
        return ""

    chars = []
    # Tính tâm (cx, cy) của từng ký tự
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        char_name = names_dict[int(cls)]
        chars.append({'char': char_name, 'cx': cx, 'cy': cy, 'h': y2 - y1})

    # Sắp xếp các ký tự theo chiều dọc (trên xuống dưới) trước
    chars.sort(key=lambda x: x['cy'])

    # Xác định xem biển số có mấy dòng bằng cách so sánh độ lệch tọa độ Y
    # Tính chiều cao trung bình của các ký tự
    avg_h = sum(c['h'] for c in chars) / len(chars)

    line_1 = []
    line_2 = []

    # Chia dòng: Nếu tọa độ Y của ký tự cách ký tự cao nhất > 0.5 lần chiều cao trung bình -> nó ở dòng 2
    min_cy = chars[0]['cy']
    for c in chars:
        if c['cy'] - min_cy > avg_h * 0.5:
            line_2.append(c)
        else:
            line_1.append(c)

    # Sắp xếp các ký tự trong mỗi dòng theo chiều ngang (trái qua phải)
    line_1.sort(key=lambda x: x['cx'])
    line_2.sort(key=lambda x: x['cx'])

    # Ghép chuỗi
    result = "".join([c['char'] for c in line_1])
    if len(line_2) > 0:
        result += "-" + "".join([c['char'] for c in line_2])

    return result


# 2. Mở luồng Video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Giai đoạn 1: Tìm vùng chứa biển số
    lp_results = lp_detector.predict(frame, conf=0.5, verbose=False)

    for result in lp_results:
        for box in result.boxes:
            # Lấy tọa độ biển số
            px1, py1, px2, py2 = map(int, box.xyxy[0])

            # Cắt ảnh biển số
            plate_img = frame[py1:py2, px1:px2]

            if plate_img.size == 0: continue

            # Giai đoạn 2: Nhận diện chữ/số trên cái biển số vừa cắt
            # Đặt conf thấp một chút (vd 0.4) để không bị sót chữ mờ
            ocr_results = ocr_detector.predict(plate_img, conf=0.4, verbose=False)

            for ocr_res in ocr_results:
                char_boxes = ocr_res.boxes.xyxy.cpu().numpy()
                char_classes = ocr_res.boxes.cls.cpu().numpy()

                # Gọi thuật toán sắp xếp
                plate_text = sort_characters(char_boxes, char_classes, ocr_detector.names)

                # Vẽ khung xanh và in kết quả cuối cùng lên màn hình
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame, f"LP: {plate_text}", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Vẽ các khung nhỏ bao quanh từng chữ cái để debug (Tùy chọn)
                for cb in char_boxes:
                    cx1, cy1, cx2, cy2 = map(int, cb)
                    # Lưu ý: Tọa độ chữ cái là tọa độ trên ảnh plate_img, cần cộng thêm px1, py1 để vẽ lên frame gốc
                    cv2.rectangle(frame, (px1 + cx1, py1 + cy1), (px1 + cx2, py1 + cy2), (255, 0, 0), 1)

    cv2.imshow('He Thong ALPR UET - 2 Stage', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()