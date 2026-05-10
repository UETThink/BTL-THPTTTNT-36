from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pika
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import time

app = FastAPI(
    title="Hệ thống ALPR API",
    description="Backend API kết nối Frontend, RabbitMQ và PostgreSQL"
)

# THÊM ĐOẠN NÀY ĐỂ MỞ CỬA CHO WEB GỌI API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# CẤU HÌNH KẾT NỐI (Hardcode cho môi trường Local/Docker)
# ==========================================
DB_CONFIG = {
    "dbname": "phat_nguoi_db",  # Đã sửa cho khớp với Docker
    "user": "admin",
    "password": "admin",        # Đã sửa cho khớp với Docker
    "host": "localhost",        # Giữ nguyên vì code đang chạy ngoài Docker
    "port": "5432"
}

RABBITMQ_HOST = 'localhost' # Đổi thành 'rabbitmq' khi đưa vào Docker Compose

# ==========================================
# MODELS DỮ LIỆU ĐẦU VÀO (Pydantic)
# ==========================================
class FramePayload(BaseModel):
    camera_id: str
    image_base64: str

# ==========================================
# CÁC HÀM TIỆN ÍCH
# ==========================================
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"[Lỗi Database] Không thể kết nối: {e}")
        raise HTTPException(status_code=500, detail="Lỗi kết nối cơ sở dữ liệu")

def publish_to_rabbitmq(queue_name: str, message: dict):
    try:
        credentials = pika.PlainCredentials('admin', 'admin123')
        parameters = pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=True)
        
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2, # Đảm bảo message không bị mất khi RabbitMQ restart
            )
        )
        connection.close()
    except Exception as e:
        print(f"[Lỗi RabbitMQ] Không thể gửi tin nhắn: {e}")
        raise HTTPException(status_code=500, detail="Lỗi kết nối Message Queue")

# ==========================================
# ENDPOINTS (CÁC API)
# ==========================================

@app.post("/api/upload-frame")
async def upload_frame(payload: FramePayload):
    """
    API nhận ảnh Base64 từ Camera/Frontend và đẩy thẳng vào RabbitMQ.
    Phản hồi ngay lập tức để luồng Camera không bị giật lag.
    """
    message = {
        "camera_id": payload.camera_id,
        "image": payload.image_base64,
        "timestamp": time.time()
    }
    
    # Đẩy vào hàng đợi 'camera_frames' cho Worker xử lý
    publish_to_rabbitmq(queue_name='camera_frames', message=message)
    
    return {"status": "success", "message": "Đã đưa frame vào hàng đợi xử lý"}

@app.get("/api/logs")
async def get_camera_logs(limit: int = 50):
    """
    API lấy danh sách lịch sử nhận diện từ PostgreSQL.
    Trả về dữ liệu dạng JSON cho Frontend hiển thị lên bảng.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor) # Dùng RealDictCursor để kết quả trả về dạng Dictionary
    
    try:
        # Lấy các bản ghi mới nhất
        query = "SELECT * FROM lich_su_camera ORDER BY thoi_gian_chup DESC LIMIT %s"
        cursor.execute(query, (limit,))
        logs = cursor.fetchall()
        return {"status": "success", "data": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/api/violations")
async def get_violations():
    """
    API lấy danh sách các xe vi phạm phạt nguội (Mock data).
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        query = "SELECT * FROM phat_nguoi ORDER BY thoi_gian_vi_pham DESC"
        cursor.execute(query)
        violations = cursor.fetchall()
        return {"status": "success", "data": violations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/api/search")
async def search_by_plate(plate: str):
    """
    API tìm kiếm lịch sử đi qua của một xe cụ thể theo biển số.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        query = "SELECT * FROM lich_su_camera WHERE bien_so ILIKE %s ORDER BY thoi_gian_chup DESC"
        cursor.execute(query, (f"%{plate}%",))
        results = cursor.fetchall()
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()
