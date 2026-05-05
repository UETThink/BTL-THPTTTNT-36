from minio import Minio
import json

def setup_storage():
    # 1. Kết nối tới MinIO Server (Cổng API 9000)
    client = Minio(
        "localhost:9000",
        access_key="admin",
        secret_key="admin123",
        secure=False # Đang chạy local nên không cần HTTPS
    )

    buckets_need_to_create = ["full-frames", "plates"]

    # 2. Tạo Buckets
    for bucket_name in buckets_need_to_create:
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
            print(f"✅ Đã tạo bucket mới: {bucket_name}")
            
            # 3. Set quyền Public (Read-Only) để Frontend đọc được ảnh
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                    }
                ]
            }
            client.set_bucket_policy(bucket_name, json.dumps(policy))
            print(f"🔓 Đã mở quyền Public cho bucket: {bucket_name}")
        else:
            print(f"⏩ Bucket '{bucket_name}' đã tồn tại, bỏ qua.")

if __name__ == "__main__":
    print("Đang cấu hình MinIO Storage...")
    setup_storage()
    print("Hoàn tất!")