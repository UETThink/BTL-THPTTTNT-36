-- Đảm bảo sử dụng múi giờ chuẩn
SET timezone = 'Asia/Ho_Chi_Minh';

-- =================================================================
-- BẢNG 1: DANH SÁCH PHẠT NGUỘI (Mock Data nội bộ)
-- =================================================================
CREATE TABLE phat_nguoi (
    id SERIAL PRIMARY KEY,
    bien_so VARCHAR(20) NOT NULL,
    loi_vi_pham VARCHAR(255) NOT NULL,
    so_tien DECIMAL(10, 2),
    trang_thai VARCHAR(50) DEFAULT 'Chưa nộp phạt',
    thoi_gian_vi_pham TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tạo Index cho cột bien_so để tăng tốc độ tìm kiếm
CREATE INDEX idx_bien_so_phat_nguoi ON phat_nguoi(bien_so);

-- Bơm dữ liệu giả vào bảng phạt nguội
INSERT INTO phat_nguoi (bien_so, loi_vi_pham, so_tien, thoi_gian_vi_pham) VALUES
('29A-123.45', 'Vượt đèn đỏ tại ngã tư Xã Đàn', 5000000, '2026-04-10 08:15:00'),
('30G-999.99', 'Chạy quá tốc độ quy định (80/60 km/h)', 4000000, '2026-04-12 14:30:00'),
('15A-678.90', 'Dừng đỗ xe sai quy định', 900000, '2026-04-15 09:00:00'),
('51H-555.55', 'Đi sai làn đường', 2000000, '2026-04-18 10:45:00'),
('99A-111.22', 'Không chấp hành hiệu lệnh biển báo', 400000, '2026-04-20 16:20:00');

-- =================================================================
-- BẢNG 2: LỊCH SỬ CAMERA NHẬN DIỆN (Lưu log hàng ngày)
-- =================================================================
CREATE TABLE lich_su_camera (
    id SERIAL PRIMARY KEY,
    bien_so VARCHAR(20) NOT NULL,
    thoi_gian_chup TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    link_anh_goc VARCHAR(255),       -- Link trỏ tới MinIO (full-frames)
    link_anh_bien_so VARCHAR(255),   -- Link trỏ tới MinIO (plates)
    co_vi_pham BOOLEAN DEFAULT FALSE -- AI phát hiện biển này có dính phạt nguội hay không
);

CREATE INDEX idx_bien_so_lich_su ON lich_su_camera(bien_so);