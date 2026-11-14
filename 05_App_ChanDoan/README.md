# 05 – Ứng dụng chẩn đoán bệnh lá lúa

## Chức năng

Người dùng đưa vào một ảnh lá lúa; ứng dụng trả về:
- tên bệnh (hoặc "Không chắc chắn" nếu xác suất cao nhất < 0,6, ngưỡng chọn trên val);
- "Dấu hiệu đặc trưng trên ảnh": các concept mô hình nhận ra, kèm độ tin cậy, và "Dấu hiệu có thể có" (dưới ngưỡng);
- một heatmap attention (tầng stride-32) cho biết vùng mô hình chú ý;
- đối chiếu ontology V1.8: "Phù hợp" (đủ tổ hợp P0x), "Phù hợp một phần", hoặc "Cảnh báo" khi mâu thuẫn với ràng buộc lá khỏe C16.

Mô hình dùng: **ConvNeXt-Tiny + Concept Attention + CBM (14 concept) + MLP, seed 43** (lượt `02_CBM_5CNN_14concept/runs/convnext_tiny_cbm_seed43`).
Giao diện: FastAPI + một trang HTML, chạy cục bộ.

## File

| File | Vai trò |
|---|---|
| `server.py`, `static/index.html` | Máy chủ web và giao diện |
| `core/predictor.py` | Nạp mô hình, dự đoán, heatmap, đối chiếu ontology |
| `model_src/models.py` | Bản sao `code/src/models.py` dùng cho app |
| `model/config.json`, `model/concept_thresholds.json` | Cấu hình và ngưỡng concept của mô hình đã chọn |
| `knowledge/ontology_v18.json` | Tên concept tiếng Việt và tổ hợp P01–P04 dùng khi hiển thị |
| `app_config.json` | Ngưỡng "Không chắc chắn", độ đậm heatmap, đường dẫn mô hình |
| `du_doan_1_anh.py`, `chan_doan_heatmap.py`, `kiem_tra_khop.py`, `kiem_tra_server.py` | Script kiểm tra từng bước |
| `dong_goi/` | Đóng gói để chạy trên máy Windows khác không cài Python (`HUONG_DAN.txt`) |

## Chạy

Cần thêm `model/best.pt` (không có trong repo, ~117 MB): chép từ lượt `convnext_tiny_cbm_seed43` vào `model/`.
Sau đó chạy `3_chay_app_thu.bat` (hoặc `python server.py`) và mở http://127.0.0.1:8000.
