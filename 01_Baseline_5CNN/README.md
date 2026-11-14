# 01 – Baseline: 5 CNN

## Mục đích

Đo kết quả phân loại 5 lớp bệnh của 5 CNN tiêu chuẩn, làm mốc so sánh cho mô hình CBM (thí nghiệm 02).

## Thiết lập

- Backbone pretrained ImageNet, thay lớp cuối bằng `Linear(·, 5)`, fine-tune toàn bộ.
- Ảnh 448×448, batch 8 × tích lũy 4, AdamW lr 1e-4, weight decay 1e-4, cosine, tối đa 60 epoch, dừng sớm sau 12 epoch.
- Chọn mô hình theo val macro-F1 (rồi val CE). Seed 42, 43, 44.
- Lệnh: `python code/run.py --backbone <tên> --mode baseline --seed <seed>` (chạy hàng loạt: `code/chay_tat_ca.bat`).
- Thời gian chạy: 08/11/2026 21:07 → 09/11/2026 00:00 (15 lượt).

## Kết quả (test, trung bình ± độ lệch chuẩn qua 3 seed)

| Backbone | Test accuracy | Test macro-F1 | Tham số (M) |
|---|---|---|---|
| ResNet18 | 0,8439 ± 0,0121 | 0,8470 ± 0,0130 | 11,2 |
| DenseNet121 | 0,9021 ± 0,0046 | 0,9033 ± 0,0050 | 7,0 |
| EfficientNet-B0 | 0,8862 ± 0,0279 | 0,8885 ± 0,0275 | 4,0 |
| ConvNeXt-Tiny | 0,9312 ± 0,0121 | 0,9326 ± 0,0121 | 27,8 |
| MobileNetV2 | 0,8677 ± 0,0255 | 0,8687 ± 0,0254 | 2,2 |

## File

- `SoSanh_5_Baseline.xlsx`: bảng so sánh 5 baseline.
- `runs/<backbone>_baseline_seed<seed>/`: kết quả từng lượt.
- Bảng chung với CBM: `../02_CBM_5CNN_14concept/tong_hop/BANG_SO_SANH.md`.
