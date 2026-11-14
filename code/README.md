# Code dùng chung (thí nghiệm 01–04)

Một bộ code duy nhất chạy cả baseline, CBM, ablation 16 concept và transfer learning. Các thí nghiệm chỉ khác nhau ở lệnh chạy.

## Thành phần

| File | Vai trò |
|---|---|
| `src/data.py` | Đọc ảnh và nhãn concept, tiền xử lý, augmentation, `pos_weight` cho từng concept |
| `src/models.py` | Baseline (backbone + lớp Linear) và CBM (backbone → chiếu 1×1 → Transformer → attention theo từng concept → 14 concept → MLP → bệnh) |
| `src/losses.py` | CE cho bệnh + BCE có trọng số cho concept (bỏ qua nhãn 9) |
| `src/train.py` | Vòng huấn luyện: AdamW, cosine, AMP, tích lũy gradient, chọn mô hình theo (val macro-F1, −val CE), dừng sớm |
| `src/evaluate.py` | Báo cáo phân loại, chọn ngưỡng concept trên val, AUC/F1 từng concept, tỉ lệ vi phạm ràng buộc C16 |
| `src/transfer.py` | Đóng băng / mở khối backbone, đóng băng BatchNorm, vòng train theo giai đoạn (thí nghiệm 04) |
| `configs/common.json` | Dữ liệu, 5 lớp, 14 concept, ảnh 448×448, seed 42/43/44, tham số CBM dùng chung |
| `configs/<backbone>.json` | lr 1e-4, weight decay 1e-4, batch 8 × tích lũy 4, tối đa 60 epoch, dừng sớm 12, nút đặc trưng cho CBM |
| `configs/transfer.json` | Thiết lập hai giai đoạn và danh sách khối stride-32 được mở cho từng backbone |
| `run.py` | Chạy 1 lượt (01, 02, 03) |
| `run_tl.py` | Chạy 1 lượt transfer learning (04) |
| `summarize.py` | Tổng hợp 01 + 02 → `BANG_SO_SANH.md`, các file `.csv` |
| `so_sanh_14_16.py` | Tổng hợp 03 → `SoSanh_14_vs_16_concept.xlsx` |
| `so_sanh_tl.py` | Tổng hợp 04 → `SoSanh_TransferLearning.xlsx` |
| `check.py` | Kiểm tra dữ liệu và chạy thử các mô hình trước khi train |

## Lệnh chạy

```
python run.py --backbone resnet18 --mode baseline --seed 42                    # 01
python run.py --backbone resnet18 --mode cbm --seed 42                         # 02
python run.py --backbone resnet18 --mode cbm --seed 42 --concept-set 16        # 03
python run_tl.py --backbone resnet18 --mode cbm --seed 42 --strategy two_stage # 04 (B và D)
python run_tl.py --backbone resnet18 --mode cbm --seed 42 --strategy scratch   # 04 (C)
```

Backbone: `resnet18`, `densenet121`, `efficientnet_b0`, `convnext_tiny`, `mobilenet_v2`.
Các file `.bat` chạy hàng loạt trên Windows: `chay_tat_ca.bat` (01 + 02), `chay_16c.bat` (03), `chay_tl.bat` (04);
`1_*.bat` là chạy thử 2 epoch trước khi chạy thật. `_env.bat` chứa đường dẫn Python của máy đã chạy.

## Đường dẫn cần sửa khi chạy trên máy khác

`configs/common.json` đang dùng đường dẫn tuyệt đối của máy gốc:
`data_root` (ảnh), `label_dir` (nhãn concept), `runs_root` (nơi ghi kết quả). Sửa ba giá trị này và `_env.bat` trước khi chạy.

## Tính lặp lại

Seed được cố định (`seed_all`), `torch.use_deterministic_algorithms(True, warn_only=True)`.
Phép lan truyền ngược của attention trên GPU không tất định hoàn toàn (PyTorch in cảnh báo), nên chạy lại cùng seed có thể lệch nhỏ.
Vì vậy mọi kết quả được báo cáo qua 3 seed (trung bình ± độ lệch chuẩn).
