# 04 – Transfer learning

*Trạng thái: code đã viết (`code/run_tl.py`, `code/src/transfer.py`, `code/configs/transfer.json`), chưa chạy. Kết quả sẽ được thêm vào folder này.*

## Mục đích

So sánh các cách tận dụng trọng số pretrained ImageNet cho bài toán ít dữ liệu (587 ảnh train), trên cả baseline và CBM,
5 backbone × 3 seed. Ý tưởng hai giai đoạn theo notebook `Transfer_Learning.ipynb` của thầy hướng dẫn.

## Bốn chiến lược

| Mã | Chiến lược | Thiết lập |
|---|---|---|
| A | Fine-tune toàn bộ (đang dùng) | Kết quả của 01 và 02, không chạy lại |
| B | Hai giai đoạn | GĐ1: đóng băng backbone, chỉ học phần đầu mới, lr 1e-3, tối đa 30 epoch, dừng sớm 10. GĐ2: từ mô hình tốt nhất của GĐ1, mở các khối stride-32, BatchNorm vẫn đóng băng, lr 1e-5, tối đa 30 epoch, dừng sớm 10 |
| D | Chỉ đóng băng backbone | Mô hình tốt nhất sau GĐ1 của B |
| C | Huấn luyện từ đầu | Không dùng trọng số ImageNet; mọi thiết lập khác giống A (giữ nguyên lr để chỉ khác một yếu tố) |

Khối mở ở GĐ2 (mọi khối ở độ phân giải thấp nhất, stride 32): ResNet18 `layer4`; DenseNet121 `transition3`, `denseblock4`, `norm5`;
EfficientNet-B0 `features.6–8`; ConvNeXt-Tiny `features.6–7`; MobileNetV2 `features.14–18`.

## Lệnh

```
code\1_thu_tl.bat     chạy thử 2 epoch
code\chay_tl.bat      60 lệnh (5 backbone × 3 seed × baseline/CBM × two_stage/scratch), tạo 90 lượt mới
code\so_sanh_tl.bat   tạo SoSanh_TransferLearning.xlsx
```
