# 04 – Transfer learning

## Mục đích

So sánh các cách tận dụng trọng số pretrained ImageNet khi tập train nhỏ (587 ảnh), trên cả baseline và CBM,
5 backbone × 3 seed. Ý tưởng hai giai đoạn lấy từ notebook `Transfer_Learning.ipynb` (Keras) của thầy hướng dẫn,
được cài lại bằng PyTorch trong bộ code chung.

## Bốn chiến lược

| Mã | Chiến lược | Thiết lập |
|---|---|---|
| A | Fine-tune toàn bộ (cách đang dùng) | Kết quả của 01 và 02, không chạy lại |
| B | Hai giai đoạn | GĐ1: đóng băng backbone, chỉ học phần đầu mới, lr 1e-3, tối đa 30 epoch, dừng sớm 10. GĐ2: từ mô hình tốt nhất của GĐ1, mở các khối stride-32, BatchNorm vẫn đóng băng, lr 1e-5, tối đa 30 epoch, dừng sớm 10 |
| D | Chỉ đóng băng backbone | Mô hình tốt nhất sau GĐ1 của B (không chạy thêm) |
| C | Huấn luyện từ đầu | Không dùng trọng số ImageNet; mọi thiết lập khác giống A (giữ nguyên lr để chỉ khác một yếu tố) |

Phần đầu mới ở GĐ1: baseline là lớp Linear cuối; CBM gồm chiếu 1×1, Transformer, truy vấn concept, đầu concept và MLP.
Khối mở ở GĐ2 (mọi khối ở độ phân giải thấp nhất, stride 32): ResNet18 `layer4`; DenseNet121 `transition3`, `denseblock4`, `norm5`;
EfficientNet-B0 `features.6–8`; ConvNeXt-Tiny `features.6–7`; MobileNetV2 `features.14–18`.

Các yếu tố giữ chung: dữ liệu 587/126/126 ảnh, tiền xử lý và augmentation, 14 concept, seed 42/43/44,
cách chọn mô hình (val macro-F1, sau đó val CE) và cách chọn ngưỡng concept (trên val, áp dụng cho test).

## Lệnh và thời gian chạy

```
code\1_thu_tl.bat     chạy thử 2 epoch
code\chay_tl.bat      60 lệnh (5 backbone × 3 seed × baseline/CBM × two_stage/scratch), tạo 90 lượt mới
code\so_sanh_tl.bat   tạo SoSanh_TransferLearning.xlsx
```

Bắt đầu 14/11/2026 20:05, bảng tổng hợp tạo lúc 15/11/2026 09:07. Đủ 90/90 lượt (mỗi chiến lược B, C, D: 30 lượt).

## Kết quả

Mỗi ô là trung bình ± độ lệch chuẩn qua 3 seed.

### Test macro-F1 – baseline

| Backbone | A. Fine-tune toàn bộ | B. Hai giai đoạn | D. Đóng băng | C. Từ đầu | B − A | D − A | C − A |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,8470 ± 0,013 | 0,7891 ± 0,013 | 0,7200 ± 0,030 | 0,6702 ± 0,047 | −0,058 | −0,127 | −0,177 |
| DenseNet121 | 0,9033 ± 0,005 | 0,8231 ± 0,005 | 0,7737 ± 0,007 | 0,7262 ± 0,072 | −0,080 | −0,130 | −0,177 |
| EfficientNet-B0 | 0,8885 ± 0,027 | 0,8536 ± 0,004 | 0,7625 ± 0,003 | 0,5865 ± 0,018 | −0,035 | −0,126 | −0,302 |
| ConvNeXt-Tiny | 0,9326 ± 0,012 | 0,8714 ± 0,018 | 0,8039 ± 0,013 | 0,5364 ± 0,069 | −0,061 | −0,129 | −0,396 |
| MobileNetV2 | 0,8687 ± 0,025 | 0,8179 ± 0,018 | 0,7575 ± 0,008 | 0,5920 ± 0,004 | −0,051 | −0,111 | −0,277 |

### Test macro-F1 – CBM

| Backbone | A. Fine-tune toàn bộ | B. Hai giai đoạn | D. Đóng băng | C. Từ đầu | B − A | D − A | C − A |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,8784 ± 0,012 | 0,8087 ± 0,009 | 0,8096 ± 0,020 | 0,6121 ± 0,037 | −0,070 | −0,069 | −0,266 |
| DenseNet121 | 0,8868 ± 0,024 | 0,7877 ± 0,025 | 0,8071 ± 0,039 | 0,6743 ± 0,051 | −0,099 | −0,080 | −0,213 |
| EfficientNet-B0 | 0,8565 ± 0,005 | 0,8679 ± 0,009 | 0,8697 ± 0,007 | 0,5766 ± 0,008 | +0,011 | +0,013 | −0,280 |
| ConvNeXt-Tiny | 0,8934 ± 0,059 | 0,8993 ± 0,008 | 0,8890 ± 0,010 | 0,6150 ± 0,018 | +0,006 | −0,004 | −0,278 |
| MobileNetV2 | 0,8586 ± 0,038 | 0,8236 ± 0,034 | 0,8256 ± 0,043 | 0,5237 ± 0,118 | −0,035 | −0,033 | −0,335 |

### AUC concept trung bình trên test (14 concept) – CBM

| Backbone | A. Fine-tune toàn bộ | B. Hai giai đoạn | D. Đóng băng | C. Từ đầu | B − A | D − A | C − A |
|---|---|---|---|---|---|---|---|
| ResNet18 | 0,9134 ± 0,005 | 0,9012 ± 0,006 | 0,9036 ± 0,006 | 0,8649 ± 0,015 | −0,012 | −0,010 | −0,049 |
| DenseNet121 | 0,9192 ± 0,013 | 0,9021 ± 0,008 | 0,9024 ± 0,006 | 0,8689 ± 0,005 | −0,017 | −0,017 | −0,050 |
| EfficientNet-B0 | 0,9127 ± 0,004 | 0,9065 ± 0,002 | 0,9076 ± 0,004 | 0,8306 ± 0,016 | −0,006 | −0,005 | −0,082 |
| ConvNeXt-Tiny | 0,9288 ± 0,011 | 0,9239 ± 0,004 | 0,9218 ± 0,003 | 0,8233 ± 0,005 | −0,005 | −0,007 | −0,105 |
| MobileNetV2 | 0,9201 ± 0,011 | 0,9098 ± 0,005 | 0,9100 ± 0,005 | 0,8203 ± 0,032 | −0,010 | −0,010 | −0,100 |

Trung bình 15 lượt (test macro-F1):

| | A | B | D | C |
|---|---|---|---|---|
| Baseline | 0,8880 | 0,8310 | 0,7635 | 0,6223 |
| CBM | 0,8747 | 0,8375 | 0,8402 | 0,6003 |

So sánh ghép cặp (backbone, seed) với A, Wilcoxon cặp đôi trên 15 cặp:

- Baseline: B − A = −0,057, D − A = −0,125, C − A = −0,266; cả ba chiến lược thấp hơn A ở 15/15 cặp, p < 0,001.
- CBM: B − A = −0,037 (cao hơn ở 5 cặp, thấp hơn ở 10 cặp, p = 0,030); D − A = −0,035 (4/11, p = 0,035);
  C − A = −0,274 (thấp hơn ở 15/15 cặp, p < 0,001).

## Nhận xét

1. Fine-tune toàn bộ (A) cho test macro-F1 trung bình cao nhất ở cả baseline và CBM, và AUC concept cao nhất ở cả 5 backbone.
   Với CBM, EfficientNet-B0 và ConvNeXt-Tiny là ngoại lệ về F1: B và D bằng hoặc cao hơn A một chút (+0,004 đến +0,013),
   nhưng với ConvNeXt-Tiny chênh lệch nhỏ hơn độ lệch chuẩn của A (0,059, do lượt seed 44 chạy kém, xem 02).
2. Đóng băng backbone làm baseline giảm mạnh hơn CBM (−0,125 so với −0,035). Khi đóng băng, baseline chỉ học
   2,6–6,4 nghìn tham số của lớp Linear cuối, còn CBM vẫn học khoảng 1,25–1,58 triệu tham số của phần đầu
   (sheet `Chi_phi`). Đây là một cách giải thích có thể, thí nghiệm này chưa tách riêng được yếu tố đó.
3. Với CBM, GĐ2 của B (mở khối stride-32, lr 1e-5) gần như không cải thiện so với D (0,8375 so với 0,8402).
   Thí nghiệm chỉ thử một giá trị lr ở GĐ2 nên chưa kết luận được là mở thêm khối không có ích.
4. Huấn luyện từ đầu (C) thấp nhất ở mọi backbone (F1 giảm 0,18–0,40; AUC concept giảm 0,05–0,11). C dùng lại lr của A,
   chưa được tinh chỉnh riêng, nên con số này phản ánh C dưới cùng thiết lập với A, không phải mức tốt nhất có thể đạt khi học từ đầu.
5. Chi phí: D nhanh nhất (khoảng 8–10 phút/lượt); B mất khoảng 14–17 phút/lượt, tương đương hoặc lâu hơn A (9–16 phút).

**Kết luận:** trên tập dữ liệu này, trọng số ImageNet là cần thiết, và fine-tune toàn bộ backbone (A) vẫn là lựa chọn
cho mô hình chính. Hai giai đoạn theo notebook của thầy không vượt A khi tính trung bình 5 backbone.

## File

- `SoSanh_TransferLearning.xlsx`: 7 sheet (Doc_truoc, Tong_hop, Chi_phi, Tung_luot, AUC_tung_concept, F1_theo_lop, Bieu_do).
- `so_sanh_transfer_learning.png`: hình cho luận văn (thanh sai số = độ lệch chuẩn qua 3 seed).
- `runs_tl/<backbone>_<baseline|cbm>_<frozen|two_stage|scratch>_seed<seed>/`: kết quả từng lượt (90 thư mục).
  Lượt `frozen` là mô hình sau GĐ1 của `two_stage`. Không đưa lên `best.pt`, `*_predictions.csv` và các file `val_*`
  (số liệu val chính có trong `metrics.json`), giống quy ước của 01–03.
