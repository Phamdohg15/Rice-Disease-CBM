# 03 – Ablation: CBM 14 concept so với 16 concept

## Mục đích

Kiểm tra việc loại C08 (vòng đồng tâm) và C13 (tổn thương trên bẹ lá) khỏi bottleneck có ảnh hưởng đến kết quả chẩn đoán
và chất lượng dự đoán các concept còn lại hay không.

## Thiết lập

- Huấn luyện lại CBM với đủ 16 concept; mọi thiết lập khác giữ nguyên như 02 (backbone, siêu tham số, seed 42/43/44).
  `pos_weight` của C08 tính như các concept khác (544/7, bị chặn ở 10).
- So sánh ghép cặp (backbone, seed) với 15 lượt CBM 14 concept của thí nghiệm 02.
- AUC/F1 concept trung bình chỉ tính trên 14 concept chung.
- Lệnh: `python code/run.py --backbone <tên> --mode cbm --seed <seed> --concept-set 16`; tổng hợp: `python code/so_sanh_14_16.py`.
- Thời gian chạy: 14/11/2026 13:00 → 16:35 (15 lượt).

## Kết quả

| Backbone | Test macro-F1, 14 concept | Test macro-F1, 16 concept | Δ (16 − 14) |
|---|---|---|---|
| ResNet18 | 0,8784 ± 0,012 | 0,8547 ± 0,024 | −0,024 |
| DenseNet121 | 0,8868 ± 0,024 | 0,9010 ± 0,017 | +0,014 |
| EfficientNet-B0 | 0,8565 ± 0,005 | 0,8932 ± 0,026 | +0,037 |
| ConvNeXt-Tiny | 0,8934 ± 0,059 | 0,9168 ± 0,036 | +0,023 |
| MobileNetV2 | 0,8586 ± 0,038 | 0,8613 ± 0,032 | +0,003 |
| **Trung bình** | **0,8747** | **0,8854** | **+0,011** |

- Bản 16 concept cao hơn ở 9/15 cặp, thấp hơn ở 6/15; Wilcoxon cặp đôi p = 0,52; độ lệch chuẩn của chênh lệch theo cặp 0,046.
  Riêng cặp ConvNeXt-Tiny seed 44 chênh +0,127 do lượt 14 concept chạy kém (xem 02); bỏ cặp này, chênh lệch trung bình còn khoảng +0,002.
- AUC trung bình 14 concept chung: 0,9188 (14 concept) và 0,9201 (16 concept).
- C08: tập val và test không có ảnh nhãn 1, nhưng mô hình 16 concept báo có C08 trên 5–19 ảnh test (trung bình 14/126).
- C13: tập val chỉ có 3 ảnh có nhãn (đều là 1) nên ngưỡng giữ mặc định 0,5; mô hình báo có C13 trên 25–100 ảnh test
  (trung bình 61/126) trong khi chỉ 5 ảnh test có nhãn 1. AUC của C13 tính trên 8 ảnh nên không đáng tin.

**Kết luận:** thêm C08, C13 không cải thiện chẩn đoán một cách có ý nghĩa thống kê, nhưng tạo thêm dấu hiệu báo nhầm trong phần giải thích.
Bottleneck giữ 14 concept; C08 và C13 vẫn giữ trong ontology. Thí nghiệm này không chứng minh được mô hình học được C08, C13,
vì val/test thiếu mẫu để đánh giá hai concept đó.

## File

- `SoSanh_14_vs_16_concept.xlsx`: 7 sheet (Doc_truoc, Tong_hop, Tung_luot, AUC_tung_concept, F1_theo_lop, C08_C13, Bieu_do).
- `so_sanh_14_vs_16.png`: hình cho luận văn (thanh sai số = độ lệch chuẩn qua 3 seed).
- `runs_16c/<backbone>_cbm_seed<seed>/`: kết quả từng lượt 16 concept.
