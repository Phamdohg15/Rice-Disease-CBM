# 02 – CBM: 5 CNN + Attention + Concept Bottleneck (14 concept) + MLP

## Mục đích

Huấn luyện mô hình chính: ảnh → backbone → attention theo từng concept → 14 concept V1.8 → MLP → bệnh,
và so sánh với baseline cùng backbone (thí nghiệm 01).

## Thiết lập

- Backbone giống 01 (pretrained ImageNet, fine-tune toàn bộ, lr 1e-4). Phần CBM mới (chiếu 1×1, Transformer, 14 truy vấn concept,
  đầu concept, MLP) dùng lr 1e-3 (chọn ngày 09/11/2026).
- Loss = CE (bệnh) + λc · BCE có `pos_weight` (concept, bỏ qua nhãn 9), λc = 1. Thành phần luật (λr) = 0.
- Ngưỡng từng concept chọn trên val rồi áp dụng cho test. Huấn luyện joint.
- Các thiết lập còn lại, seed và cách chọn mô hình giống 01.
- Lệnh: `python code/run.py --backbone <tên> --mode cbm --seed <seed>`.
- Thời gian chạy: 09/11/2026 00:29 → 03:50 (15 lượt).

## Kết quả (test, trung bình ± độ lệch chuẩn qua 3 seed)

| Backbone | Baseline macro-F1 | CBM macro-F1 | Chênh lệch (CBM − baseline) | AUC concept (CBM) | F1 concept (CBM) |
|---|---|---|---|---|---|
| ResNet18 | 0,8470 | 0,8784 ± 0,0120 | +0,0314 | 0,9134 ± 0,0053 | 0,7654 ± 0,0125 |
| DenseNet121 | 0,9033 | 0,8868 ± 0,0237 | −0,0165 | 0,9192 ± 0,0134 | 0,7604 ± 0,0181 |
| EfficientNet-B0 | 0,8885 | 0,8565 ± 0,0049 | −0,0321 | 0,9127 ± 0,0043 | 0,7453 ± 0,0049 |
| ConvNeXt-Tiny | 0,9326 | 0,8934 ± 0,0590 | −0,0392 | 0,9288 ± 0,0107 | 0,7736 ± 0,0395 |
| MobileNetV2 | 0,8687 | 0,8586 ± 0,0381 | −0,0101 | 0,9201 ± 0,0107 | 0,7638 ± 0,0164 |

So với baseline, CBM cao hơn ở ResNet18 và thấp hơn ở 4 backbone còn lại; đổi lại, mô hình đưa ra dự đoán qua 14 concept có thể đọc được.
Lượt ConvNeXt-Tiny seed 44 dừng sớm ở epoch 20 (mô hình tốt nhất ở epoch 8, test F1 0,8265), làm độ lệch chuẩn của ConvNeXt-Tiny lớn.

## File

- `tong_hop/BANG_SO_SANH.md`: bảng đầy đủ (val, test, concept, số tham số, thời gian) và AUC từng concept theo backbone.
- `tong_hop/tong_hop_mean_std.csv`, `tat_ca_luot.csv`, `so_sanh_cap.csv`, `concept_auc_theo_backbone.csv`, `so_sanh.png`.
- `SoSanh_5_CBM.xlsx`, `SoSanh_Tong.xlsx`, `SoSanh_<backbone>.xlsx`: bảng so sánh dạng Excel.
- `runs/<backbone>_cbm_seed<seed>/`: kết quả từng lượt, có thêm `test_concept_metrics.csv` và `concept_thresholds.json`.
