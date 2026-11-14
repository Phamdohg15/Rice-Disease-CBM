# 00 – Ontology và tập concept V1.8

## Nội dung

- `Ontology_V18/`: ontology bệnh lúa V1.8 (OWL, Turtle), bảng concept, bảng bằng chứng E01–E27, 4 tổ hợp P01–P04,
  truy vấn SPARQL mẫu, script dựng và kiểm tra ontology. Đọc `Ontology_V18/DOC_TRUOC_V18.md` trước.
- Nguồn tri thức: Giáo trình Cây lúa (2008), 27 bằng chứng.
- `Bang_concept_v1_8.xlsx`: bảng gốc dùng để dựng ontology. Gồm các sheet định nghĩa 16 concept, tổ hợp ứng viên,
  bằng chứng E01–E27, lịch sử thay đổi V1.6 → V1.7 → V1.8 và hướng dẫn gán nhãn.
  Sheet `Bottleneck_v1_8` được tính ngày 04/11/2026 trên 2650 ảnh, trước vòng rà lại nhãn, nên số liệu khác bảng bên dưới.

## 16 concept và tổ hợp

| Tổ hợp | Bệnh | Concept bắt buộc | Concept tùy chọn |
|---|---|---|---|
| P01 | Đạo ôn | C01 + C02 + C03 | C05 |
| P02 | Đốm nâu | C06 + C07 + C03 + C08 | |
| P03 | Bạc lá | C09 + C10 + C11 | C04 |
| P04 | Khô vằn | C10 + C03 + C13 | C06; C15 |

Các tổ hợp mô tả dấu hiệu, không phải luật IF–THEN; thiếu tổ hợp không loại trừ bệnh. Tên đầy đủ của từng concept: `Ontology_V18/Concept_V18.csv`.

## Chọn concept vào bottleneck (chốt ngày 08/11/2026)

Quy tắc: concept chỉ vào bottleneck khi tập train có **≥ 80 ảnh nhãn 1 và ≥ 80 ảnh nhãn 0**.
Thống kê trên tập train cuối cùng (587 ảnh, file `Train_V18_FINAL_587.csv`, không đưa lên repo):

| Concept | n1 | n0 | n9 (không xác định) | Vào bottleneck |
|---|---|---|---|---|
| C01 | 95 | 446 | 46 | Có |
| C02 | 85 | 473 | 29 | Có |
| C03 | 345 | 187 | 55 | Có |
| C04 | 440 | 138 | 9 | Có |
| C05 | 158 | 357 | 72 | Có |
| C06 | 216 | 350 | 21 | Có |
| C07 | 89 | 450 | 48 | Có |
| C08 | 7 | 544 | 36 | **Không** (quá ít nhãn 1) |
| C09 | 325 | 220 | 42 | Có |
| C10 | 209 | 344 | 34 | Có |
| C11 | 133 | 418 | 36 | Có |
| C12 | 132 | 312 | 143 | Có |
| C13 | 18 | 6 | 563 | **Không** (96% ảnh không quan sát được bẹ lá) |
| C14 | 243 | 292 | 52 | Có |
| C15 | 83 | 481 | 23 | Có |
| C16 | 103 | 484 | 0 | Có |

Bottleneck: **14 concept** (C01–C07, C09–C12, C14–C16). C08 và C13 vẫn nằm trong ontology.
Việc loại hai concept này được kiểm tra bằng thí nghiệm `03_Ablation_14_vs_16concept`.

## Lưu ý về số liệu trong `Ontology_V18/`

Các cột `used_in_bottleneck`, `train_n1/n0/n9` trong `Concept_V18.csv`, `cbm_mapping_V18.json` và mục "Concept trong bottleneck"
của `DOC_TRUOC_V18.md` được tính **trên bộ nhãn trước khi gán lại** (2650 ảnh). Khi đó C02 chưa đủ ngưỡng nên bottleneck có 13 concept.
Sau khi gán lại nhãn tập train (08/11/2026), C02 đạt ngưỡng; bảng ở trên là số liệu đã dùng cho mọi thí nghiệm 01–05.
