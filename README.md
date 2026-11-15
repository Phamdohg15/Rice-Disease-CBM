# Rice-Disease-CBM

Mô hình Concept Bottleneck có giải thích cho chẩn đoán bệnh trên lá lúa (luận văn thạc sĩ).

Bài toán gồm 5 lớp ảnh: Đạo ôn, Đốm nâu, Bạc lá, Khô vằn, Lá khỏe. Mô hình chính là
**CNN backbone + Multi-Head Attention theo concept + Concept Bottleneck (14 concept V1.8) + MLP**, huấn luyện joint.
Tầng concept được đối chiếu với ontology bệnh lúa V1.8 (tổ hợp P01–P04).

## Bảng tiến trình

Thứ tự folder là thứ tự thực hiện. Ngày chạy lấy từ trường `finished` trong `metrics.json` của từng lượt.

| # | Folder | Nội dung | Thời gian chạy | Trạng thái | Kết quả chính (test, trung bình 3 seed) |
|---|---|---|---|---|---|
| 00 | `00_Ontology_Concept` | Ontology V1.8: 16 concept, tổ hợp P01–P04; chọn 14 concept vào bottleneck | 08/11/2026 (chốt nhãn) | Xong | 14/16 concept đạt ngưỡng ≥ 80 ảnh nhãn 1 và ≥ 80 ảnh nhãn 0; loại C08, C13 |
| 01 | `01_Baseline_5CNN` | 5 CNN baseline × 3 seed | 08–09/11/2026 | Xong | Macro-F1 từ 0,8470 (ResNet18) đến 0,9326 (ConvNeXt-Tiny) |
| 02 | `02_CBM_5CNN_14concept` | 5 CNN + Attention + CBM + MLP × 3 seed, so với 01 | 09/11/2026 | Xong | Macro-F1 từ 0,8565 đến 0,8934; AUC concept 0,913–0,929 |
| 03 | `03_Ablation_14_vs_16concept` | Huấn luyện lại CBM với 16 concept để kiểm tra việc loại C08, C13 | 14/11/2026 | Xong | Δ macro-F1 (16 − 14) = +0,011, Wilcoxon p = 0,52 |
| 04 | `04_TransferLearning` | So sánh chiến lược transfer learning (fine-tune toàn bộ / hai giai đoạn / đóng băng / từ đầu) trên baseline và CBM | 14–15/11/2026 | Xong | Fine-tune toàn bộ cao nhất: CBM 0,8747; hai giai đoạn 0,8375; đóng băng 0,8402; từ đầu 0,6003 |
| 05 | `05_App_ChanDoan` | Ứng dụng web chẩn đoán 1 ảnh: bệnh, dấu hiệu, heatmap, đối chiếu ontology | 09/11/2026 | Xong | Dùng mô hình ConvNeXt-Tiny CBM seed 43 |

Mỗi folder có `README.md` riêng ghi mục đích, thiết lập, lệnh đã chạy và kết quả.

## Cấu trúc

```
code/                          Code dùng chung cho 01–04 (một bộ code, khác nhau ở lệnh chạy)
00_Ontology_Concept/           Ontology V1.8 + thống kê nhãn concept (không chứa file nhãn từng ảnh)
01_Baseline_5CNN/              Kết quả baseline
02_CBM_5CNN_14concept/         Kết quả CBM 14 concept + bảng so sánh với baseline
03_Ablation_14_vs_16concept/   Kết quả CBM 16 concept + so sánh 14 vs 16
04_TransferLearning/           Kết quả 4 chiến lược transfer learning + bảng so sánh
05_App_ChanDoan/               Code ứng dụng chẩn đoán
docs/                          Báo cáo nguồn dữ liệu RiceyLeafDisease
```

Mỗi lượt chạy (thư mục `runs/<backbone>_<loại>_seed<seed>/`) gồm: `config.json` (toàn bộ cấu hình và phiên bản thư viện),
`log.txt`, `history.csv` (từng epoch), `learning_curves.png`, `metrics.json`, báo cáo phân loại và ma trận nhầm lẫn trên test;
với CBM có thêm `test_concept_metrics.csv` và `concept_thresholds.json`.

## Dữ liệu

- 839 ảnh lá lúa, 5 lớp, chia cố định: train 587 / val 126 / test 126.
  - Train: Đạo ôn 125, Đốm nâu 120, Bạc lá 110, Khô vằn 129, Lá khỏe 103.
  - Val và test (mỗi tập): Đạo ôn 27, Đốm nâu 26, Bạc lá 24, Khô vằn 27, Lá khỏe 22.
- Nhãn concept V1.8 (0 = không có, 1 = có, 9 = không xác định) cho từng ảnh.
- **Ảnh và file nhãn concept không được đưa lên repo.**

### Nguồn ảnh

Ảnh lấy từ bộ dữ liệu **RiceyLeafDisease** (Hasan và cộng sự, 2023), công bố trên Mendeley Data, phiên bản 1,
DOI [10.17632/t46kkgh2yw.1](https://doi.org/10.17632/t46kkgh2yw.1), giấy phép CC BY 4.0.
Bộ gốc có 8 nhóm nhãn, 1.701 ảnh gốc và 5.188 ảnh tăng cường. Ảnh được chụp tại Sirajganj và Pabna (Bangladesh)
trong tháng 7–10/2023, cả ngoài ruộng và trong nhà.

Luận văn dùng 5 trong 8 nhóm: Leaf Blast (Đạo ôn), Brown Spot (Đốm nâu), Bacterial Leaf Blight (Bạc lá),
Sheath Blight (Khô vằn), Healthy Rice Leaf (Lá khỏe). Ba nhóm không dùng là Leaf Scald, Narrow Brown Leaf Spot và Rice Hispa.
Tập 839 ảnh của luận văn gồm hai nguồn:

| Nguồn | Số ảnh |
|---|---|
| RiceyLeafDisease, chỉ lấy phần ảnh gốc (Original) | 689 |
| Ảnh của viện lúa *(cần ghi tên đầy đủ của viện)* | 150 |
| **Tổng** | **839** |

Không dùng phần ảnh tăng cường (Augmented) có sẵn của RiceyLeafDisease. Vì vậy không có trường hợp ảnh tăng cường
của cùng một ảnh gốc nằm ở hai tập khác nhau. Tăng cường dữ liệu chỉ được áp dụng trực tiếp khi huấn luyện, trên tập train.

Báo cáo rà soát nguồn dữ liệu và các công trình đã dùng hoặc trích dẫn bộ ảnh: `docs/Bao_cao_data_RiceyLeafDisease.docx`.

Trích dẫn:

> Hasan, M., Khatun, S., Raihan, M. A., & Uddin, A. H. (2023). *RiceyLeafDisease: An Extensive Image Dataset of Diverse
> Rice Leaf Diseases from Different Regions of Bangladesh* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/t46kkgh2yw.1

## Những gì không có trong repo

- Ảnh dataset và file nhãn concept từng ảnh.
- Trọng số mô hình `best.pt` (mỗi file 9–117 MB; GitHub giới hạn 100 MB/file).
- File dự đoán từng ảnh (`*_predictions.csv`) và các báo cáo trên tập val của từng lượt; các số liệu val chính vẫn có trong `metrics.json`.
- Các lượt chạy thử nhanh (`runs_smoke`, `runs_16c_smoke`, `runs_tl_smoke`, `runs_cu`).

## Môi trường

Windows 11, Python 3.11.0, PyTorch 2.7.1 + CUDA 12.8, GPU NVIDIA GeForce RTX 5070. Danh sách thư viện: `requirements.txt`.

```
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
