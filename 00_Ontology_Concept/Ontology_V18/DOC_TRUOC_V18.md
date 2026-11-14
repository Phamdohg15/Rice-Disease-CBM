# Ontology bệnh lúa V1.8 – đọc trước

Ontology này dựng lại theo **Bang_concept_v1_8.xlsx**. Thư mục cũ `D:\Thesis\Ontology\Ontology` (V1.6) **không bị sửa**.

## Có gì trong thư mục

| Tệp | Dùng để |
|---|---|
| `Ontology_benh_lua_V18.owl` | Mở trong Protégé (File → Open) |
| `Ontology_benh_lua_V18.ttl` | Cùng đồ thị, định dạng Turtle (không cần mở cả hai) |
| `cbm_mapping_V18.json` | Code CBM đọc file này: danh sách concept, bottleneck, tổ hợp, ràng buộc nhãn |
| `Concept_V18.csv`, `Bang_chung_V18.csv`, `To_hop_V18.csv` | Bảng xuất, mở bằng Excel |
| `Truy_van_V18.sparql` | 4 truy vấn mẫu; trong Protégé chạy từng khối ở tab SPARQL Query |
| `tra_cuu_V18.py` | Tra nhanh: `python tra_cuu_V18.py C10`, `P03`, `E16`, `"Khô vằn"` |
| `kiem_tra_ontology_V18.py` | Kiểm tra tự động; ghi `Bao_cao_kiem_tra_V18.json` |
| `build_ontology_v18.py`, `rdfmini.py` | Dựng lại ontology khi bảng concept hoặc nhãn thay đổi |

Chỉ cần Python 3 có pandas + openpyxl (đã có trong môi trường CBM). Không bắt buộc rdflib; nếu có, script kiểm tra sẽ kiểm thêm bằng rdflib và chạy SPARQL.

```powershell
cd D:\Thesis\Ontology\Ontology_V18
python kiem_tra_ontology_V18.py --labels "D:\Thesis\Gan Nhan\Tong_hop_Train_V18.csv"
# Sau khi sửa bảng concept hoặc rà lại nhãn:
python build_ontology_v18.py "D:\Thesis\Concept\Bang_concept_v1_8.xlsx" "D:\Thesis\Gan Nhan\Tong_hop_Train_V18.csv" .
```

## Thay đổi so với ontology V1.6

| | V1.6 (cũ) | V1.8 (mới) |
|---|---|---|
| Concept | 17 mã, đánh số theo V1.6 | 16 mã C01–C16 theo V1.8 |
| Nguồn | 2 sách (Giáo trình + IRRI 2025), 55 bằng chứng | 1 sách: Giáo trình Cây lúa 2008, 27 bằng chứng E01–E27 |
| Luật | 14 luật R01–R14 “tăng mạnh / rất mạnh” (đều tắt) | Bỏ; thay bằng 4 tổ hợp nghiên cứu P01–P04 |
| Lá khỏe | Concept C17, coi như một “bệnh” | Lớp ảnh riêng (ImageLevelLabel) gắn với C16, không phải bệnh |
| Liên kết mô hình | Không có | Mỗi concept có cờ bottleneck, số nhãn train; mỗi tổ hợp có cờ “đánh giá được từ mô hình” |

Ánh xạ mã V1.6 → V1.8: C08 cũ (nhiều đốm rời rạc) bị bỏ; C09→C08, C10→C09, C11→C10, C12→C11, C13→C12, C14→C13, C15→C14, C16→C15, C17→C16; C17 V1.7 (vết xám) gộp vào C10.

## Cấu trúc

- **Lớp:** RiceDisease (FungalDisease / BacterialDisease), ImageLevelLabel, OutOfScopeDisease, VisualConcept, Evidence, SourceBook, CandidateCombination, ObservationUnit.
- **Quan hệ chính:** `describedForDisease` (concept → bệnh, *không độc quyền, không phải điều kiện cần/đủ*), `hasEvidence`, `evidenceFrom`, `requiresConcept` / `optionalConcept`, `combinationScope`.
- Không có SWRL, không có tiên đề điều kiện đủ chẩn đoán. Bật reasoner không tạo ra chẩn đoán.

## Concept trong bottleneck CBM

Ngưỡng của sheet `Bottleneck_v1_8`: ≥ 80 ảnh nhãn 1 **và** ≥ 80 ảnh nhãn 0 trên train.

- **Đưa vào (13):** C01, C03, C04, C05, C06, C07, C09, C10, C11, C12, C14, C15, C16
- **Chưa đủ dữ liệu (3):** C02 (33 ảnh = 1), C08 (8), C13 (0 – toàn bộ là 9)

## Điểm cần biết khi báo cáo

1. **Chỉ P03 (Bạc lá) đánh giá trọn vẹn được từ mô hình.** P01 cần C02, P02 cần C08, P04 cần C13 – cả ba chưa vào bottleneck. Báo cáo từng ảnh sẽ ghi “khớp k/n concept đo được; thiếu C.. chưa đo”, không kết luận tổ hợp không thỏa.
2. **P04 là tổ hợp vết khô vằn trên bẹ**, trong khi dữ liệu chủ yếu là ảnh phiến lá. Nếu thầy đồng ý, có thể thêm một tổ hợp khô vằn trên phiến lá (C10 + C03 + C14 hoặc C15). Lưu ý: E16–E17 trong sách mô tả vết *trên bẹ*, nên đây là diễn giải, cần thầy duyệt; ontology V1.8 **chưa** thêm tổ hợp này.
3. **Nhãn cấp ảnh:** P01–P04 yêu cầu các dấu thuộc cùng một vết/vùng. Nhãn và dự đoán CBM ở cấp ảnh nên chỉ được gọi là “phù hợp ở cấp ảnh”.
4. **Tên tác nhân bạc lá** (E26 “pv. oyzae”) vẫn cần xem hình trang gốc để phân biệt lỗi in và lỗi nhận dạng chữ.
5. Số nhãn train trong ontology chụp tại thời điểm dựng. Sau mỗi vòng rà nhãn (C01, C02, C07, C08, C13), chạy lại `build_ontology_v18.py` để cập nhật bottleneck.

## Chưa thực hiện

Mở bằng Protégé trên máy thật; chạy reasoner OWL DL (HermiT/Pellet); thẩm định chuyên gia bệnh học; đo độ chính xác giải thích của mô hình.
