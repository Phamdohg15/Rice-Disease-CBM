"""Kiểm tra ontology bệnh lúa V1.8.

Cách dùng (chạy trong thư mục Ontology_V18):
    python kiem_tra_ontology_V18.py
    python kiem_tra_ontology_V18.py --labels "D:\\Thesis\\Gan Nhan\\Tong_hop_Train_V18.csv"

Chỉ cần Python chuẩn. Nếu máy có rdflib, script đọc thêm TTL/OWL bằng rdflib
và chạy các truy vấn trong Truy_van_V18.sparql để kiểm tra độc lập.
Kết quả ghi vào Bao_cao_kiem_tra_V18.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rdfmini import OWL, RDF, RDFS, SKOS, Graph, Literal, Namespace, URIRef  # noqa: E402

EX = Namespace("http://example.org/rice-cbm/ontology/v18#")
CONCEPTS = [f"C{i:02d}" for i in range(1, 17)]
LEAF = [c for c in CONCEPTS if c not in ("C13", "C16")]


class Checker:
    def __init__(self):
        self.passed, self.failed, self.warnings = [], [], []

    def check(self, ok, name, detail=""):
        (self.passed if ok else self.failed).append(name + (f" — {detail}" if detail and not ok else ""))
        return ok

    def warn(self, text):
        self.warnings.append(text)


def typed(g, cls):
    return set(g.subjects(RDF.type, cls))


def one(g, s, p):
    v = g.objects(s, p)
    return v[0] if len(v) == 1 else None


def check_graph(g, ck, mapping):
    concepts = typed(g, EX.VisualConcept)
    codes = sorted(str(one(g, c, EX.code)) for c in concepts)
    ck.check(codes == CONCEPTS, "Đủ 16 concept C01–C16, không còn C17", f"thấy {codes}")
    ck.check(EX.C17 not in {s for s, _, _ in g}, "Không còn cá thể C17 (đã gộp vào C10)")

    for c in sorted(concepts):
        code = str(one(g, c, EX.code))
        has = all(g.objects(c, p) for p in (RDFS.label, SKOS.definition, EX.assignedPer, EX.conceptStatus, EX.usedInBottleneck,
                                              EX.rule1, EX.rule0, EX.rule9))
        ck.check(has, f"{code} có đủ tên, định nghĩa, đơn vị gán, trạng thái, quy tắc 1/0/9")

    diseases = typed(g, EX.RiceDisease)
    ck.check(len(diseases) == 4, "Đúng 4 bệnh mục tiêu", f"thấy {len(diseases)}")
    ck.check(len(typed(g, EX.ImageLevelLabel)) == 1, "Lá khỏe là nhãn cấp ảnh riêng, không phải bệnh")
    bad = [(s, o) for s, p, o in g if p == EX.describedForDisease and o not in diseases]
    ck.check(not bad, "Mọi quan hệ concept–bệnh trỏ tới bệnh mục tiêu", str(bad))

    evidence = typed(g, EX.Evidence)
    ck.check(len(evidence) == 27, "Đủ 27 bằng chứng E01–E27", f"thấy {len(evidence)}")
    books = typed(g, EX.SourceBook)
    ck.check(books == {EX.S01}, "Chỉ một sách nguồn S01")
    ck.check(all(g.objects(e, EX.evidenceFrom) == [EX.S01] for e in evidence), "Mọi bằng chứng trích từ S01")
    ck.check(all(one(g, e, EX.quote) is not None and one(g, e, EX.printedPage) is not None for e in evidence),
             "Mọi bằng chứng có trích đoạn và trang in")

    dangling = [(s, o) for s, p, o in g if p in (EX.hasEvidence, EX.combinationEvidence) and o not in evidence]
    ck.check(not dangling, "Không có tham chiếu bằng chứng bị đứt", str(dangling))
    no_ev = [str(one(g, c, EX.code)) for c in concepts if not g.objects(c, EX.hasEvidence) and str(one(g, c, EX.code)) != "C16"]
    ck.check(not no_ev, "Mọi concept (trừ C16 quy ước dự án) có ít nhất một bằng chứng", str(no_ev))
    used_ev = {o for s, p, o in g if p in (EX.hasEvidence, EX.combinationEvidence)}
    unused = sorted(str(one(g, e, EX.code)) for e in evidence - used_ev)
    if unused:
        ck.warn("Bằng chứng không gắn concept/tổ hợp (tên tác nhân, diễn tiến, bối cảnh): " + ", ".join(unused))

    combos = typed(g, EX.CandidateCombination)
    ck.check(len(combos) == 4, "Đúng 4 tổ hợp P01–P04")
    for p in sorted(combos):
        code = str(one(g, p, EX.code))
        req = g.objects(p, EX.requiresConcept)
        ck.check(len(req) >= 2 and all(r in concepts for r in req), f"{code}: concept bắt buộc hợp lệ")
        ck.check(len(g.objects(p, EX.combinationFor)) == 1 and one(g, p, EX.combinationFor) in diseases, f"{code}: gắn đúng một bệnh")
        ck.check(str(one(g, p, EX.combinationStatus)) == "RESEARCH_QUERY_NOT_DIAGNOSTIC", f"{code}: đánh dấu là truy vấn nghiên cứu")
        used = {c for c in concepts if str(one(g, c, EX.usedInBottleneck)) == "true"}
        expected = all(r in used for r in req)
        ck.check(str(one(g, p, EX.evaluableFromModel)) == str(expected).lower(), f"{code}: cờ đánh giá được từ mô hình khớp bottleneck")

    classes = typed(g, OWL.Class)
    individuals = typed(g, OWL.NamedIndividual)
    ck.check(not classes & individuals, "Không IRI nào vừa là lớp vừa là cá thể")
    forbidden = {OWL.equivalentClass, OWL.hasValue, OWL.someValuesFrom, OWL.allValuesFrom,
                 URIRef("http://www.w3.org/2003/11/swrl#Imp")}
    ck.check(not any(p in forbidden or o in forbidden for _, p, o in g), "Không có SWRL hoặc tiên đề điều kiện đủ chẩn đoán")
    declared = typed(g, OWL.ObjectProperty) | typed(g, OWL.DatatypeProperty)
    used_props = {p for _, p, _ in g if p.startswith(str(EX))}
    ck.check(used_props <= declared, "Mọi thuộc tính ex: đều được khai báo", str(sorted(used_props - declared)))

    # khớp file ánh xạ cho CBM
    bn_graph = sorted(str(one(g, c, EX.code)) for c in concepts if str(one(g, c, EX.usedInBottleneck)) == "true")
    ck.check(bn_graph == mapping["bottleneck_concepts"], "Danh sách bottleneck trong ontology khớp cbm_mapping_V18.json")
    for code, info in mapping["concepts"].items():
        c = EX[code]
        names = sorted(str(one(g, d, RDFS.label)) for d in g.objects(c, EX.describedForDisease))
        ck.check(names == sorted(info["described_for"]), f"{code}: quan hệ bệnh khớp file ánh xạ", f"{names} vs {info['described_for']}")
    return bn_graph


def check_labels(path, g, ck):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ck.check(all(c in rows[0] for c in CONCEPTS), "File nhãn có đủ cột C01–C16")
    bad_vals = sum(1 for r in rows for c in CONCEPTS if r[c] not in ("0", "1", "9"))
    ck.check(bad_vals == 0, "Nhãn chỉ gồm 0/1/9", f"{bad_vals} ô sai")
    viol = [r["file_name"] for r in rows if r["C16"] == "1" and any(r[c] != "0" for c in LEAF)]
    ck.check(not viol, "C16 = 1 kéo theo concept bệnh trên phiến lá = 0", f"{len(viol)} ảnh vi phạm, ví dụ {viol[:3]}")
    if "C10_v17" in rows[0] and "C17_v17" in rows[0]:
        def merged(a, b):
            return "1" if "1" in (a, b) else ("0" if a == b == "0" else "9")
        mism = [r["file_name"] for r in rows if r["C10"] != merged(r["C10_v17"], r["C17_v17"])]
        ck.check(not mism, "C10 V1.8 = gộp đúng C10 và C17 của V1.7", f"{len(mism)} ảnh lệch")
    for code in CONCEPTS:
        n1 = sum(r[code] == "1" for r in rows); n0 = sum(r[code] == "0" for r in rows)
        g1 = int(str(one(g, EX[code], EX.trainPositive))); g0 = int(str(one(g, EX[code], EX.trainNegative)))
        if (n1, n0) != (g1, g0):
            ck.warn(f"{code}: số nhãn trong file hiện tại ({n1}/{n0}) khác số đã ghi trong ontology ({g1}/{g0}) — dựng lại ontology sau khi rà nhãn")
    healthy = [r for r in rows if r.get("disease") == "Lá khỏe"]
    if healthy:
        lesion = [r["file_name"] for r in healthy if r["C16"] == "0"]
        unk = sum(r["C16"] == "9" for r in healthy)
        if lesion:
            ck.warn(f"{len(lesion)} ảnh lớp Lá khỏe có C16 = 0 (thấy tổn thương) — cần xem lại: {', '.join(lesion[:5])}{' …' if len(lesion) > 5 else ''}")
        if unk:
            ck.warn(f"{unk} ảnh lớp Lá khỏe có C16 = 9 (chưa cắt/chọn lá)")
    return len(rows)


def check_with_rdflib(ck):
    try:
        import rdflib
    except ImportError:
        ck.warn("Máy không có rdflib: bỏ qua kiểm tra độc lập bằng rdflib và chạy SPARQL (pip install rdflib để bật)")
        return None
    a = rdflib.Graph().parse(HERE / "Ontology_benh_lua_V18.ttl", format="turtle")
    b = rdflib.Graph().parse(HERE / "Ontology_benh_lua_V18.owl", format="xml")
    from rdflib.compare import isomorphic
    ck.check(isomorphic(a, b), "rdflib: TTL và OWL là cùng một đồ thị")
    text = (HERE / "Truy_van_V18.sparql").read_text(encoding="utf-8")
    blocks = [q.strip() for q in text.split("# ---") if "SELECT" in q]
    for i, q in enumerate(blocks, 1):
        rows = list(a.query(q))
        ck.check(len(rows) > 0, f"rdflib: truy vấn SPARQL số {i} trả kết quả", "rỗng")
    return rdflib.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None, help="Đường dẫn Tong_hop_Train_V18.csv (tùy chọn)")
    args = ap.parse_args()
    ck = Checker()

    g_xml = Graph.parse_rdfxml(HERE / "Ontology_benh_lua_V18.owl")
    ttl_text = (HERE / "Ontology_benh_lua_V18.ttl").read_text(encoding="utf-8")
    g_xml.bind("ex", EX)
    ck.check(g_xml.to_turtle() == ttl_text, "OWL (RDF/XML) và TTL biểu diễn cùng đồ thị")
    mapping = json.loads((HERE / "cbm_mapping_V18.json").read_text(encoding="utf-8"))
    bottleneck = check_graph(g_xml, ck, mapping)
    n_labels = check_labels(args.labels, g_xml, ck) if args.labels else None
    rdflib_version = check_with_rdflib(ck)

    report = {
        "status": "PASS" if not ck.failed else "FAIL",
        "n_triples": len(g_xml), "n_checks": len(ck.passed) + len(ck.failed),
        "passed": ck.passed, "failed": ck.failed, "warnings": ck.warnings,
        "bottleneck_concepts": bottleneck, "label_rows_checked": n_labels, "rdflib": rdflib_version,
        "not_performed": ["Mở bằng Protégé desktop", "Reasoner OWL DL (HermiT/Pellet)", "Thẩm định chuyên gia bệnh học",
                          "Đánh giá độ chính xác giải thích của mô hình"],
    }
    (HERE / "Bao_cao_kiem_tra_V18.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['status']}: {len(ck.passed)}/{report['n_checks']} kiểm tra đạt, {len(ck.warnings)} lưu ý")
    for f in ck.failed: print("  LỖI:", f)
    for w in ck.warnings: print("  Lưu ý:", w)
    return 0 if not ck.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
