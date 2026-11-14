"""Dựng ontology bệnh lúa V1.8 từ Bang_concept_v1_8.xlsx.

Cách dùng:
    python build_ontology_v18.py <Bang_concept_v1_8.xlsx> <Tong_hop_Train_V18.csv> <thu_muc_ra>

Đầu ra (thư mục mới, không ghi đè bản V1.6):
    Ontology_benh_lua_V18.ttl / .owl   – cùng một đồ thị, hai định dạng
    Concept_V18.csv, Bang_chung_V18.csv, To_hop_V18.csv – bản xuất bảng
    cbm_mapping_V18.json               – ánh xạ cho code CBM (concept, bottleneck, tổ hợp)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rdfmini import DCTERMS, OWL, RDF, RDFS, SKOS, XSD, Graph, Literal, Namespace, URIRef  # noqa: E402

VERSION = "1.8.0"
ONTO_IRI = "http://example.org/rice-cbm/ontology/v18"
EX = Namespace(ONTO_IRI + "#")

CONCEPTS = [f"C{i:02d}" for i in range(1, 17)]
# Concept quan sát trên phiến lá (C13 là bẹ, C16 là nhãn không tổn thương)
LEAF_DISEASE_CONCEPTS = [c for c in CONCEPTS if c not in ("C13", "C16")]

DISEASES = {
    # mã lớp dữ liệu, tên Việt, thư mục ảnh, IRI
    "Đạo ôn": ("03_dao_on_la", "Leaf Blast", "D_DaoOn"),
    "Đốm nâu": ("02_dom_nau", "Brown Spot", "D_DomNau"),
    "Bạc lá": ("01_bac_la", "Bacterial Leaf Blight", "D_BacLa"),
    "Khô vằn": ("04_kho_van", "Sheath Blight", "D_KhoVan"),
}
HEALTHY = ("05_la_khoe", "Healthy Rice Leaf", "L_LaKhoe")

# Bằng chứng nào làm căn cứ cho concept nào (đọc từ trích đoạn trong sheet Bang_chung).
CONCEPT_EVIDENCE = {
    "C01": ["E01"], "C02": ["E01"], "C03": ["E02", "E09", "E16"], "C04": ["E22"],
    "C05": ["E03"], "C06": ["E08", "E15"], "C07": ["E09"], "C08": ["E10"],
    "C09": ["E21", "E22"], "C10": ["E16", "E23"], "C11": ["E24"], "C12": ["E04"],
    "C13": ["E15", "E25"], "C14": ["E17"], "C15": ["E17"], "C16": [],
}

# Concept tùy chọn (bổ sung) của từng tổ hợp, theo cột "Diễn giải và giới hạn".
COMBO_OPTIONAL = {"P01": ["C05"], "P02": [], "P03": ["C04"], "P04": ["C06", "C15"]}
COMBO_EVIDENCE = {
    "P01": ["E01", "E02", "E03"], "P02": ["E08", "E09", "E10"],
    "P03": ["E21", "E22", "E23", "E24"], "P04": ["E15", "E16", "E17"],
}

UNIT_IRI = {
    "Từng vết": "U_TungVet",
    "Cùng vùng trên cùng lá": "U_CungVungTrenLa",
    "Bẹ lá mang tổn thương": "U_BeLa",
    "Cùng vùng trên cùng lá hoặc bẹ": "U_CungVungLaHoacBe",
    "Từng vết hoặc cùng vùng trên cùng lá/bẹ": "U_TungVetHoacVung",
    "Phần phiến lá quan sát": "U_PhienLaQuanSat",
}

MIN_POS = 80  # ngưỡng tạm trong sheet Bottleneck_v1_8
MIN_NEG = 80


def vi(text):
    return Literal(str(text).strip(), lang="vi")


def clean(v):
    return None if pd.isna(v) else str(v).strip()


def load(xlsx, labels_csv):
    x = pd.ExcelFile(xlsx)
    concepts = pd.read_excel(x, "Concept_v1_8")
    concepts = concepts[concepts["Mã"].astype(str).str.fullmatch(r"C\d{2}")].copy()
    evidence = pd.read_excel(x, "Bang_chung")
    combos = pd.read_excel(x, "To_hop_ung_vien")
    changes = pd.read_excel(x, "Thay_doi_v1_8")
    bottleneck = pd.read_excel(x, "Bottleneck_v1_8")
    bottleneck = bottleneck[bottleneck["Mã"].astype(str).str.fullmatch(r"C\d{2}")].copy()
    labels = pd.read_csv(labels_csv, encoding="utf-8-sig")
    return concepts, evidence, combos, changes, bottleneck, labels


def label_counts(labels):
    out = {}
    for c in CONCEPTS:
        v = labels[c]
        out[c] = {"n1": int((v == 1).sum()), "n0": int((v == 0).sum()), "n9": int((v == 9).sum())}
    return out


def build(xlsx, labels_csv, out_dir):
    concepts, evidence, combos, changes, bottleneck, labels = load(xlsx, labels_csv)
    assert sorted(concepts["Mã"]) == CONCEPTS, "Sheet Concept_v1_8 phải có đúng C01–C16"
    counts = label_counts(labels)

    g = Graph()
    g.bind("ex", EX); g.bind("owl", OWL); g.bind("skos", SKOS); g.bind("dcterms", DCTERMS)
    onto = URIRef(ONTO_IRI)
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, OWL.versionInfo, Literal(VERSION)))
    g.add((onto, RDFS.label, vi("Ontology bệnh lúa cho CBM – concept V1.8")))
    g.add((onto, RDFS.comment, vi(
        "16 concept C01–C16 theo Bang_concept_v1_8. Nguồn bệnh học duy nhất: S01 Giáo trình Cây lúa (2008). "
        "Tổ hợp P01–P04 là thiết kế truy vấn nghiên cứu, không phải luật chẩn đoán. "
        "Không có SWRL hoặc tiên đề điều kiện đủ.")))
    g.add((onto, OWL.priorVersion, URIRef("https://example.org/rice-v16-two-books")))

    # ---- Lớp ----
    classes = {
        "RiceDisease": "Bệnh lúa mục tiêu",
        "FungalDisease": "Bệnh do nấm",
        "BacterialDisease": "Bệnh do vi khuẩn",
        "OutOfScopeDisease": "Bệnh ngoài phạm vi (bối cảnh)",
        "ImageLevelLabel": "Nhãn biểu hiện cấp ảnh (không phải bệnh)",
        "DatasetClass": "Lớp ảnh trong dữ liệu",
        "VisualConcept": "Concept thị giác",
        "Evidence": "Bằng chứng trích từ sách",
        "SourceBook": "Sách nguồn",
        "CandidateCombination": "Tổ hợp ứng viên nghiên cứu",
        "ObservationUnit": "Đơn vị gán nhãn",
    }
    for name, label in classes.items():
        g.add((EX[name], RDF.type, OWL.Class)); g.add((EX[name], RDFS.label, vi(label)))
    g.add((EX.RiceDisease, RDFS.subClassOf, EX.DatasetClass))
    g.add((EX.FungalDisease, RDFS.subClassOf, EX.RiceDisease))
    g.add((EX.BacterialDisease, RDFS.subClassOf, EX.RiceDisease))
    g.add((EX.FungalDisease, OWL.disjointWith, EX.BacterialDisease))
    g.add((EX.ImageLevelLabel, RDFS.subClassOf, EX.DatasetClass))
    g.add((EX.RiceDisease, OWL.disjointWith, EX.ImageLevelLabel))
    g.add((EX.VisualConcept, OWL.disjointWith, EX.DatasetClass))
    g.add((EX.Evidence, OWL.disjointWith, EX.VisualConcept))

    # ---- Thuộc tính đối tượng ----
    obj_props = {
        "describedForDisease": ("VisualConcept", "RiceDisease", "Sách mô tả concept ở bệnh này (không độc quyền, không phải điều kiện cần/đủ)"),
        "hasEvidence": ("VisualConcept", "Evidence", "Concept có căn cứ từ bằng chứng"),
        "evidenceFrom": ("Evidence", "SourceBook", "Bằng chứng trích từ sách"),
        "evidenceAbout": ("Evidence", None, "Bằng chứng nói về bệnh"),
        "assignedPer": ("VisualConcept", "ObservationUnit", "Đơn vị gán nhãn của concept"),
        "combinationFor": ("CandidateCombination", "RiceDisease", "Tổ hợp phù hợp mô tả bệnh"),
        "requiresConcept": ("CandidateCombination", "VisualConcept", "Concept bắt buộc trong tổ hợp"),
        "optionalConcept": ("CandidateCombination", "VisualConcept", "Concept bổ sung, thiếu không loại tổ hợp"),
        "combinationEvidence": ("CandidateCombination", "Evidence", "Bằng chứng cho tổ hợp"),
        "combinationScope": ("CandidateCombination", "ObservationUnit", "Phạm vi đồng xuất hiện yêu cầu"),
        "imageLabelConcept": ("ImageLevelLabel", "VisualConcept", "Concept biểu diễn nhãn cấp ảnh"),
    }
    for name, (dom, rng, label) in obj_props.items():
        p = EX[name]
        g.add((p, RDF.type, OWL.ObjectProperty)); g.add((p, RDFS.label, vi(label)))
        if dom: g.add((p, RDFS.domain, EX[dom]))
        if rng: g.add((p, RDFS.range, EX[rng]))

    # ---- Thuộc tính dữ liệu ----
    data_props = {
        "code": XSD.string, "conceptStatus": XSD.string, "basis": XSD.string, "sourcePages": XSD.string,
        "rule1": XSD.string, "rule0": XSD.string, "rule9": XSD.string, "migrationFromV17": XSD.string,
        "usedInBottleneck": XSD.boolean, "bottleneckStatus": XSD.string, "bottleneckTodo": XSD.string,
        "trainPositive": XSD.integer, "trainNegative": XSD.integer, "trainUnknown": XSD.integer,
        "observableOnLeafImages": XSD.boolean,
        "printedPage": XSD.integer, "pdfPage": XSD.integer, "bookSection": XSD.string,
        "quote": XSD.string, "meaningUsed": XSD.string, "statementType": XSD.string, "limitation": XSD.string,
        "interpretation": XSD.string, "combinationStatus": XSD.string, "evaluableFromModel": XSD.boolean,
        "datasetCode": XSD.string, "imageFolder": XSD.string, "bibliography": XSD.string, "url": XSD.anyURI,
        "pathogenAsInSource": XSD.string,
    }
    for name, rng in data_props.items():
        g.add((EX[name], RDF.type, OWL.DatatypeProperty)); g.add((EX[name], RDFS.range, rng))

    # ---- Sách nguồn ----
    s01 = EX.S01
    g.add((s01, RDF.type, OWL.NamedIndividual)); g.add((s01, RDF.type, EX.SourceBook))
    g.add((s01, EX.code, Literal("S01")))
    g.add((s01, RDFS.label, vi("Giáo trình Cây lúa")))
    g.add((s01, EX.bibliography, vi("Nguyễn Ngọc Đệ (2008). Giáo trình Cây lúa. Trường Đại học Cần Thơ, "
                                     "Viện Nghiên cứu Phát triển Đồng bằng sông Cửu Long. Năm chốt theo bìa; chân trang ghi 2007.")))
    g.add((s01, EX.url, Literal("https://drive.google.com/file/d/1nZLpPHPk5jtoE9Gg5qb0zI1iwthE7dx0/view", datatype=XSD.anyURI)))

    # ---- Đơn vị gán ----
    for label, iri in UNIT_IRI.items():
        u = EX[iri]
        g.add((u, RDF.type, OWL.NamedIndividual)); g.add((u, RDF.type, EX.ObservationUnit)); g.add((u, RDFS.label, vi(label)))

    # ---- Bệnh và lớp dữ liệu ----
    disease_iri = {}
    pathogens = {"Đạo ôn": "E05", "Đốm nâu": "E11", "Khô vằn": "E14", "Bạc lá": "E26"}
    ev_by_code = {r["Mã bằng chứng"]: r for _, r in evidence.iterrows()}
    for name, (dcode, folder, iri) in DISEASES.items():
        d = EX[iri]; disease_iri[name] = d
        g.add((d, RDF.type, OWL.NamedIndividual)); g.add((d, RDF.type, EX.RiceDisease))
        g.add((d, RDFS.label, vi(name))); g.add((d, SKOS.altLabel, Literal(folder, lang="en")))
        g.add((d, EX.datasetCode, Literal(dcode))); g.add((d, EX.imageFolder, Literal(folder)))
        g.add((d, EX.pathogenAsInSource, Literal(str(ev_by_code[pathogens[name]]["Trích đoạn giáo trình"]).strip())))
        g.add((d, RDF.type, EX.BacterialDisease if name == "Bạc lá" else EX.FungalDisease))
    hcode, hfolder, hiri = HEALTHY
    h = EX[hiri]
    g.add((h, RDF.type, OWL.NamedIndividual)); g.add((h, RDF.type, EX.ImageLevelLabel))
    g.add((h, RDFS.label, vi("Lá khỏe (không thấy tổn thương rõ)"))); g.add((h, SKOS.altLabel, Literal(hfolder, lang="en")))
    g.add((h, EX.datasetCode, Literal(hcode))); g.add((h, EX.imageFolder, Literal(hfolder)))
    g.add((h, RDFS.comment, vi("Quy ước dự án: lớp ảnh, không phải bệnh và không kết luận cây khỏe sinh học. Gắn với C16.")))
    g.add((h, EX.imageLabelConcept, EX.C16))
    oos = EX.D_ThoiBe
    g.add((oos, RDF.type, OWL.NamedIndividual)); g.add((oos, RDF.type, EX.OutOfScopeDisease))
    g.add((oos, RDFS.label, vi("Thối bẹ (Sheath rot)")))
    g.add((oos, RDFS.comment, vi("Bối cảnh ngoài 4 bệnh (E27): tổn thương bẹ không độc quyền khô vằn. Không phải lớp dữ liệu.")))
    disease_topic = {"Bối cảnh ngoài 4 bệnh": oos, **disease_iri}

    # ---- Bằng chứng ----
    for _, r in evidence.iterrows():
        code = str(r["Mã bằng chứng"]).strip()
        e = EX[code]
        g.add((e, RDF.type, OWL.NamedIndividual)); g.add((e, RDF.type, EX.Evidence))
        g.add((e, EX.code, Literal(code))); g.add((e, RDFS.label, vi(f"{code} – {r['Chủ đề bệnh']}")))
        g.add((e, EX.evidenceFrom, s01)); g.add((e, EX.evidenceAbout, disease_topic[str(r["Chủ đề bệnh"]).strip()]))
        g.add((e, EX.printedPage, Literal(int(r["Trang in"])))); g.add((e, EX.pdfPage, Literal(int(r["Trang PDF tham chiếu"]))))
        g.add((e, EX.bookSection, Literal(str(r["Mục sách"])))); g.add((e, EX.quote, vi(r["Trích đoạn giáo trình"])))
        for col, prop in [("Ý nghĩa được dùng", EX.meaningUsed), ("Loại phát biểu", EX.statementType), ("Giới hạn", EX.limitation)]:
            if clean(r.get(col)): g.add((e, prop, vi(r[col])))

    # ---- Concept ----
    bn = {r["Mã"]: r for _, r in bottleneck.iterrows()}
    concept_rows = []
    for _, r in concepts.iterrows():
        code = r["Mã"]; c = EX[code]
        g.add((c, RDF.type, OWL.NamedIndividual)); g.add((c, RDF.type, EX.VisualConcept))
        g.add((c, EX.code, Literal(code))); g.add((c, RDFS.label, vi(r["Tên concept v1.8"])))
        g.add((c, SKOS.definition, vi(r["Định nghĩa"])))
        g.add((c, EX.conceptStatus, Literal(r["Trạng thái"]))); g.add((c, EX.basis, vi(r["Căn cứ"])))
        g.add((c, EX.sourcePages, vi(r["Nguồn / trang in"])))
        g.add((c, EX.rule1, vi(r["Nhãn 1"]))); g.add((c, EX.rule0, vi(r["Nhãn 0"]))); g.add((c, EX.rule9, vi(r["Nhãn 9"])))
        g.add((c, EX.migrationFromV17, vi(r["Chuyển nhãn từ v1.7"])))
        g.add((c, EX.assignedPer, EX[UNIT_IRI[str(r["Đơn vị gán"]).strip()]]))
        diseases = [d.strip() for d in str(r["Bệnh có căn cứ trong sách"]).split(";")] if clean(r["Bệnh có căn cứ trong sách"]) else []
        for d in diseases: g.add((c, EX.describedForDisease, disease_iri[d]))
        for e in CONCEPT_EVIDENCE[code]: g.add((c, EX.hasEvidence, EX[e]))
        n = counts[code]
        used = n["n1"] >= MIN_POS and n["n0"] >= MIN_NEG
        sheet_status = str(bn[code]["Trạng thái"]).strip()
        assert used == (sheet_status == "Đưa vào bottleneck"), f"{code}: ngưỡng tính lại khác sheet Bottleneck ({sheet_status})"
        g.add((c, EX.usedInBottleneck, Literal(used))); g.add((c, EX.bottleneckStatus, vi(sheet_status)))
        if clean(bn[code].get("Việc cần làm")): g.add((c, EX.bottleneckTodo, vi(bn[code]["Việc cần làm"])))
        g.add((c, EX.trainPositive, Literal(n["n1"]))); g.add((c, EX.trainNegative, Literal(n["n0"]))); g.add((c, EX.trainUnknown, Literal(n["n9"])))
        g.add((c, EX.observableOnLeafImages, Literal(code != "C13")))
        if code == "C16":
            g.add((c, RDFS.comment, vi("Ràng buộc gán nhãn: C16 = 1 kéo theo C01–C12, C14, C15 = 0 (không để 9); C13 theo quan sát bẹ.")))
        if code == "C10":
            g.add((c, RDFS.comment, vi("V1.8 gộp C10 (xám trắng – bạc lá) và C17 (vết màu xám – khô vằn) của V1.7. Không gồm phần giữa xám tro hình mắt én (C02).")))
        if code == "C13":
            g.add((c, RDFS.comment, vi("Dữ liệu hiện tại chủ yếu là ảnh phiến lá: toàn bộ nhãn train = 9. Giữ trong ontology như tri thức sách.")))
        concept_rows.append(dict(code=code, name=r["Tên concept v1.8"], diseases="; ".join(diseases),
                                 evidence="; ".join(CONCEPT_EVIDENCE[code]), unit=r["Đơn vị gán"], status=r["Trạng thái"],
                                 used_in_bottleneck=used, train_n1=n["n1"], train_n0=n["n0"], train_n9=n["n9"]))

    # ---- Tổ hợp ----
    used_codes = {row["code"] for row in concept_rows if row["used_in_bottleneck"]}
    combo_rows = []
    for _, r in combos.iterrows():
        code = r["Mã"]; p = EX[code]
        required = [x.strip() for x in str(r["Concept cùng đối tượng"]).split("+")]
        g.add((p, RDF.type, OWL.NamedIndividual)); g.add((p, RDF.type, EX.CandidateCombination))
        g.add((p, EX.code, Literal(code))); g.add((p, RDFS.label, vi(r["Tổ hợp nghiên cứu"])))
        g.add((p, EX.combinationFor, disease_iri[str(r["Bệnh ứng viên"]).strip()]))
        for c in required: g.add((p, EX.requiresConcept, EX[c]))
        for c in COMBO_OPTIONAL[code]: g.add((p, EX.optionalConcept, EX[c]))
        for e in COMBO_EVIDENCE[code]: g.add((p, EX.combinationEvidence, EX[e]))
        scope = str(r["Phạm vi"]).strip()
        scope_iri = {"Cùng một vết": "U_TungVet", "Cùng vùng trên cùng lá": "U_CungVungTrenLa", "Cùng một vết trên bẹ": "U_BeLa"}[scope]
        g.add((p, EX.combinationScope, EX[scope_iri]))
        g.add((p, EX.printedPage, Literal(int(r["Trang in"]))))
        g.add((p, EX.interpretation, vi(r["Diễn giải và giới hạn"])))
        g.add((p, EX.combinationStatus, Literal("RESEARCH_QUERY_NOT_DIAGNOSTIC")))
        missing = [c for c in required if c not in used_codes]
        g.add((p, EX.evaluableFromModel, Literal(not missing)))
        if missing:
            g.add((p, EX.limitation, vi("Mô hình CBM hiện không dự đoán " + ", ".join(missing) +
                                       " (chưa vào bottleneck) nên chỉ đánh giá được một phần tổ hợp.")))
        combo_rows.append(dict(code=code, disease=str(r["Bệnh ứng viên"]).strip(), required=required,
                               optional=COMBO_OPTIONAL[code], scope=scope, page=int(r["Trang in"]),
                               evidence=COMBO_EVIDENCE[code], missing_from_bottleneck=missing,
                               evaluable_from_model=not missing, interpretation=str(r["Diễn giải và giới hạn"]).strip()))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    g.serialize(out / "Ontology_benh_lua_V18.ttl", format="turtle", encoding="utf-8")
    g.serialize(out / "Ontology_benh_lua_V18.owl", format="pretty-xml", encoding="utf-8")

    pd.DataFrame(concept_rows).to_csv(out / "Concept_V18.csv", index=False, encoding="utf-8-sig")
    evidence.drop(columns=[c for c in evidence.columns if str(c).startswith("Unnamed")]).to_csv(
        out / "Bang_chung_V18.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{**r, "required": " + ".join(r["required"]), "optional": "; ".join(r["optional"]),
                   "evidence": "; ".join(r["evidence"]), "missing_from_bottleneck": "; ".join(r["missing_from_bottleneck"])}
                  for r in combo_rows]).to_csv(out / "To_hop_V18.csv", index=False, encoding="utf-8-sig")

    mapping = {
        "version": VERSION, "concept_version": "1.8", "ontology_iri": ONTO_IRI,
        "label_values": {"1": "quan sát có", "0": "đủ rõ và không có", "9": "không xác định – bị mask khi học, không coi là 0"},
        "classes": {**{v[0]: {"name_vi": k, "folder": v[1], "iri": ONTO_IRI + "#" + v[2], "type": "RiceDisease"} for k, v in DISEASES.items()},
                    hcode: {"name_vi": "Lá khỏe", "folder": hfolder, "iri": ONTO_IRI + "#" + hiri, "type": "ImageLevelLabel"}},
        "concepts": {row["code"]: {"name_vi": row["name"], "iri": ONTO_IRI + "#" + row["code"],
                                   "described_for": row["diseases"].split("; ") if row["diseases"] else [],
                                   "evidence": CONCEPT_EVIDENCE[row["code"]], "used_in_bottleneck": row["used_in_bottleneck"],
                                   "train_counts": {"1": row["train_n1"], "0": row["train_n0"], "9": row["train_n9"]}}
                     for row in concept_rows},
        "bottleneck_concepts": [row["code"] for row in concept_rows if row["used_in_bottleneck"]],
        "bottleneck_rule": f">= {MIN_POS} ảnh nhãn 1 và >= {MIN_NEG} ảnh nhãn 0 trên train; sau train loại khỏi giải thích nếu AUC val < 0.75",
        "label_constraints": [{"if": {"C16": 1}, "then_zero": LEAF_DISEASE_CONCEPTS}],
        "combinations": combo_rows,
        "combination_note": "Nhãn và dự đoán ở cấp ảnh: chỉ báo 'phù hợp ở cấp ảnh', không khẳng định cùng vết. Thiếu tổ hợp không loại trừ bệnh.",
    }
    (out / "cbm_mapping_V18.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return g, mapping


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    g, m = build(*sys.argv[1:])
    print("Triples:", len(g), "| bottleneck:", ", ".join(m["bottleneck_concepts"]))
