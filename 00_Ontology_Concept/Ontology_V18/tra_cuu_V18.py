"""Tra cứu nhanh ontology V1.8 (chỉ cần Python chuẩn).

    python tra_cuu_V18.py C10      # concept
    python tra_cuu_V18.py P03      # tổ hợp
    python tra_cuu_V18.py E16      # bằng chứng
    python tra_cuu_V18.py "Khô vằn"
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rdfmini import RDF, RDFS, SKOS, Graph, Namespace  # noqa: E402

EX = Namespace("http://example.org/rice-cbm/ontology/v18#")
g = Graph.parse_rdfxml(HERE / "Ontology_benh_lua_V18.owl")


def val(s, p):
    return ", ".join(sorted(str(o) for o in g.objects(s, p)))


def name(s):
    code = val(s, EX.code)
    return f"{code} – {val(s, RDFS.label)}" if code else val(s, RDFS.label)


def show_evidence(e, indent="  "):
    print(f"{indent}{val(e, EX.code)} (tr. {val(e, EX.printedPage)}, mục {val(e, EX.bookSection)}): “{val(e, EX.quote)}”")
    if g.objects(e, EX.limitation):
        print(f"{indent}  Giới hạn: {val(e, EX.limitation)}")


def main(q):
    s = EX[q.upper()] if q.upper()[:1] in "CPE" and q[1:].isdigit() else None
    if s is None:
        hits = [x for x in g.subjects(RDF.type, EX.RiceDisease) if val(x, RDFS.label).lower() == q.lower()]
        if not hits:
            raise SystemExit(f"Không tìm thấy: {q}")
        d = hits[0]
        print(f"Bệnh: {val(d, RDFS.label)} | tác nhân theo sách: {val(d, EX.pathogenAsInSource)}")
        print("Concept được sách mô tả:")
        for c in sorted(g.subjects(EX.describedForDisease, d)):
            print(f"  {name(c)}  [bottleneck: {val(c, EX.usedInBottleneck)}]")
        for p in sorted(g.subjects(EX.combinationFor, d)):
            print(f"Tổ hợp {name(p)}: " + " + ".join(sorted(val(c, EX.code) for c in g.objects(p, EX.requiresConcept))))
        return
    types = g.objects(s, RDF.type)
    if EX.VisualConcept in types:
        print(name(s)); print("Định nghĩa:", val(s, SKOS.definition))
        print("Bệnh có căn cứ:", ", ".join(val(d, RDFS.label) for d in g.objects(s, EX.describedForDisease)) or "— (quy ước dự án)")
        print("Đơn vị gán:", ", ".join(val(u, RDFS.label) for u in g.objects(s, EX.assignedPer)))
        print(f"Train 1/0/9: {val(s, EX.trainPositive)}/{val(s, EX.trainNegative)}/{val(s, EX.trainUnknown)} | bottleneck: {val(s, EX.usedInBottleneck)}")
        if g.objects(s, RDFS.comment): print("Ghi chú:", val(s, RDFS.comment))
        print("Bằng chứng:")
        for e in sorted(g.objects(s, EX.hasEvidence)): show_evidence(e)
    elif EX.CandidateCombination in types:
        print(name(s), "→", ", ".join(val(d, RDFS.label) for d in g.objects(s, EX.combinationFor)))
        print("Bắt buộc:", " + ".join(sorted(val(c, EX.code) for c in g.objects(s, EX.requiresConcept))),
              "| bổ sung:", ", ".join(sorted(val(c, EX.code) for c in g.objects(s, EX.optionalConcept))) or "—")
        print("Phạm vi:", ", ".join(val(u, RDFS.label) for u in g.objects(s, EX.combinationScope)),
              "| đánh giá được từ mô hình:", val(s, EX.evaluableFromModel))
        print("Diễn giải:", val(s, EX.interpretation))
        if g.objects(s, EX.limitation): print("Giới hạn:", val(s, EX.limitation))
        for e in sorted(g.objects(s, EX.combinationEvidence)): show_evidence(e)
    elif EX.Evidence in types:
        show_evidence(s, ""); print("Ý nghĩa dùng:", val(s, EX.meaningUsed))
    else:
        raise SystemExit(f"Không tìm thấy: {q}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
