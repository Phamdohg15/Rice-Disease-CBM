"""Bộ RDF tối giản chỉ dùng thư viện chuẩn Python.

Đủ để dựng ontology, ghi Turtle và RDF/XML (mở được bằng Protégé), và đọc lại
RDF/XML do chính module này ghi để kiểm tra. Không thay thế rdflib cho dữ liệu
RDF tùy ý. Nếu máy có rdflib, kiem_tra_ontology_V18.py sẽ dùng rdflib để kiểm tra
độc lập thêm một lần.
"""
import xml.etree.ElementTree as ET
from pathlib import Path


class URIRef(str):
    pass


class Literal:
    __slots__ = ("value", "lang", "datatype")

    def __init__(self, value, lang=None, datatype=None):
        if isinstance(value, bool):
            datatype = datatype or XSD.boolean
            value = "true" if value else "false"
        elif isinstance(value, int):
            datatype = datatype or XSD.integer
            value = str(value)
        self.value = str(value)
        self.lang = lang
        self.datatype = URIRef(datatype) if datatype else None
        if lang and datatype:
            raise ValueError("Literal không được có cả lang và datatype")

    def _key(self):
        return (self.value, self.lang, self.datatype)

    def __eq__(self, other):
        return isinstance(other, Literal) and self._key() == other._key()

    def __hash__(self):
        return hash(("L",) + self._key())

    def __repr__(self):
        return f"Literal({self.value!r}, lang={self.lang!r}, datatype={self.datatype!r})"

    def __str__(self):
        return self.value


class Namespace(str):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return URIRef(self + name)

    def __getitem__(self, name):
        return URIRef(self + name)


RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
STANDARD_PREFIXES = {"rdf": RDF, "rdfs": RDFS, "owl": OWL, "xsd": XSD, "skos": SKOS, "dcterms": DCTERMS}


def _esc_turtle(s):
    return (s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))


class Graph:
    def __init__(self):
        self.triples = set()
        self.prefixes = dict(STANDARD_PREFIXES)

    def bind(self, prefix, ns):
        self.prefixes[prefix] = Namespace(ns)

    def add(self, triple):
        s, p, o = triple
        if not isinstance(s, URIRef) or not isinstance(p, URIRef):
            raise TypeError(f"Chủ thể/vị từ phải là URIRef: {triple}")
        if not isinstance(o, (URIRef, Literal)):
            raise TypeError(f"Đối tượng phải là URIRef hoặc Literal: {triple}")
        self.triples.add((URIRef(s), URIRef(p), o))

    def __len__(self):
        return len(self.triples)

    def __iter__(self):
        return iter(self.triples)

    def objects(self, s=None, p=None):
        return [o for (ss, pp, o) in self.triples if (s is None or ss == s) and (p is None or pp == p)]

    def subjects(self, p=None, o=None):
        return [s for (s, pp, oo) in self.triples if (p is None or pp == p) and (o is None or oo == o)]

    # ---------- ghi ----------
    def _qname(self, uri):
        best = None
        for pfx, ns in self.prefixes.items():
            if uri.startswith(ns):
                local = uri[len(ns):]
                if local and all(ch.isalnum() or ch in "_-" for ch in local) and not local[0].isdigit():
                    if best is None or len(ns) > len(self.prefixes[best[0]]):
                        best = (pfx, local)
        return f"{best[0]}:{best[1]}" if best else f"<{uri}>"

    def _term_turtle(self, t):
        if isinstance(t, URIRef):
            return self._qname(t)
        s = f'"{_esc_turtle(t.value)}"'
        if t.lang:
            return s + "@" + t.lang
        if t.datatype:
            return s + "^^" + self._qname(t.datatype)
        return s

    def _sorted_subjects(self):
        return sorted({s for s, _, _ in self.triples})

    def to_turtle(self):
        used = set()
        body = []
        for s in self._sorted_subjects():
            preds = sorted({p for ss, p, _ in self.triples if ss == s}, key=lambda p: (p != RDF.type, p))
            lines = []
            for p in preds:
                objs = sorted((o for ss, pp, o in self.triples if ss == s and pp == p), key=lambda o: (isinstance(o, Literal), str(o)))
                ptxt = "a" if p == RDF.type else self._qname(p)
                lines.append(f"    {ptxt} " + " ,\n        ".join(self._term_turtle(o) for o in objs))
            body.append(self._qname(s) + "\n" + " ;\n".join(lines) + " .\n")
        text = "\n".join(body)
        for pfx, ns in self.prefixes.items():
            if (pfx + ":") in text:
                used.add(pfx)
        head = "".join(f"@prefix {p}: <{self.prefixes[p]}> .\n" for p in sorted(used))
        return head + "\n" + text

    def to_rdfxml(self):
        nsmap = {p: ns for p, ns in self.prefixes.items()}
        out = ['<?xml version="1.0" encoding="utf-8"?>']
        attrs = " ".join(f'xmlns:{p}="{ns}"' for p, ns in sorted(nsmap.items()))
        out.append(f"<rdf:RDF {attrs}>")

        def esc(v):
            return v.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

        def pred_tag(p):
            for pfx, ns in sorted(nsmap.items(), key=lambda kv: -len(kv[1])):
                if p.startswith(ns):
                    local = p[len(ns):]
                    if local and (local[0].isalpha() or local[0] == "_") and all(ch.isalnum() or ch in "_-." for ch in local):
                        return f"{pfx}:{local}"
            raise ValueError(f"Không tạo được tên thẻ XML cho vị từ {p}; cần bind namespace")

        for s in self._sorted_subjects():
            out.append(f'  <rdf:Description rdf:about="{esc(s)}">')
            for (ss, p, o) in sorted((t for t in self.triples if t[0] == s), key=lambda t: (t[1], str(t[2]))):
                tag = pred_tag(p)
                if isinstance(o, URIRef):
                    out.append(f'    <{tag} rdf:resource="{esc(o)}"/>')
                elif o.lang:
                    out.append(f'    <{tag} xml:lang="{o.lang}">{esc(o.value)}</{tag}>')
                elif o.datatype:
                    out.append(f'    <{tag} rdf:datatype="{esc(o.datatype)}">{esc(o.value)}</{tag}>')
                else:
                    out.append(f"    <{tag}>{esc(o.value)}</{tag}>")
            out.append("  </rdf:Description>")
        out.append("</rdf:RDF>")
        return "\n".join(out) + "\n"

    def serialize(self, path, format="turtle", encoding="utf-8"):
        text = self.to_turtle() if format == "turtle" else self.to_rdfxml()
        Path(path).write_text(text, encoding=encoding)

    # ---------- đọc lại RDF/XML (định dạng do module này ghi) ----------
    @classmethod
    def parse_rdfxml(cls, path):
        g = cls()
        R = "{" + RDF + "}"
        XMLLANG = "{http://www.w3.org/XML/1998/namespace}lang"
        root = ET.parse(path).getroot()
        if root.tag != R + "RDF":
            raise ValueError("Không phải rdf:RDF")
        for desc in root:
            if desc.tag != R + "Description":
                raise ValueError(f"Phần tử không hỗ trợ: {desc.tag}")
            s = URIRef(desc.attrib[R + "about"])
            for child in desc:
                ns, local = child.tag[1:].split("}")
                p = URIRef(ns + local)
                if R + "resource" in child.attrib:
                    g.add((s, p, URIRef(child.attrib[R + "resource"])))
                else:
                    g.add((s, p, Literal(child.text or "", lang=child.attrib.get(XMLLANG), datatype=child.attrib.get(R + "datatype"))))
        return g
