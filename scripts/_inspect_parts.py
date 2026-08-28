import json
from collections import Counter
from pathlib import Path

sid = "1511437367716814848"
doc = json.loads(Path(f"data/raw/solution_{sid}/solution_doc.json").read_text(encoding="utf-8"))
by = {}
for el in doc.get("doc") or []:
    by.setdefault(el.get("_ctor_") or "?", []).append(el)
print("ctors sample", sorted(by.keys())[:40])
print("CadView count", len(by.get("CadView", [])))
# CadView fields
views = by.get("CadView", [])
if views:
    print("view keys", sorted(views[0].keys()))
    for k in ("partNo", "partNumber", "partId", "partName", "name", "title", "label", "drawingNo", "instanceName", "pid", "parent", "sheetId", "modelId"):
        vals = Counter(str(v.get(k)) for v in views if v.get(k) is not None)
        if vals:
            print(k, "n=", sum(vals.values()), "uniq=", len(vals), "sample=", list(vals.most_common(5)))

# search TEF648102101 anywhere
needle = "TEF648102101"
hits = []
for el in doc.get("doc") or []:
    s = json.dumps(el, ensure_ascii=False)
    if needle in s:
        hits.append(el.get("_ctor_"))
print("needle hits ctors", Counter(hits).most_common(20), "total", len(hits))

# also check sheets / drawings
for ctor in ("CadSheet", "CadDrawing", "CadPart", "CadAssembly", "CadInstance", "MistPart", "CadLayout"):
    print(ctor, len(by.get(ctor, [])))
    if by.get(ctor):
        print("  keys", sorted(by[ctor][0].keys())[:30])
