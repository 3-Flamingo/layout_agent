import json
from pathlib import Path

sid = "1511437367716814848"
doc = json.loads(Path(f"data/raw/solution_{sid}/solution_doc.json").read_text(encoding="utf-8"))
by = {}
for el in doc.get("doc") or []:
    by.setdefault(el.get("_ctor_") or "?", []).append(el)

needle = "TEF648102101"
for el in by.get("CadTreeNode", []):
    s = json.dumps(el, ensure_ascii=False)
    if needle in s:
        print("TreeNode keys", sorted(el.keys()))
        print(json.dumps(el, ensure_ascii=False, indent=2)[:2000])

drawings = by.get("CadDrawing", [])
print("\nCadDrawing sample partDetails type", type(drawings[0].get("partDetails")))
pd = drawings[0].get("partDetails")
print(json.dumps(pd, ensure_ascii=False, indent=2)[:1500] if pd else None)

# find drawing containing needle
for d in drawings:
    s = json.dumps(d, ensure_ascii=False)
    if needle in s:
        print("\nFOUND drawing id", d.get("id"))
        print(json.dumps(d.get("partDetails"), ensure_ascii=False, indent=2)[:2000])
        did = d["id"]["id"] if isinstance(d.get("id"), dict) else d.get("id")
        views = [v for v in by.get("CadView", []) if (v.get("pid") == did or (isinstance(v.get("pid"), dict) and v["pid"].get("id") == did))]
        # pid might be int
        views2 = []
        for v in by.get("CadView", []):
            pid = v.get("pid")
            if isinstance(pid, dict):
                pid = pid.get("id")
            if pid == did or str(pid) == str(did):
                views2.append(v)
        print("views under drawing", len(views2), "types", sorted({v.get("viewType") for v in views2}))
        for v in views2[:8]:
            print(" ", v.get("viewType"), v.get("uuid"), "id", v.get("id"))
