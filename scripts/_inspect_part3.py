import json
from collections import Counter
from pathlib import Path

sid = "1511437367716814848"
doc = json.loads(Path(f"data/raw/solution_{sid}/solution_doc.json").read_text(encoding="utf-8"))
by = {}
for el in doc.get("doc") or []:
    by.setdefault(el.get("_ctor_") or "?", []).append(el)

needle = "TEF648102101"
node = next(n for n in by["CadTreeNode"] if n.get("name") == needle or n.get("pureName") == needle)
gid = node.get("groupId")
print("node groupId", gid, "id", node.get("id"), "frameId", node.get("frameId"))

views = [v for v in by["CadView"] if v.get("groupId") == gid]
print("views by groupId", len(views), Counter(v.get("viewType") for v in views))

# CadViewGroup?
groups = by.get("CadViewGroup", [])
print("CadViewGroup", len(groups))
if groups:
    print("group keys", sorted(groups[0].keys()))
    ghit = [g for g in groups if g.get("groupId") == gid or json.dumps(g, ensure_ascii=False).find(needle) >= 0]
    print("groups hit", len(ghit))
    if ghit:
        print(json.dumps(ghit[0], ensure_ascii=False, indent=2)[:1500])

# drawings with partDetails containing name
matched_draw = []
for d in by["CadDrawing"]:
    pd = d.get("partDetails") or {}
    blob = json.dumps(pd, ensure_ascii=False)
    if needle in blob or gid in blob:
        matched_draw.append(d)
print("drawings matched", len(matched_draw))
for d in matched_draw[:3]:
    print(" drawing id", d.get("id"), "partDetails keys", list((d.get("partDetails") or {}).keys())[:20])
    print(json.dumps(d.get("partDetails"), ensure_ascii=False)[:500])

# map drawing id -> part name via tree node frameId / ref?
# views pid -> drawing
if views:
    pids = Counter()
    for v in views:
        pid = v.get("pid")
        if isinstance(pid, dict):
            pid = pid.get("id")
        pids[pid] += 1
    print("view pids", pids)
    for pid in pids:
        d = next((x for x in by["CadDrawing"] if (x.get("id") or {}).get("id") == pid or x.get("id") == pid), None)
        print(" drawing for pid", pid, "found", bool(d), "partDetails", (d or {}).get("partDetails"))
