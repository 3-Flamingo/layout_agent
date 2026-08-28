import json
from collections import Counter
from pathlib import Path

sid = "1511437367716814848"
doc = json.loads(Path(f"data/raw/solution_{sid}/solution_doc.json").read_text(encoding="utf-8"))
by = {}
for el in doc.get("doc") or []:
    by.setdefault(el.get("_ctor_") or "?", []).append(el)

def eid(obj):
    if obj is None: return None
    if isinstance(obj, dict) and "id" in obj: return int(obj["id"])
    if isinstance(obj, (int, float)): return int(obj)
    return None

needle = "TEF648102101"
node = next(n for n in by["CadTreeNode"] if n.get("name") == needle or n.get("pureName") == needle)
nid = eid(node.get("id"))
print("treeNodeId", nid)

# ViewGroups referencing this tree node
vg_hits = []
for g in by["CadViewGroup"]:
    tids = g.get("dataSourceTreeNodeIds") or []
    tid = eid(g.get("treeNodeId"))
    flat = []
    for t in tids:
        flat.append(eid(t) if not isinstance(t, list) else t)
    # sometimes list of ints
    raw = tids
    if nid == tid or nid in (eid(x) for x in (tids if isinstance(tids, list) else [])):
        vg_hits.append(g)
    elif isinstance(tids, list) and nid in [eid(x) for x in tids]:
        vg_hits.append(g)
print("viewgroups hit", len(vg_hits))
for g in vg_hits:
    print(" vg id", eid(g.get("id")), "pid(drawing?)", eid(g.get("pid")), "treeNodeId", eid(g.get("treeNodeId")), "dataSource", g.get("dataSourceTreeNodeIds"))

# Also: CadView.groupId might equal CadViewGroup somehow? check view groupId uniqueness
print("sample view groupId", by["CadView"][0].get("groupId"))
# CadViewDesc?
descs = by.get("CadViewDesc", [])
print("CadViewDesc", len(descs))
if descs:
    print("desc keys", sorted(descs[0].keys()))
    dhit = [d for d in descs if needle in json.dumps(d, ensure_ascii=False) or str(nid) in json.dumps(d)]
    print("desc hits", len(dhit))

# Match ViewGroup.pid == CadView.pid (same drawing), filter views on that drawing where...?
# Maybe all views on drawings owned by part
# CadDrawing id linked via ViewGroup.pid
if not vg_hits:
    # try string compare in whole group
    for g in by["CadViewGroup"]:
        if str(nid) in json.dumps(g):
            vg_hits.append(g)
    print("viewgroups by str id", len(vg_hits))
    for g in vg_hits[:5]:
        print(json.dumps({k: g.get(k) for k in ("id","pid","treeNodeId","dataSourceTreeNodeIds","isBlanking")}, ensure_ascii=False))

# How many tree nodes have type part
parts = [n for n in by["CadTreeNode"] if n.get("type") == "part"]
print("part nodes", len(parts))
# For each part, find viewgroups
from collections import defaultdict
part_to_drawings = defaultdict(set)
for g in by["CadViewGroup"]:
    tid = eid(g.get("treeNodeId"))
    ds = g.get("dataSourceTreeNodeIds") or []
    ids = set()
    if tid is not None:
        ids.add(tid)
    for x in ds:
        e = eid(x)
        if e is not None:
            ids.add(e)
    for i in ids:
        part_to_drawings[i].add(eid(g.get("pid")))

print("drawings for our part", part_to_drawings.get(nid))
# maybe treeNodeId points to assembly not part - check parent chain
print("node pid", node.get("pid"), "level", node.get("level"), "type", node.get("type"))
