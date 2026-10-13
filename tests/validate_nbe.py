"""Validate an nbe.json against the schema the hero player (public/js/nbe.js) reads.

    python tests/validate_nbe.py output/nbe.json

Exit code 0 = valid (line-length warnings may still be printed), 1 = errors.
"""
from __future__ import annotations

import json
import sys

KINDS = {"cmd", "out", "dim", "ok", "warn", "refused", "hi", "blank"}
EVENTS = {"add": {"ids"}, "cluster": {"id"}, "bridge": {"id"}, "pulse": {"id"}, "curve": {"id"},
          "refuse": {"id"}, "balance": {"id", "ok"}, "lineage": {"id"}, "focus": {"id"}}
SOFT_MAX, HARD_MAX = 56, 60
MAX_SESSION_LINES = 35


def validate(doc):
    """-> (errors, warnings)"""
    E, Wn = [], []
    for k in ("generated", "date", "graph", "sessions"):
        if k not in doc:
            E.append(f"missing top-level '{k}'")
    if E:
        return E, Wn
    g = doc["graph"]
    for k in ("nodes", "links", "clusters"):
        if not isinstance(g.get(k), list):
            E.append(f"graph.{k} must be a list")
    if E:
        return E, Wn
    ids, cids = set(), {c.get("id") for c in g["clusters"]}
    for c in g["clusters"]:
        if not isinstance(c.get("id"), str):
            E.append(f"cluster without string id: {c}")
        if "label" in c and not isinstance(c["label"], str):
            E.append(f"cluster {c.get('id')}: label must be a string")
    for n in g["nodes"]:
        nid = n.get("id")
        if not isinstance(nid, str) or not nid:
            E.append(f"node without id: {n}")
            continue
        if nid in ids:
            E.append(f"duplicate node id {nid}")
        ids.add(nid)
        if not isinstance(n.get("label"), str) or not n["label"]:
            E.append(f"node {nid}: label missing")
        elif len(n["label"]) > 22:
            Wn.append(f"node {nid}: label longer than 22 chars ({n['label']!r})")
        if not isinstance(n.get("views"), (int, float)) or n["views"] < 0:
            E.append(f"node {nid}: views must be a non-negative number")
        if "cluster" in n and n["cluster"] not in cids:
            E.append(f"node {nid}: unknown cluster {n['cluster']}")
        if "curve" in n:
            cv = n["curve"]
            if not isinstance(cv, list) or len(cv) < 2 or not all(isinstance(x, (int, float)) for x in cv):
                E.append(f"node {nid}: curve must be a list of >= 2 numbers")
        if "info" in n and (not isinstance(n["info"], str) or len(n["info"]) > 60):
            E.append(f"node {nid}: info must be one short string (<= 60 chars)")
    for l in g["links"]:
        if l.get("source") not in ids or l.get("target") not in ids:
            E.append(f"link to unknown node: {l}")
        w = l.get("weight")
        if not isinstance(w, (int, float)) or not 0 <= w <= 1:
            E.append(f"link weight must be in [0, 1]: {l}")
    if not doc["sessions"]:
        E.append("no sessions")
    for si, s in enumerate(doc["sessions"]):
        if not isinstance(s.get("title"), str):
            E.append(f"session {si}: title missing")
        lines = s.get("lines")
        if not isinstance(lines, list) or not lines:
            E.append(f"session {si}: no lines")
            continue
        if len(lines) > MAX_SESSION_LINES:
            Wn.append(f"session {si}: {len(lines)} lines > {MAX_SESSION_LINES}")
        for li, l in enumerate(lines):
            where = f"session {si} line {li}"
            if not isinstance(l, list) or len(l) not in (2, 3):
                E.append(f"{where}: must be [kind, text] or [kind, text, event]")
                continue
            kind, text = l[0], l[1]
            if kind not in KINDS:
                E.append(f"{where}: unknown kind {kind!r}")
            if not isinstance(text, str):
                E.append(f"{where}: text must be a string")
                continue
            n = len(text) + (2 if kind == "cmd" else 0)          # cmd lines get a "$ " prompt
            if n > HARD_MAX:
                E.append(f"{where}: {n} chars > {HARD_MAX}: {text!r}")
            elif n > SOFT_MAX:
                Wn.append(f"{where}: {n} chars > {SOFT_MAX}: {text!r}")
            if len(l) == 3:
                ev = l[2]
                t = ev.get("type") if isinstance(ev, dict) else None
                if t not in EVENTS:
                    E.append(f"{where}: unknown event {ev!r}")
                    continue
                missing = EVENTS[t] - set(ev)
                if missing:
                    E.append(f"{where}: event {t} missing {sorted(missing)}")
                if t == "add":
                    bad = [i for i in ev["ids"] if i not in ids]
                    if bad:
                        E.append(f"{where}: add of unknown ids {bad}")
                elif t in ("cluster", "balance", "lineage"):
                    if ev["id"] not in cids:
                        E.append(f"{where}: {t} of unknown cluster {ev['id']}")
                elif ev["id"] not in ids:
                    E.append(f"{where}: {t} of unknown node {ev['id']}")
                if t == "curve":
                    node = next((n for n in g["nodes"] if n.get("id") == ev["id"]), {})
                    if "curve" not in node:
                        E.append(f"{where}: curve event on a node without a curve")
                if t == "balance" and not isinstance(ev["ok"], bool):
                    E.append(f"{where}: balance.ok must be a boolean")
    return E, Wn


def main(path):
    doc = json.loads(open(path, encoding="utf-8").read())
    errors, warnings = validate(doc)
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("ERROR:", e)
    n_lines = sum(len(s.get("lines", [])) for s in doc.get("sessions", []))
    print(f"{path}: {len(doc['graph']['nodes'])} nodes, {len(doc['graph']['links'])} links, "
          f"{len(doc['sessions'])} sessions, {n_lines} lines -> "
          f"{'OK' if not errors else f'{len(errors)} error(s)'}, {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "output/nbe.json"))
