"""Compose the hero-panel narrative (nbe.json) from one run's results.

Every number printed here comes from the run's results dict `R` (computed this
run, or loaded from state/ with its provenance). A section whose stage failed
prints an honest `warn` line instead. Terminal lines are kept <= 46 chars where
possible (the strip is ~45 characters wide); the validator enforces <= 56.
"""
from __future__ import annotations

import datetime as dt
import statistics

from .select import SENSITIVE
from .util import fmt_views, human_day, label, parse_ymd

MAX_LINE = 46
LBL = {}          # article -> display label, disambiguated per run (set by compose)


def make_labels(titles, maxlen=22):
    """label() per title; when two titles collide, keep a short qualifier:
    'East of Eden (2026)' / 'East of Eden (novel)'."""
    base = {t: label(t, 99) for t in titles}
    out = {}
    for t in titles:
        b = base[t]
        if sum(1 for u in titles if base[u] == b) > 1 and "(" in t:
            q = t[t.rfind("(") + 1:].rstrip(")").split("_")[0]
            b = f"{b} ({q})"
        out[t] = b if len(b) <= maxlen else b[: maxlen - 1].rstrip() + "…"
    return out


def _n(a, n=18):
    s = LBL.get(a) or label(a, 99)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _x(m):
    return f"{m:,.0f}×" if m >= 10 else f"{m:.1f}×"


def _num(n):
    return f"{n:,}"


def _iso(day):
    return parse_ymd(day).isoformat()


class Session:
    def __init__(self, title):
        self.title, self.lines = title, []

    def add(self, kind, text="", event=None):
        self.lines.append([kind, text, event] if event else [kind, text])

    def blank(self):
        if self.lines and self.lines[-1][0] != "blank":
            self.lines.append(["blank", ""])

    def doc(self):
        while self.lines and self.lines[-1][0] == "blank":
            self.lines.pop()
        return {"title": self.title, "lines": self.lines}


def _warn_for(R, stage):
    return [w for w in R["warnings"] if w["stage"] == stage]


def _bucket_text(b):
    lo, hi = b
    return f"≥{lo:.0f}×" if hi > 1e8 else f"{lo:.0f}–{hi:.0f}×"


def _bot_reason(gate):
    """Short reason from the series gate's first bot gap: ('automated 52%', '9 Oct')."""
    for g in gate.get("gaps", []):
        if g[0] == "bot" and len(g) >= 3:
            why = g[2].replace("automated agents ", "automated ").replace(" of all traffic", " of views")
            why = why.replace("desktop share ", "desktop ")
            return why, human_day(g[1])
        if g[0] == "missing":
            return "days missing", human_day(g[1])
        if g[0] == "implausible":
            return "implausible values", human_day(g[1])
    return None, None


# ----------------------------------------------------------------------------- graph
def build_graph(R):
    an = R.get("analysis", {})
    gate = R.get("gate", {})
    net = R.get("net") or {}
    cl_of, clusters = {}, []
    for c in net.get("clusters", [])[:6]:
        cid = c["id"]
        clusters.append({"id": cid, "label": label(c["label"] or "", 18) if c.get("label") else ""})
        for a in c["members"]:
            cl_of[a] = cid
    nodes = []
    for a in R["stories"]:
        hist = R.get("hist_tail", {}).get(a, [])
        node = {"id": a, "label": _n(a, 22), "views": int(R["views_today"].get(a, 0))}
        if a in cl_of:
            node["cluster"] = cl_of[a]
        if hist:
            node["curve"] = hist
        info = _info(a, an.get(a), gate.get(a), R)
        if a in cl_of and info:
            c = next(c for c in net["clusters"] if c.get("id") == cl_of[a])
            names = _via_names([t for t in c["ties"] if a in t["pair"]])
            if names:
                parts = info.split(" · ")
                while parts and len(" · ".join(parts) + f" · via {names[0]}") > 60:
                    parts.pop()
                info = " · ".join(parts + [f"via {names[0]}"])
        if info:
            node["info"] = info
        nodes.append(node)
    ids = {n["id"] for n in nodes}
    links = [{"source": a, "target": b, "weight": round(min(1.0, w / 1000), 2)}
             for a, b, w in net.get("links", []) if a in ids and b in ids]
    return {"nodes": nodes, "links": links, "clusters": clusters}


def _info(a, s, g, R):
    if g and g["status"] == "refused":
        why, day = _bot_reason(g)
        return f"refused: {why} ({day})" if why else "refused by the series gate"
    if not s:
        n = R.get("hist_len", {}).get(a)
        return f"new page: {n} days of views, too new to model" if n else None
    cp = s.get("cp")
    if s["fresh"] and cp:
        txt = f"fresh spike · {_x(cp['multiple'])} · {cp['z']:.0f}σ"
        b = s.get("bucket")
        if b and b["half_life_q"][1] is not None:
            txt += f" · halves ~{b['half_life_q'][1]:.1f}d"
        return txt
    if s.get("formula") and s.get("half_life") and s.get("tail_err", 1) <= 0.25:
        core = s["formula"].replace("v(t) ≈ ", "v ≈ ").split(" + ")[0].split(" − ")[0]
        return f"{core} · t½ {s['half_life']:.1f}d"
    if cp:
        return f"{s['shape']} · break {human_day(cp['date'])} ({cp['z']:.0f}σ)"
    return s["shape"]


# ----------------------------------------------------------------------------- session 1
def session_today(R):
    S = Session("wikipedia news today · top stories")
    D = R["date"]
    S.add("cmd", f"nbe run --date {_iso(D)}")
    sk = R.get("skipped", {})
    n_skip = R.get("junk_today", 0) + len(sk.get("perennial", [])) + len(sk.get("evergreen", []))
    S.add("out", f"[fetch] top list {human_day(D)}: {len(R['stories'])} stories", {"type": "add", "ids": R["stories"]})
    parts = [f"{R.get('junk_today', 0)} junk"]
    if sk.get("perennial"):
        parts.append(f"{len(sk['perennial'])} perennial")
    if sk.get("evergreen"):
        parts.append(f"{len(sk['evergreen'])} evergreen")
    S.add("dim", f"[fetch]   skipped {n_skip} non-news pages" if len(parts) == 1
          else f"[fetch]   skipped {n_skip}: " + ", ".join(parts))

    # LFC series gate
    gate = R.get("gate", {})
    if _warn_for(R, "verify") or not gate:
        S.add("warn", "[verify] series gate failed; unverified data")
    else:
        st = [g["status"] for g in gate.values()]
        line = f"[verify] series gate: {st.count('accepted')} ok"
        if st.count("corrected"):
            line += f", {st.count('corrected')} fixed"
        S.add("out", line + f", {st.count('refused')} refused")
        for a, g in gate.items():
            if g["status"] != "refused":
                continue
            why, day = _bot_reason(g)
            S.add("refused", f"[verify] REFUSED {_n(a, 20)}", {"type": "refuse", "id": a})
            if why:
                S.add("dim", f"[verify]   {why} on {day}")
        for a, g in gate.items():
            if g["status"] == "corrected":
                S.add("dim", f"[verify]   {_n(a, 16)}: {g['reason'][:22]}")
                break
    c = R.get("counters")
    if c and c.get("n_laws"):
        k = "ok" if not c["breaks_today"] else "warn"
        tail = "books balance" if not c["breaks_today"] else f"{len(c['breaks_today'])} breaks"
        S.add(k, f"[nrf] {c['n_counters']} counters obey {c['n_laws']} laws: {tail}")
    elif _warn_for(R, "nrf-counters"):
        S.add("warn", "[nrf] traffic-counter check failed")
    S.blank()

    an = R.get("analysis", {})
    if _warn_for(R, "tnn+ppf"):
        S.add("warn", "[tnn] analysis stage failed")
    fresh = sorted((a for a, s in an.items() if s["fresh"] and s.get("cp") and not SENSITIVE.search(a)),
                   key=lambda a: -an[a]["cp"]["z"])
    for i, a in enumerate(fresh[:2]):
        s = an[a]
        cp = s["cp"]
        if i == 0:
            S.add("hi", f"[tnn] {_n(a)}: break {human_day(cp['date'])}, {cp['z']:.0f}σ", {"type": "pulse", "id": a})
            S.add("hi", f"[tnn]   fresh spike: {_x(cp['multiple'])} baseline")
            S.add("dim", f"[tnn]   ~{fmt_views(cp['baseline'])}/day → {fmt_views(s['naive'])}")
            if cp.get("edit_z") is not None and cp["edit_z"] >= 3:
                S.add("dim", f"[tnn]   edits jumped the same day: {cp['edit_z']:.0f}σ")
        else:
            S.add("hi", f"[tnn] {_n(a)}: {cp['z']:.0f}σ, {_x(cp['multiple'])} baseline",
                  {"type": "pulse", "id": a})
    if fresh:
        top = an[fresh[0]]
        b = top.get("bucket")
        pr = R.get("prior", {})
        if b and b["half_life_q"][1] is not None:
            S.add("out", f"[prior] past {_bucket_text(b['bucket'])} spikes halve in ~{b['half_life_q'][1]:.1f} d")
            S.add("dim", f"[prior]   n={_num(b['n'])} of {_num(pr.get('n_events', 0))} spikes, "
                         f"{_num(pr.get('n_articles', 0))} pages")
        pf = top.get("prior_forecast")
        if pf:
            S.add("out", f"[prior] {_n(fresh[0], 14)} tomorrow ~{fmt_views(pf['value'])}")
            S.add("dim", f"[prior]   IQR {fmt_views(pf['lo'])}–{fmt_views(pf['hi'])}; "
                         f"repeat-today {fmt_views(top['naive'])}")
    S.blank()

    # maturing stories with a clean, compact formula
    # clean decay fits that still track the last days (tail error <= 20%, level not above today)
    mat = sorted((a for a, s in an.items() if not s["fresh"] and s.get("formula") and s.get("half_life")
                  and s["shape"] == "spike-and-decay" and s["r2"] >= 0.8 and s.get("tail_err", 1) <= 0.2
                  and (s.get("long_run") or 0) <= 1.1 * s["naive"] and not SENSITIVE.search(a)),
                 key=lambda a: -R["views_today"].get(a, 0))
    for i, a in enumerate(mat[:2]):
        s = an[a]
        S.add("out", f"[ppf] {_n(a, 20)}, peak {human_day(s['x1_date'])}", {"type": "curve", "id": a})
        S.add("hi", f"[ppf] {s['formula']}")
        if i == 0:
            S.add("dim", "[ppf]   t = days since the peak")
        S.add("out", f"[ppf]   half-life {s['half_life']:.1f} d · R² {s['r2']:.2f}")
        if s.get("forecast") is not None:
            S.add("out", f"[score] tomorrow {fmt_views(s['forecast'])} · repeat-today {fmt_views(s['naive'])}")
    if not mat and not _warn_for(R, "tnn+ppf"):
        S.add("dim", "[ppf] no maturing story with a clean fit")

    # honest scorecards: all clean fits AND the guarded subset (guard adopted 13 Oct, post-hoc)
    if _warn_for(R, "score"):
        S.add("warn", "[score] scoring stage failed")
    bt = [b for b in R.get("backtest", []) if b["method"] == "formula"]
    if bt:
        S.add("out", f"[score] backtest: refit to {human_day(_prev(R['date']))}, call {human_day(R['date'])}")
        _score_lines(S, bt, "[score]  ")
    g = R.get("graded") or {}
    fwd = g.get("forward_rows") or []
    if g.get("rows") and g["rows"][0].get("post_hoc"):
        S.add("out", f"[score] replayed {human_day(g['issued'])} run, graded today:")
        _score_lines(S, [r for r in g["rows"] if r["method"] == "formula"], "[score]  ")
        pr = [r for r in g["rows"] if r["method"] == "prior"]
        if pr:
            m = statistics.median(r["ape"] for r in pr)
            nv = statistics.median(r["ape_naive"] for r in pr)
            S.add("out", f"[score]   prior {m:.0f}% · repeat {nv:.0f}% (n={len(pr)})")
    if fwd:
        S.add("out", f"[score] since {human_day(g['adopted'])}, graded forward:")
        _score_lines(S, [r for r in fwd if r["method"] == "formula"], "[score]  ")
    else:
        S.add("dim", f"[score] forward grading starts {human_day(g.get('adopted') or R['date'])}")
    return S


def _score_lines(S, rows, pre):
    """Median abs error of ALL clean-fit forecasts and of the guarded subset, each next to
    repeat-today on the same stories, so the post-hoc guard never hides the unguarded result."""
    if not rows:
        return
    allm = statistics.median(r["ape"] for r in rows)
    alln = statistics.median(r["ape_naive"] for r in rows)
    S.add("out", f"{pre} all fits {allm:.0f}% · repeat {alln:.0f}% (n={len(rows)})")
    gd = [r for r in rows if r.get("guarded")]
    if gd:
        gm = statistics.median(r["ape"] for r in gd)
        gn = statistics.median(r["ape_naive"] for r in gd)
        S.add("out", f"{pre} guarded {gm:.0f}% · repeat {gn:.0f}% (n={len(gd)})")


def _prev(day):
    return (parse_ymd(day) - dt.timedelta(days=1)).strftime("%Y%m%d")


# ----------------------------------------------------------------------------- session 2
MIN_VIA_SCORE = 4.0   # specificity idf*(1 + #leads carrying the link); with 50 articles max idf is
                      # ln(51) = 3.9, so a name must be specific AND sit in at least one lead section


def _via_names(ties, exclude=None):
    """Tie names worth printing: specific enough, not a crime/death term, not the label."""
    cand = {}
    for t in ties:
        for x, sc, _lead in t["after"]:
            cand[x] = max(sc, cand.get(x, 0))
    ok = [x for x, sc in cand.items() if sc >= MIN_VIA_SCORE and not SENSITIVE.search(x)
          and label(x, 99) != exclude]
    ok.sort(key=lambda x: (-round(cand[x], 1), len(label(x, 99)) > 20, x))
    names = []
    for x in ok:
        nm = label(x, 20)
        if nm not in names:
            names.append(nm)
    return names


def _via(c):
    """What ties a cluster (top 2 names), or None when no name is specific enough."""
    lab = c.get("label") if c.get("label_source") == "link" else None
    names = _via_names(c["ties"], lab)
    return " · ".join(names[:2]) if names else None


MOVES = {"co-surge": "attention rose together (co-surge)",
         "co-fade": "attention fell together (co-fade)",
         "substitution": "one rose as another fell (substitution)",
         "solo": "one story moved on its own (solo)"}


def session_connect(R):
    S = Session("wikipedia news today · how stories connect")
    S.add("cmd", f"nbe connect --date {_iso(R['date'])}")
    net = R.get("net")
    if not net:
        S.add("warn", "[holant] link stage failed; no story graph")
        return S
    S.add("out", f"[holant] {net['n_articles']} stories, {net['n_edges']} link ties",
          {"type": "add", "ids": R["stories"]})
    S.add("dim", "[holant]   tie = shared links + direct links")
    p = net["pairing"]
    S.add("out", f"[holant] best pairing: {len(p['pairs'])} pairs, exact")
    if p.get("verified"):
        S.add("ok", "[holant]   matches exhaustive search")
    elif p.get("checked_all"):
        S.add("warn", "[holant]   exhaustive check disagrees")
    S.add("out", f"[nrf] {len(net['clusters'])} story clusters")
    S.blank()
    for c in net["clusters"][:4]:
        cid = c.get("id")
        lab = label(c["label"] or "", 18) if c.get("label") else "untitled"
        S.add("out", f"[nrf] \"{lab}\": {len(c['members'])} stories",
              {"type": "cluster", "id": cid, "label": label(c["label"] or "", 18)})
        via = _via(c)
        if via:
            S.add("dim", "      via " + via)
        for b in c["bridges"][:1]:
            S.add("hi", f"[holant] bridge: {_n(b, 22)}", {"type": "bridge", "id": b})
        ln = c.get("lineage") or {}
        if ln.get("first"):
            joined = [(d, a) for a, d in ln["joined"].items() if d > ln["first"]]
            txt = f"[lineage] first seen {human_day(ln['first'])}"
            S.add("out", txt, {"type": "lineage", "id": cid})
            if joined:
                d, a = max(joined)
                who = "joined today" if d == R["date"] else f"joined {human_day(d)}"
                S.add("dim", f"[lineage]   {_n(a, 16)} {who}")
        else:
            S.add("dim", "[lineage] new thread today")
        at = c.get("attention")
        if at:
            if at["today_break"]:
                S.add("out", f"[nrf] {MOVES.get(at['today_kind'], at['today_kind'])}",
                      {"type": "balance", "id": cid, "ok": False})
            else:
                S.add("ok", "[nrf] cluster total held steady today", {"type": "balance", "id": cid, "ok": True})
            S.add("dim", f"[nrf]   broke on {at['breaks']} of {at['steps']} days")
        S.blank()
    return S


# ----------------------------------------------------------------------------- session 3
def session_lifecycle(R):
    S = Session("wikipedia news today · story lifecycles")
    S.add("cmd", f"nbe lifecycle --date {_iso(R['date'])}")
    P = R.get("petra")
    ids = [a for a in R["stories"] if R.get("gate", {}).get(a, {}).get("status") != "refused"]
    if not P:
        S.add("warn", "[petra] life-cycle stage failed", {"type": "add", "ids": ids})
    else:
        S.add("out", f"[petra] {P['cases']['days']} days of top-100 lists", {"type": "add", "ids": ids})
        d = P["gate"]["dropped"]
        S.add("dim", f"[lfc] log gate dropped {sum(d.values())} non-stories")
        S.add("dim", "[lfc]   " + ", ".join(f"{v} {k}" for k, v in d.items()))
        S.add("out", f"[petra] {_num(P['cases']['train'])} finished stories → net")
        n = P["net"]
        S.add("ok" if n["sound"] else "warn",
              f"[petra] {n['places']} states, {n['transitions']} moves, " + ("sound" if n["sound"] else "NOT sound"))
        S.add("dim", f"[petra]   {P['steps']} replay steps, NLL {P['loss'][0]:.2f}→{P['loss'][1]:.2f}")
        r0 = next((r for r in P["rules"] if r["state"] == "ENTER"), None)
        if r0:
            S.add("out", f"[petra] after entry: {r0['p_exit']:.0%} gone next day")
            S.add("dim", f"[petra]   counts say {r0['emp_exit']:.0%} (n={_num(r0['support'])})")
        S.blank()
        live = {a: v for a, v in P["today"].items() if a in ids and v.get("p_last_day") is not None}
        order = sorted(live, key=lambda a: -live[a]["p_last_day"])
        pick = []
        if order:
            pick = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]
            pick = list(dict.fromkeys(pick))
        for a in pick:
            v = live[a]
            S.add("hi", f"[petra] {_n(a, 17)} P(last day) {v['p_last_day']:.2f}", {"type": "focus", "id": a})
            if v["count_n"]:
                S.add("dim", f"[petra]   day {v['days']}; counts {v['count_p']:.2f} (n={_num(v['count_n'])})")
        S.blank()
    C = R.get("claims")
    if not C:
        S.add("warn", "[lfc] claim gate failed")
        return S
    S.add("out", "[lfc] gate decay claims: v = A·2^(−t/h)")
    shown = 0
    soon = [a for a, c in C.items() if c["status"] == "refused" and c["gaps"] and c["gaps"][0][0] == "insufficient_data"]
    poor = [a for a, c in C.items() if c["status"] == "refused" and any(g[0] == "poor_fit" for g in c["gaps"])]
    series = [a for a, c in C.items() if c["status"] == "refused" and c["stage"] == "series"]
    corr = [a for a, c in C.items() if c["status"] == "corrected"]
    acc = sorted((a for a, c in C.items() if c["status"] == "accepted"),
                 key=lambda a: -C[a]["claim"]["r2"])
    if soon:
        a = max(soon, key=lambda a: R["views_today"].get(a, 0))
        g = C[a]["gaps"][0]
        ago = (parse_ymd(R["date"]) - parse_ymd(g[2])).days
        S.add("refused", f"[lfc] REFUSED {_n(a, 18)}: too soon", {"type": "focus", "id": a})
        S.add("dim", f"[lfc]   peak {ago} day{'s' if ago != 1 else ''} ago; need 5 days of decay"
              if ago else "[lfc]   peak is today; need 5 days of decay")
        shown += 1
    if poor:
        a = poor[0]
        g = next(g for g in C[a]["gaps"] if g[0] == "poor_fit")
        S.add("refused", f"[lfc] REFUSED {_n(a, 18)}: no clean fit", {"type": "focus", "id": a})
        S.add("dim", f"[lfc]   R² {g[1]:.2f} < 0.5 on the raw views")
    if series:
        S.add("refused", f"[lfc] REFUSED {_n(series[0], 16)}: bad input")
        S.add("dim", "[lfc]   series gate failed; claim not checked")
    if corr:
        a = corr[0]
        S.add("ok", f"[lfc] CORRECTED {_n(a, 18)}", {"type": "focus", "id": a})
        S.add("dim", f"[lfc]   claimed {C[a]['claim']['r2']:.2f}, data say {C[a]['final']['r2']:.2f} (R²)")
    if acc:
        a = acc[0]
        cl = C[a]["final"]
        S.add("ok", f"[lfc] ACCEPTED {_n(a, 18)}: t½ {cl['half_life']:.1f} d", {"type": "focus", "id": a})
    st = [c["status"] for c in C.values()]
    S.add("dim", f"[lfc] {st.count('refused')} refused, {st.count('corrected')} corrected, "
                 f"{st.count('accepted')} accepted")
    return S


def compose(R):
    LBL.clear()
    LBL.update(make_labels(R.get("stories", [])))
    for i, c in enumerate((R.get("net") or {}).get("clusters", [])):
        c["id"] = f"c{i + 1}"
    sessions = []
    for fn in (session_today, session_connect, session_lifecycle):
        try:
            sessions.append(fn(R).doc())
        except Exception as e:      # a broken section must not take the panel down
            sessions.append({"title": "wikipedia news today", "lines": [["warn", f"[story] {fn.__name__} failed"],
                                                                     ["dim", f"  {type(e).__name__}"[:46]]]})
    graph = build_graph(R) if R.get("stories") else {"nodes": [], "links": [], "clusters": []}
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "date": _iso(R["date"]), "graph": graph, "sessions": sessions}
