"""Compose the hero-panel narrative (nbe.json) from one run's results.

Every number printed here comes from the run's results dict `R` (computed this
run, or loaded from state/ with its provenance). A section whose stage failed
prints an honest `warn` line instead. Terminal lines are kept <= 56 chars where
possible (the panel fits ~62-73 on laptop/desktop); the validator enforces <= 60.
"""
from __future__ import annotations

import datetime as dt
import statistics

from .select import SENSITIVE
from .util import fmt_views, human_day, label, parse_ymd

MAX_LINE = 56
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



def _plain_reason(gate):
    """Why the series gate refused a story, for people: (short, sentence).

    The gate's own reasons are terse ("desktop share 100% vs usual 56%");
    these say what that means and why it disqualifies the numbers."""
    import re
    for g in gate.get("gaps", []):
        kind, day = g[0], human_day(g[1]) if len(g) > 1 else ""
        text = g[2] if len(g) > 2 else ""
        m = re.search(r"desktop share (\d+)% vs usual (\d+)%", text)
        if kind == "bot" and m:
            now, usual = int(m.group(1)), int(m.group(2))
            share = "all" if now >= 99 else f"{now}%"
            return ("looks like bot traffic",
                    f"From {day}, {share} of the views Wikipedia counted as human came from "
                    f"desktop computers. Normally it's about {usual}%, with the rest on phones. "
                    "Real readers don't switch devices overnight, so these are almost "
                    "certainly automated programs (bots) being counted as people.")
        m = re.search(r"automated agents (\d+)%", text)
        if kind == "bot" and m:
            return ("mostly automated traffic",
                    f"On {day}, {m.group(1)}% of all traffic to this page came from software "
                    "Wikipedia itself flags as automated, so its count of human readers "
                    "can't be trusted either.")
        if kind == "bot":
            return ("looks like bot traffic",
                    "Its daily views stayed almost exactly level for days on end. Real "
                    "attention rises and falls; flat lines like that come from automated "
                    "traffic (bots).")
        if kind == "missing":
            return ("days of data missing",
                    f"Wikipedia's view counts are missing for some days (from {day}), so its "
                    "attention curve can't be analysed reliably.")
        if kind == "implausible":
            return ("impossible values",
                    f"Some of its view counts (from {day}) are impossible, so the data "
                    "can't be trusted.")
    return ("failed the data check",
            "Its page-view data failed the automatic data check, so no claims are made about it.")


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
        g = gate.get(a)
        if g and g.get("status") == "refused":
            node["why"] = _plain_reason(g)[1]
        nodes.append(node)
    ids = {n["id"] for n in nodes}
    links = [{"source": a, "target": b, "weight": round(min(1.0, w / 1000), 2)}
             for a, b, w in net.get("links", []) if a in ids and b in ids]
    return {"nodes": nodes, "links": links, "clusters": clusters}


def _info(a, s, g, R):
    if g and g["status"] == "refused":
        return "refused: " + _plain_reason(g)[0]
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
def _gate_detail(g):
    """One plain line of evidence behind a series-gate refusal, or None."""
    import re
    for gp in g.get("gaps", []):
        day = human_day(gp[1]) if len(gp) > 1 else ""
        text = gp[2] if len(gp) > 2 else ""
        m = re.search(r"desktop share (\d+)% vs usual (\d+)%", text)
        if gp[0] == "bot" and m:
            now = "all" if int(m.group(1)) >= 99 else f"{m.group(1)}% of"
            return f"from {day}, {now} views on desktop (usually {m.group(2)}%)"
        m = re.search(r"automated agents (\d+)%", text)
        if gp[0] == "bot" and m:
            return f"on {day}, {m.group(1)}% of traffic was flagged automated"
        if gp[0] in ("missing", "implausible") and day:
            return f"the problem starts {day}"
    return None


def _formula_lines(S, s):
    """The plain reading of a fitted formula: extra views at the peak, how fast
    they halve, the level they settle at, and how well the curve fits."""
    import re
    m = re.match(r"v\(t\) ≈ ([\d.]+[kM]?)·\S+?(?: ([+−]) ([\d.]+[kM]?))?$", s["formula"])
    if m:
        S.add("dim", f"[formula]   peak: +{m.group(1)} views, halving every {s['half_life']:.1f} days")
    else:
        S.add("dim", f"[formula]   the extra views halve every {s['half_life']:.1f} days")
    fit = f"fit quality {s['r2']:.0%}"
    if m and m.group(2) == "+":
        S.add("dim", f"[formula]   settling at {m.group(3)}/day · {fit}")
    else:
        S.add("dim", f"[formula]   {fit}")


def session_today(R):
    S = Session("wikipedia news today · top stories")
    D = R["date"]
    S.add("cmd", f"nbe run --date {_iso(D)}")
    sk = R.get("skipped", {})
    n_skip = R.get("junk_today", 0) + len(sk.get("perennial", [])) + len(sk.get("evergreen", []))
    S.add("out", f"[fetch] Wikipedia's most-read, {human_day(D)}: {len(R['stories'])} stories",
          {"type": "add", "ids": R["stories"]})
    parts = [f"{R.get('junk_today', 0)} junk"]
    if sk.get("perennial"):
        parts.append(f"{len(sk['perennial'])} always listed")
    if sk.get("evergreen"):
        parts.append(f"{len(sk['evergreen'])} generic")
    S.add("dim", f"[fetch]   skipped {n_skip} pages that aren't news" if len(parts) == 1
          else f"[fetch]   skipped {n_skip} non-news: " + ", ".join(parts))

    # LFC series gate
    gate = R.get("gate", {})
    if _warn_for(R, "verify") or not gate:
        S.add("warn", "[check] data check failed; views not verified")
    else:
        st = [g["status"] for g in gate.values()]
        line = f"[check] view data checked: {st.count('accepted')} passed"
        if st.count("corrected"):
            line += f", {st.count('corrected')} fixed"
        S.add("out", line + f", {st.count('refused')} refused")
        for a, g in gate.items():
            if g["status"] != "refused":
                continue
            why = _plain_reason(g)[0]
            head = f"[check] REFUSED {_n(a, 20)}: {why}"
            if len(head) <= MAX_LINE:
                S.add("refused", head, {"type": "refuse", "id": a})
            else:
                S.add("refused", f"[check] REFUSED {_n(a, 20)}", {"type": "refuse", "id": a})
                S.add("dim", f"[check]   {why}")
            det = _gate_detail(g)
            if det:
                S.add("dim", f"[check]   {det}")
        for a, g in gate.items():
            if g["status"] == "corrected":
                S.add("dim", f"[check]   fixed {_n(a, 16)}: {g['reason'][:24]}")
                break
    c = R.get("counters")
    if c and c.get("n_laws"):
        if not c["breaks_today"]:
            S.add("ok", f"[balance] traffic counts add up ({c['n_counters']} counts, {c['n_laws']} rules)")
        else:
            S.add("warn", f"[balance] view counters don't add up: {len(c['breaks_today'])} breaks")
    elif _warn_for(R, "nrf-counters"):
        S.add("warn", "[balance] traffic-counter check failed")
    S.blank()

    an = R.get("analysis", {})
    if _warn_for(R, "tnn+ppf"):
        S.add("warn", "[tnn] jump detection failed")
    fresh = sorted((a for a, s in an.items() if s["fresh"] and s.get("cp") and not SENSITIVE.search(a)),
                   key=lambda a: -an[a]["cp"]["z"])
    for i, a in enumerate(fresh[:2]):
        s = an[a]
        cp = s["cp"]
        if i == 0:
            S.add("hi", f"[tnn] {_n(a)}: sudden jump on {human_day(cp['date'])}", {"type": "pulse", "id": a})
            S.add("hi", f"[tnn]   {cp['z']:.0f}× bigger than its usual day-to-day swings")
            S.add("dim", f"[tnn]   views {_x(cp['multiple'])} normal: ~{fmt_views(cp['baseline'])}/day → "
                         f"{fmt_views(s['naive'])}")
        else:
            S.add("hi", f"[tnn] {_n(a)}: {_x(cp['multiple'])} normal views, {cp['z']:.0f}× usual swings",
                  {"type": "pulse", "id": a})
    if fresh:
        top = an[fresh[0]]
        b = top.get("bucket")
        if b and b["half_life_q"][1] is not None:
            S.add("out", f"[history] {_num(b['n'])} past {_bucket_text(b['bucket'])} jumps "
                         f"typically halved in {b['half_life_q'][1]:.1f} d")
        pf = top.get("prior_forecast")
        if pf:
            S.add("out", f"[forecast] {_n(fresh[0], 14)} tomorrow: ~{fmt_views(pf['value'])} views")
            S.add("dim", f"[forecast]   half of past cases {fmt_views(pf['lo'])}–{fmt_views(pf['hi'])}; "
                         f"today {fmt_views(top['naive'])}")
    S.blank()

    # maturing stories with a clean, compact formula
    # clean decay fits that still track the last days (tail error <= 20%, level not above today)
    mat = sorted((a for a, s in an.items() if not s["fresh"] and s.get("formula") and s.get("half_life")
                  and s["shape"] == "spike-and-decay" and s["r2"] >= 0.8 and s.get("tail_err", 1) <= 0.2
                  and (s.get("long_run") or 0) <= 1.1 * s["naive"] and not SENSITIVE.search(a)),
                 key=lambda a: -R["views_today"].get(a, 0))
    for i, a in enumerate(mat[:2]):
        s = an[a]
        S.add("out", f"[formula] {_n(a, 20)}, t = days since {human_day(s['x1_date'])} peak",
              {"type": "curve", "id": a})
        S.add("hi", f"[formula] {s['formula']}")
        _formula_lines(S, s)
        if s.get("forecast") is not None:
            S.add("out", f"[forecast] tomorrow ~{fmt_views(s['forecast'])}; "
                         f"'same as today' says {fmt_views(s['naive'])}")
    if not mat and not _warn_for(R, "tnn+ppf"):
        S.add("dim", "[formula] no fading story with a clean fit")

    # honest scorecards: all clean fits AND the guarded subset (guard adopted 13 Oct, post-hoc)
    if _warn_for(R, "score"):
        S.add("warn", "[score] scoring stage failed")
    bt = [b for b in R.get("backtest", []) if b["method"] == "formula"]
    g = R.get("graded") or {}
    if bt or g.get("rows") or g.get("forward_rows"):
        S.add("out", "[score] how far off were forecasts? (lower is better)")
    if bt:
        S.add("out", f"[score]   backtest: fit to {human_day(_prev(R['date']))} → "
                     f"predict {human_day(R['date'])}")
        _score_lines(S, bt, "[score]    ")
    fwd = g.get("forward_rows") or []
    if g.get("rows") and g["rows"][0].get("post_hoc"):
        S.add("out", f"[score]   {human_day(g['issued'])} forecasts, checked against today")
        _score_lines(S, [r for r in g["rows"] if r["method"] == "formula"], "[score]    ")
        pr = [r for r in g["rows"] if r["method"] == "prior"]
        if pr:
            m = statistics.median(r["ape"] for r in pr)
            nv = statistics.median(r["ape_naive"] for r in pr)
            S.add("out", f"[score]    {len(pr)} from past jumps: {m:.0f}% · 'same as today' {nv:.0f}%")
    if fwd:
        S.add("out", f"[score]   new forecasts since {human_day(g['adopted'])}")
        _score_lines(S, [r for r in fwd if r["method"] == "formula"], "[score]    ")
    else:
        S.add("dim", f"[score] scoring of new forecasts starts {human_day(g.get('adopted') or R['date'])}")
    return S


def _score_lines(S, rows, pre):
    """Median abs error of ALL clean-fit forecasts and of the guarded subset, each next to
    repeat-today on the same stories, so the post-hoc guard never hides the unguarded result."""
    if not rows:
        return
    allm = statistics.median(r["ape"] for r in rows)
    alln = statistics.median(r["ape_naive"] for r in rows)
    S.add("out", f"{pre}all {len(rows)} formulas: {allm:.0f}% · 'same as today' {alln:.0f}%")
    gd = [r for r in rows if r.get("guarded")]
    if gd:
        gm = statistics.median(r["ape"] for r in gd)
        gn = statistics.median(r["ape_naive"] for r in gd)
        S.add("out", f"{pre}{len(gd)} best-fitting: {gm:.0f}% · 'same as today' {gn:.0f}%")


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


MOVES = {"co-surge": "today their views rose together",
         "co-fade": "today their views fell together",
         "substitution": "today one rose as another fell",
         "solo": "today one story moved on its own"}


def session_connect(R):
    S = Session("wikipedia news today · how stories connect")
    S.add("cmd", f"nbe connect --date {_iso(R['date'])}")
    net = R.get("net")
    if not net:
        S.add("warn", "[links] link stage failed; no story map")
        return S
    S.add("out", f"[links] {net['n_articles']} stories, {net['n_edges']} links between them",
          {"type": "add", "ids": R["stories"]})
    S.add("dim", "[links]   (pages that cite each other or the same pages)")
    p = net["pairing"]
    if not p.get("verified") and p.get("checked_all"):
        S.add("warn", "[links]   pairing check disagrees")
    S.add("out", f"[clusters] {len(net['clusters'])} groups of related stories")
    S.blank()
    for c in net["clusters"][:4]:
        cid = c.get("id")
        lab = label(c["label"] or "", 18) if c.get("label") else "untitled"
        S.add("out", f"[clusters] \"{lab}\": {len(c['members'])} stories",
              {"type": "cluster", "id": cid, "label": label(c["label"] or "", 18)})
        via = _via(c)
        if via:
            line = "[clusters]   via " + via
            S.add("dim", line if len(line) <= MAX_LINE else line.split(" · ")[0])
        for b in c["bridges"][:1]:
            S.add("hi", f"[links] {_n(b, 22)} holds this group together", {"type": "bridge", "id": b})
        ln = c.get("lineage") or {}
        if ln.get("first"):
            joined = [(d, a) for a, d in ln["joined"].items() if d > ln["first"]]
            S.add("out", f"[history] this story thread began {human_day(ln['first'])}",
                  {"type": "lineage", "id": cid})
            if joined:
                d, a = max(joined)
                who = "joined today" if d == R["date"] else f"joined on {human_day(d)}"
                S.add("dim", f"[history]   {_n(a, 16)} {who}")
        else:
            S.add("dim", "[history] a new story thread today")
        at = c.get("attention")
        if at:
            if at["today_break"]:
                S.add("out", f"[balance] {MOVES.get(at['today_kind'], at['today_kind'])}",
                      {"type": "balance", "id": cid, "ok": False})
            else:
                S.add("ok", "[balance] their combined views held steady today",
                      {"type": "balance", "id": cid, "ok": True})
            S.add("dim", f"[balance]   combined views held steady "
                         f"{at['steps'] - at['breaks']} of {at['steps']} days")
        S.blank()
    return S


# ----------------------------------------------------------------------------- session 3
DROPPED = {"perennial": "always listed", "evergreen": "generic", "bot": "bot-like"}


def session_lifecycle(R):
    S = Session("wikipedia news today · story lifecycles")
    S.add("cmd", f"nbe lifecycle --date {_iso(R['date'])}")
    P = R.get("petra")
    ids = [a for a in R["stories"] if R.get("gate", {}).get(a, {}).get("status") != "refused"]
    if not P:
        S.add("warn", "[lifecycle] life-cycle stage failed", {"type": "add", "ids": ids})
    else:
        S.add("out", f"[lifecycle] {P['cases']['days']} days of Wikipedia top-100 lists",
              {"type": "add", "ids": ids})
        d = P["gate"]["dropped"]
        S.add("dim", f"[check] set aside {sum(d.values())} non-stories from the lists")
        S.add("dim", "[check]   " + ", ".join(f"{v} {DROPPED.get(k, k)}" for k, v in d.items()))
        S.add("out", f"[lifecycle] learned from {_num(P['cases']['train'])} past stories")
        n = P["net"]
        S.add("ok" if n["sound"] else "warn",
              f"[lifecycle] model: {n['places']} stages, {n['transitions']} steps, "
              + ("no dead ends" if n["sound"] else "has dead ends"))
        r0 = next((r for r in P["rules"] if r["state"] == "ENTER"), None)
        if r0:
            S.add("out", f"[lifecycle] new entries: {r0['p_exit']:.0%} gone the next day")
            S.add("dim", f"[lifecycle]   raw counts say {r0['emp_exit']:.0%} (of {_num(r0['support'])})")
        S.blank()
        live = {a: v for a, v in P["today"].items() if a in ids and v.get("p_last_day") is not None}
        order = sorted(live, key=lambda a: -live[a]["p_last_day"])
        pick = []
        if order:
            pick = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]
            pick = list(dict.fromkeys(pick))
        if pick:
            S.add("out", "[lifecycle] chance of dropping out of top 100 tomorrow:")
        for a in pick:
            v = live[a]
            S.add("hi", f"[lifecycle]   {_n(a, 22)}: {v['p_last_day']:.0%}", {"type": "focus", "id": a})
            if v["count_n"]:
                S.add("dim", f"[lifecycle]     day {v['days']}; past: "
                             f"{v['count_p']:.0%} of {_num(v['count_n'])} dropped out")
        S.blank()
    C = R.get("claims")
    if not C:
        S.add("warn", "[check] claim check failed")
        return S
    S.add("out", "[check] testing claims 'views halve every h days'")
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
        S.add("refused", f"[check] REFUSED {_n(a, 18)}: too soon", {"type": "focus", "id": a})
        S.add("dim", f"[check]   peaked {ago} day{'s' if ago != 1 else ''} ago; needs 5 days of decline"
              if ago else "[check]   peaked today; needs 5 days of decline")
    if poor:
        a = poor[0]
        g = next(g for g in C[a]["gaps"] if g[0] == "poor_fit")
        S.add("refused", f"[check] REFUSED {_n(a, 18)}: poor fit", {"type": "focus", "id": a})
        S.add("dim", f"[check]   curve fits real views {max(g[1], 0):.0%} (needs 50%)")
    if series:
        S.add("refused", f"[check] REFUSED {_n(series[0], 16)}: bad input")
        S.add("dim", "[check]   failed the data check, so not tested")
    if corr:
        a = corr[0]
        S.add("ok", f"[check] CORRECTED {_n(a, 18)}", {"type": "focus", "id": a})
        S.add("dim", f"[check]   claimed fit {C[a]['claim']['r2']:.0%}; on real views "
                     f"{C[a]['final']['r2']:.0%}, fixed")
    if acc:
        a = acc[0]
        cl = C[a]["final"]
        S.add("ok", f"[check] ACCEPTED {_n(a, 18)}: halves every {cl['half_life']:.1f} d",
              {"type": "focus", "id": a})
    st = [c["status"] for c in C.values()]
    S.add("dim", f"[check] all {len(st)}: {st.count('refused')} refused, {st.count('corrected')} corrected, "
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
                                                                     ["dim", f"  {type(e).__name__}"[:MAX_LINE]]]})
    graph = build_graph(R) if R.get("stories") else {"nodes": [], "links": [], "clusters": []}
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "date": _iso(R["date"]), "graph": graph, "sessions": sessions}
