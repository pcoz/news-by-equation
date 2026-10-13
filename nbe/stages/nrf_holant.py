"""Stage: numeric-rule-finder (NRF) + holant-tools / structural-computing.

    relatedness(articles, links)      integer edge weights from shared out-links + direct links
    clusters(edges)                   story clusters = NRF exact independent groups
    bridges(cluster, edges)           articles whose removal splits their cluster
    best_pairing(articles, W)         exact max-weight pairing (structural-computing -> holant-tools),
                                      checked against exhaustive search per component
    via_names_before / via_names      naming what ties two stories (old vs specificity-weighted)
    counter_laws(channels, today)     NRF: laws of Wikimedia's 7 traffic counters, checked today
    cluster_attention(cluster, hist)  NRF: is a cluster's total attention conserved? typed breaks
    lineage(cluster, day_clusters)    when the thread first appeared and who joined when
"""
from __future__ import annotations

import collections
import itertools
import math
import re

from numeric_rule_finder import Reconciler, check_conservation_noisy, discover_invariants
from numeric_rule_finder.invariants import check_conservation
from structural_computing import NotInFamily, Orchestrator, StructuralComputer

from ..util import parse_ymd

GENERIC_DF = 4          # an out-link shared by >= 4 of the day's stories is not a tie
MIN_WEIGHT = 50         # edge kept if 1000*(cosine + 0.25*direct) >= 50

# Navbox / reference / geography links that say nothing about what ties two stories.
GENERIC = re.compile(
    r"^(List_of_|Index_of_|Outline_of_|Timeline_of_|Lists_of_)|_\(identifier\)$|^\d{1,4}s?(_BC)?$|^\d{4}_in_|"
    r"^(January|February|March|April|May|June|July|August|September|October|November|December)(_\d{1,2})?$|"
    r"^(United_States|United_Kingdom|England|Canada|Australia|India|New_York_City|Los_Angeles|California|"
    r"New_York_\(state\)|London|Americans|American_people|English_language|Netflix|IMDb|Rotten_Tomatoes|"
    r"Metacritic|AllMovie|Box_Office_Mojo|TV_Guide|Variety_\(magazine\)|The_Hollywood_Reporter|Deadline_Hollywood|"
    r"The_New_York_Times|The_Washington_Post|The_Guardian|BBC|BBC_News|CNN|Associated_Press|Reuters|NBC_News|"
    r"ABC_News|ABC_News_\(United_States\)|CBS_News|Fox_News|NPR|USA_Today|Los_Angeles_Times|People_\(magazine\)|"
    r"Time_\(magazine\)|The_Independent|The_Daily_Telegraph|Newsweek|Rolling_Stone|"
    r"Wikidata|Wikipedia|YouTube|Instagram|Facebook|Twitter|X_\(social_network\)|TikTok|Amazon_Prime_Video|"
    r"Prime_Video|HBO|Hulu|Disney\+|Apple_TV\+|Television|Film|Actor|Actress|Television_film|Streaming_media|"
    r"Miniseries|Television_miniseries|Limited_series|Television_show|Television_series|Streaming_television|"
    r"Film_director|Screenwriter|Record_producer|Singer|Comedian|Model_\(person\)|Podcast|"
    r"Primetime_Emmy_Award|Golden_Globe_Award|Academy_Awards|Screen_Actors_Guild_Award|Democratic_Party_\(United_States\)|"
    r"Republican_Party_\(United_States\))$")


# ---------------------------------------------------------------- relatedness / clusters
def relatedness(articles, links, generic_df=GENERIC_DF, scale=1000):
    """W[(a,b)] = round(1000 * (cosine of shared out-links + 0.25 per direct link direction)).
    Out-links held by >= generic_df of the given articles and '(identifier)' links are dropped.
    Returns (W, D) with D[(a,b)] = {"shared": sorted shared links, "direct": 0..2}."""
    L = {a: links.get(a, set()) for a in articles}
    df = collections.Counter(x for a in articles for x in L[a])
    F = {a: {x for x in L[a] if df[x] < generic_df and not x.endswith("(identifier)")} for a in articles}
    W, D = {}, {}
    for a, b in itertools.combinations(articles, 2):
        sh = F[a] & F[b]
        cos = len(sh) / math.sqrt(len(F[a]) * len(F[b])) if F[a] and F[b] else 0.0
        direct = (b in L[a]) + (a in L[b])
        W[(a, b)] = round(scale * (cos + 0.25 * direct))
        D[(a, b)] = {"shared": sorted(sh), "direct": direct}
    return W, D


def edges_of(W, min_weight=MIN_WEIGHT):
    return [e for e, w in W.items() if w >= min_weight]


def clusters(edges):
    """Story clusters = NRF's exact one-pass independent groups over (article, edge) rows."""
    rows = [{"article": x, "edge": i} for i, e in enumerate(edges) for x in e]
    return [sorted(c) for c in Reconciler(rows).independent_groups("article", "edge")] if rows else []


def bridges(cluster, edges):
    """Articles whose removal splits their cluster (exact: re-group without them)."""
    out = []
    for a in cluster:
        rest = [e for e in edges if a not in e and e[0] in cluster and e[1] in cluster]
        left = set(cluster) - {a}
        groups = clusters(rest)
        covered = set().union(*map(set, groups)) if groups else set()
        if len(groups) + len(left - covered) > 1:
            out.append(a)
    return out


def _brute_max_matching(vertices, W):
    vs = list(vertices)
    if not vs:
        return 0
    v, rest = vs[0], vs[1:]
    best = _brute_max_matching(rest, W)
    for u in rest:
        w = W.get((v, u), W.get((u, v), 0))
        if w > 0:
            best = max(best, w + _brute_max_matching([x for x in rest if x != u], W))
    return best


def best_pairing(articles, W, min_weight=MIN_WEIGHT, brute_limit=12):
    """Exact max-weight pairing of the stories by relatedness.

    Max-weight matching == min-weight PERFECT matching on K_n with cost -w
    (unrelated pairs cost 0), which structural-computing routes to holant-tools.
    Verified against exhaustive search on every relatedness component of size <= brute_limit."""
    arts = list(articles)[: len(articles) // 2 * 2]
    Wm = {k: (w if w >= min_weight else 0) for k, w in W.items() if k[0] in arts and k[1] in arts}
    res = Orchestrator().evaluate({"vertices": arts, "edges": list(Wm), "weights": {e: -w for e, w in Wm.items()}},
                                  "min_weight_matching")
    ans = res.answer
    pairs = [(u, v, Wm.get((u, v), Wm.get((v, u), 0))) for u, v in ans["matching"]]
    pairs = sorted((p for p in pairs if p[2] > 0), key=lambda p: -p[2])
    total = int(-ans["cost"])
    rel = [e for e, w in Wm.items() if w > 0]
    comps = clusters(rel)
    checked = all(len(c) <= brute_limit for c in comps)
    brute = sum(_brute_max_matching(c, Wm) for c in comps if len(c) <= brute_limit)
    sc = StructuralComputer()
    pm = {}
    for c in comps:
        ce = [e for e in rel if e[0] in c and e[1] in c]
        try:
            pm[tuple(c)] = sc.count_matchings(ce)
        except NotInFamily:
            pm[tuple(c)] = None
    return {"pairs": pairs, "total": total, "verified": checked and brute == total, "checked_all": checked,
            "evaluator": res.leaf_evaluator_used, "perfect_matchings": pm}


# ---------------------------------------------------------------- naming the ties
def via_names_before(pair, D, corpus, k=3):
    """Prototype naming: direct link first, then shared links that are themselves
    trending pages in the corpus, then alphabetical."""
    d = D[pair]
    sh = sorted(d["shared"], key=lambda x: (x not in corpus, x))
    return (["direct link"] if d["direct"] else []) + sh[:k]


def link_df(link_sets):
    """Document frequency of each out-link across a day's articles."""
    return collections.Counter(x for s in link_sets for x in set(s))


def via_names(pair, links, lead, df, n_docs, k=3):
    """Specificity-weighted naming: each shared out-link scores idf = log(N/df)
    (df over all of the day's articles' link sets), x2 for each article whose LEAD
    section carries it; navbox/generic links are dropped. Returns [(link, score, in_a_lead)]."""
    a, b = pair
    shared = (links.get(a, set()) & links.get(b, set()))
    la, lb = set(lead.get(a) or ()), set(lead.get(b) or ())
    out = []
    for x in shared:
        if GENERIC.search(x) or x in (a, b):
            continue
        idf = math.log((n_docs + 1) / (df.get(x, 0) + 1))
        if idf <= 0:
            continue
        n_lead = (x in la) + (x in lb)
        out.append((x, idf * (1 + n_lead), n_lead > 0))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out[:k]


# ---------------------------------------------------------------- NRF: counters
def counter_laws(channels, today):
    """Which accounting laws do Wikimedia's 7 traffic counters obey? NRF discovers
    the left null space blind on history (all days before today); today's numbers
    are then checked against those laws. channels: {article: {"access/agent": {day: n}}}."""
    hist_rows, today_rows = [], []
    for a, ch in channels.items():
        days = sorted(set().union(*map(set, ch.values()))) if ch else []
        for d in days:
            for k, v in ch.items():
                (today_rows if d == today else hist_rows).append({"counter": k, "event": f"{a}@{d}",
                                                                  "views": v.get(d, 0)})
    disc = discover_invariants(hist_rows, "counter", "event", "views")
    checked = check_conservation(today_rows, disc.laws, "counter", "event", "views")
    breaks = [(ev, int(val)) for _law, _t, viol in checked for ev, val in viol]
    return {"n_counters": len(disc.entities), "n_events": len(disc.events), "n_laws": disc.n_laws,
            "laws": [l.render() for l in disc.laws], "breaks_today": breaks, "n_today": len(today_rows)}


# ---------------------------------------------------------------- NRF: cluster attention
class _SumLaw:
    """'sum of members = const' as a law object NRF's checker accepts."""

    def __init__(self, members):
        self.coeffs = {m: 1 for m in members}

    def render(self):
        return " + ".join(self.coeffs) + "  =  const"


def cluster_attention(cluster, hist, today):
    """NRF on attention inside a cluster: is the members' total human attention
    conserved day to day? Noise per member = robust sd of its own daily changes
    (MAD*1.4826); a day breaks the law at |z| > 3. Each break is typed by its
    significant movers: one member ("solo"), several the same way ("co-surge" /
    "co-fade"), or opposite ways ("substitution")."""
    if not all(a in hist for a in cluster):
        return None
    common = sorted(set.intersection(*(set(hist[a]) for a in cluster)))
    if len(common) < 15:
        return None
    flows = [{"a": a, "step": d1, "dv": hist[a][d1] - hist[a][d0]}
             for d0, d1 in zip(common, common[1:]) for a in cluster]
    disc = discover_invariants(flows, "a", "step", "dv")
    for a in cluster:
        ds = sorted(abs(f["dv"]) for f in flows if f["a"] == a)
        mad = ds[len(ds) // 2] * 1.4826 or 1.0
        for f in flows:
            if f["a"] == a:
                f["sd"] = mad
    rep = check_conservation_noisy(flows, [_SumLaw(cluster)], "a", "step", "dv", sigma_key="sd", z_break=3.0)
    viol = {v: (val, z) for v, val, _s, z in rep.results[0].violators}
    movers = collections.defaultdict(dict)
    for f in flows:
        if abs(f["dv"]) > 3 * f["sd"]:
            movers[f["step"]][f["a"]] = f["dv"]

    def kind(d):
        m = movers[d]
        if len(m) <= 1:
            return "solo"
        signs = {x > 0 for x in m.values()}
        return "substitution" if len(signs) == 2 else ("co-surge" if True in signs else "co-fade")

    kinds = collections.Counter(kind(d) for d in viol)
    t = viol.get(today)
    return {"n_laws": disc.n_laws, "steps": len(common) - 1, "breaks": len(viol), "kinds": dict(kinds),
            "today_break": t is not None, "today_kind": kind(today) if t is not None else None,
            "today_residual": int(t[0]) if t else None, "today_z": float(t[1]) if t else None,
            "today_movers": dict(movers.get(today, {}))}


# ---------------------------------------------------------------- lineage
def lineage(cluster, day_clusters, today, max_gap=7):
    """Story thread back in time. A past day belongs to the thread if one of its
    clusters shares >= 2 articles with today's cluster (or all of a 2-story one);
    walking back from today, the thread ends at a gap of > max_gap days.
    Returns the first thread day and, per member, the first thread day it was in
    the overlapping cluster (members never seen join today)."""
    cset = set(cluster)
    need = min(2, len(cset))
    hits = {}
    for d in sorted((d for d in day_clusters if d < today), reverse=True):
        ov = [cl for cl in day_clusters[d] if len(set(cl) & cset) >= need]
        if ov:
            hits[d] = ov
    run, prev = [], today
    for d in sorted(hits, reverse=True):
        if (parse_ymd(prev) - parse_ymd(d)).days > max_gap + 1:
            break
        run.append(d)
        prev = d
    joined = {a: today for a in cluster}
    for d in sorted(run):
        for cl in hits[d]:
            for a in cl:
                if a in cset and joined[a] == today:
                    joined[a] = d
    return {"first": min(run) if run else None, "days": len(run), "joined": joined}


# ---------------------------------------------------------------- cluster labels
_STOP = {"the", "of", "and", "a", "an", "in", "on", "for", "to", "film", "series", "season", "novel", "miniseries",
         "united", "states", "day"}


def _words(title):
    return re.sub(r"\s*\([^)]*\)\s*$", "", title.replace("_", " ")).split()


def _common_run(a, b):
    """Longest common contiguous word run of two word lists."""
    best = []
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k].lower() == b[j + k].lower():
                k += 1
            if k > len(best):
                best = a[i:i + k]
    return best


def cluster_label(members, ties):
    """Short label: a word run shared by member titles ("East of Eden", "Columbus"),
    else the best specificity-weighted tie name, grouped by base name
    ("Spider-Man (2002 film)" + "Spider-Man 3" -> "Spider-Man")."""
    runs = []
    for a, b in itertools.combinations(members, 2):
        r = _common_run(_words(a), _words(b))
        if any(len(w) >= 4 and w.lower() not in _STOP for w in r):
            runs.append(" ".join(r))
    if runs:
        return max(runs, key=lambda r: (runs.count(r), len(r))), "title"
    score = collections.Counter()
    for t in ties:
        for x, s, *_ in t["after"]:
            base = re.sub(r"(\s+\d+)$", "", " ".join(_words(x)))
            score[base] += s
    if score:
        return score.most_common(1)[0][0], "link"
    return None, None
