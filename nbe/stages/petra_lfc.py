"""Stage: PETRA (petra-nn) story life cycles + llm-feedback-control verify-and-refuse gates.

PETRA
    build_cases(top_lists, today)   one case per (article, episode in the top 100)
    gate_log(cases)                 LFC feedback_loop drops perennial / evergreen / bot-flat cases
    train_petra(train)              order-aware net (discover_transition_system) + train_by_replay
    lifecycle(...)                  net summary, rules, P(today is the last day in the top 100)
                                    with a plain count-table cross-check

LFC
    verify_series(article, series, aux, refetch)  completeness / plausibility / bot gate
    fit_decay_claim(article, series)              deterministic decay-claim generator
    verify_claim(claim, series, aux, refetch)     cascade: series gate -> claim gate
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict

import torch
from llm_feedback_control import cascade, combine_references, feedback_loop, loop_stage, schmitt_gate
from petri_net_nn import (PetriNetModule, XESEvent, XESTrace, check_soundness, discover_transition_system,
                          extract_next_step_rules, next_step_distribution, train_by_replay)

from ..util import day_range

# =========================================================================== PETRA
EXIT_GAP = 14       # absent > 14 days => the story's episode is over
PERENNIAL = 0.40    # in the list on > 40% of days => not a news story
EVERGREEN = re.compile(r"^(United_States|United_Kingdom|India|China|Russia|Canada|Australia|"
                       r"Donald_Trump|Elon_Musk|Taylor_Swift|World_War_II|\..+|.+\.(xyz|io|ai|co|me))$")
STATES = ("UP", "HOLD", "FALL", "CRASH", "RETURN", "EXIT")


def _clip01(x):
    return min(max(float(x), 0.0), 1.0)


def day_state(prev_views, views, returned):
    if returned:
        return "RETURN"            # back in the list after >= 1 day out
    r = views / prev_views
    return "UP" if r > 1.15 else "HOLD" if r > 0.85 else "FALL" if r > 0.5 else "CRASH"


def build_cases(top_lists, today):
    """One case per (article, episode); one event per list day: ENTER, then
    UP/HOLD/FALL/CRASH/RETURN, then EXIT once out of the list for > 14 days.
    Attributes: entry_strength = log10(entry views / that day's #100 views),
    presence = share of days in the list, flat_days = longest run of consecutive
    list days with views within 1% (cheap bot signal)."""
    days = sorted(d for d in top_lists if d <= today)
    T = len(days) - 1
    cut = [top_lists[d][-1]["views"] for d in days]
    pres = {}
    for i, d in enumerate(days):
        for a in top_lists[d]:
            pres.setdefault(a["article"], {})[i] = a["views"]
    cases = []
    for art, dv in pres.items():
        idx = sorted(dv)
        eps = [[idx[0]]]
        for i in idx[1:]:
            (eps.append([i]) if i - eps[-1][-1] - 1 > EXIT_GAP else eps[-1].append(i))
        for ep in eps:
            if ep[0] < 3:
                continue                                   # left-censored
            ongoing = ep[-1] == T
            if not ongoing and ep[-1] > T - EXIT_GAP - 1:
                continue                                   # right-censored, not today
            ev, flat, run = [XESEvent("ENTER", {"day": days[ep[0]]})], 1, 1
            for prev, i in zip(ep, ep[1:]):
                ev.append(XESEvent(day_state(dv[prev], dv[i], i - prev > 1), {"day": days[i]}))
                run = run + 1 if (i - prev == 1 and abs(dv[i] / dv[prev] - 1) <= 0.01) else 1
                flat = max(flat, run)
            if not ongoing:
                ev.append(XESEvent("EXIT"))
            attrs = {"concept:name": art, "start": days[ep[0]],
                     "entry_strength": math.log10(dv[ep[0]] / cut[ep[0]]),
                     "days_in_list": len(ep), "presence": len(dv) / len(days), "flat_days": flat}
            cases.append((XESTrace(attributes=attrs, events=ev), ongoing))
    return cases, days


def _ref_perennial(src, log):
    return [("perennial", c.name, f"in top-100 on {c.attributes['presence']:.0%} of days")
            for c in log if c.attributes["presence"] > PERENNIAL]


def _ref_evergreen(src, log):
    return [("evergreen", c.name, "evergreen/junk title") for c in log if EVERGREEN.match(c.name)]


def _ref_flat(src, log):
    return [("bot", c.name, f"{c.attributes['flat_days']} consecutive days with views within 1%")
            for c in log if c.attributes["flat_days"] >= 3]


LOG_REFERENCE = combine_references(_ref_perennial, _ref_evergreen, _ref_flat)


def gate_log(cases):
    """feedback_loop over the event log: references name non-story cases, the
    repair drops them, the loop re-checks to a fixed point. -> (kept, dropped, converged)."""
    def repair(src, log, gaps):
        bad = {g[1] for g in gaps}
        return [c for c in log if c.name not in bad]
    log, initial, _hist, ok = feedback_loop(cases, extract=list, reference=LOG_REFERENCE, repair=repair,
                                            signature=lambda log: len(log), max_iters=3, label="log")
    dropped = {}
    for g in LOG_REFERENCE(None, initial):
        dropped.setdefault(g[0], {}).setdefault(g[1], g[2])
    return log, dropped, ok


def _es(c):
    return _clip01(c.attributes["entry_strength"])


def _bucket(c):
    e = c.attributes["entry_strength"]
    return 0 if e < .05 else 1 if e < .15 else 2 if e < .3 else 3 if e < .6 else 4


def _decisions(cases):
    for c in cases:
        names = [e.name for e in c.events]
        for i in range(1, len(names)):
            yield c, names[:i], names[i]


def train_petra(train, *, horizon=2, steps=300, lr=0.05, weight_decay=1e-4, seed=0):
    """Order-preserving discovery + order-aware replay training. Each state place
    carries the case's entry strength (clipped to [0, 1]), so every next-step
    choice is a per-state logistic function of how hard the story entered."""
    t0 = time.perf_counter()
    net = discover_transition_system(train, horizon=horizon)
    t_disc = time.perf_counter() - t0
    torch.manual_seed(seed)
    module = PetriNetModule(net, num_steps=1)
    places = sorted(net.places)
    a2m = lambda c: dict.fromkeys(places, _es(c))  # noqa: E731
    t0 = time.perf_counter()
    losses = train_by_replay(module, train, attribute_to_marking=a2m, steps=steps, lr=lr,
                             weight_decay=weight_decay)
    return {"net": net, "module": module, "a2m": a2m, "losses": [float(x) for x in losses], "horizon": horizon,
            "t_disc": t_disc, "t_train": time.perf_counter() - t0,
            "place_of": {v: k for k, v in net.place_labels.items()}}


def petra_next(model, prefix, es):
    """{next state: P} after `prefix` (list of state labels) for entry strength es."""
    p = model["place_of"].get(" > ".join(prefix[-model["horizon"]:]))
    if p is None:
        return {}
    dist = next_step_distribution(model["module"], {p: 1}, place_values={p: es})
    return {model["net"].transition_labels[t]: q for t, q in dist.items()}


def lifecycle(top_lists, today, *, refused_today=(), horizon=2, steps=300, min_support=150):
    """Train on finished stories; score today's ongoing ones. `refused_today`: titles
    the series gate refused (left out of the live set)."""
    t_all = time.perf_counter()
    cases, days = build_cases(top_lists, today)
    kept, dropped, gate_ok = gate_log([c for c, _ in cases])
    keep_ids = {id(c) for c in kept}
    train = [c for c, ongo in cases if not ongo and id(c) in keep_ids]
    live = [c for c, ongo in cases if ongo and id(c) in keep_ids and c.name not in set(refused_today)]
    m = train_petra(train, horizon=horizon, steps=steps)
    net, module = m["net"], m["module"]
    sound = check_soundness(net).is_sound

    # count tables: (last-2 states) and (last-2 states, entry bucket) -> next state
    m2, m2b = defaultdict(Counter), defaultdict(Counter)
    for c, pre, nxt in _decisions(train):
        st = tuple(pre[-horizon:])
        m2[st][nxt] += 1
        m2b[(st, _bucket(c))][nxt] += 1

    med_es = statistics.median(_es(c) for c in train)
    rules = []
    for r in extract_next_step_rules(module, value=med_es, min_choices=2):
        st = tuple(r.place_label.split(" > "))
        n_vis = sum(m2[st].values())
        if n_vis < min_support:
            continue
        probs = dict(r.probabilities)
        rules.append({"state": r.place_label, "support": n_vis, "probs": probs,
                      "p_exit": probs.get("EXIT", 0.0), "emp_exit": m2[st]["EXIT"] / n_vis})
    rules.sort(key=lambda r: -r["support"])

    today_rows = {}
    for c in live:
        names = [e.name for e in c.events]
        st = tuple(names[-horizon:])
        cnt = m2b.get((st, _bucket(c)), Counter())
        src = "state x entry"
        if sum(cnt.values()) < 20:
            cnt, src = m2.get(st, Counter()), "state"
        n = sum(cnt.values())
        today_rows[c.name] = {
            "days": c.attributes["days_in_list"], "entry_x": 10 ** c.attributes["entry_strength"],
            "path": names, "p_last_day": petra_next(m, names, _es(c)).get("EXIT"),
            "count_p": (cnt["EXIT"] / n) if n else None, "count_n": n, "count_src": src}
    return {"net": {"places": len(net.places), "transitions": len(net.transitions),
                    "silent": len(net.silent_transitions), "sound": sound, "horizon": horizon},
            "gate": {"dropped": {k: len(v) for k, v in dropped.items()},
                     "examples": {k: list(v.items())[:3] for k, v in dropped.items()}, "converged": gate_ok},
            "cases": {"train": len(train), "live": len(live), "days": len(days)},
            "loss": (m["losses"][0], m["losses"][-1]), "steps": steps, "rules": rules, "today": today_rows,
            "seconds": {"discover": m["t_disc"], "train": m["t_train"], "total": time.perf_counter() - t_all}}


# =========================================================================== LFC
def ref_complete(src, cand):
    want = day_range(src["start"], src["end"])
    return [("missing", d) for d in want if d not in cand["series"] and d not in cand["excluded"]]


def ref_plausible(src, cand):
    # no jump-size rule: real news spikes exceed 1000x on obscure pages
    return [("implausible", d, f"value {v!r}") for d, v in cand["series"].items()
            if not isinstance(v, int) or v < 0]


def ref_bot(src, cand):
    """Bot-like traffic: (a) desktop-share surge on a high day (Schmitt trigger, so
    a borderline share does not chatter), (b) automated agents a large share of all
    traffic on a high day, (c) flat plateaus (>= 4 elevated days within 0.5%)."""
    s, aux, gaps = cand["series"], src.get("aux") or {}, []
    if not s:
        return gaps
    med = statistics.median(s.values())
    desk, allag = aux.get("desktop", {}), aux.get("all_agents", {})
    base_share = statistics.median([desk[d] / s[d] for d in s if d in desk and s[d] > 0] or [0.5])
    gate = schmitt_gate(base_share + 0.20, base_share + 0.35, start=False)
    for d in sorted(s):
        if d in cand["excluded"] or s[d] <= 0:
            continue
        share = desk.get(d, base_share * s[d]) / s[d]
        if gate(share) and s[d] > 2 * med:
            gaps.append(("bot", d, f"desktop share {share:.0%} vs usual {base_share:.0%}"))
        if d in allag and allag[d] > 0 and (allag[d] - s[d]) / allag[d] > 0.5 and s[d] > 2 * med:
            gaps.append(("bot", d, f"automated agents {(allag[d] - s[d]) / allag[d]:.0%} of all traffic"))
    ds = [d for d in sorted(s) if d not in cand["excluded"]]
    run = [ds[0]] if ds else []
    for a, b in zip(ds, ds[1:]):
        if s[a] > 2 * med and abs(s[b] - s[a]) <= 0.005 * s[a]:
            run.append(b)
        else:
            if len(run) >= 4:
                gaps += [("bot", d, "flat plateau") for d in run]
            run = [b]
    if len(run) >= 4:
        gaps += [("bot", d, "flat plateau") for d in run]
    return gaps


SERIES_REFERENCE = combine_references(ref_complete, ref_plausible, ref_bot)


def _series_extract(src):
    return {"article": src["article"], "series": {str(d): v for d, v in src["series"]},
            "excluded": {}, "repairs": []}


def _series_repair(src, cand, gaps):
    """missing -> re-fetch those dates; bot days -> exclude if a small minority and
    not the series' own peak; anything else -> None (refusal)."""
    new = {**cand, "series": dict(cand["series"]), "excluded": dict(cand["excluded"]),
           "repairs": list(cand["repairs"])}
    missing = sorted(g[1] for g in gaps if g[0] == "missing")
    bots = sorted({g[1]: g[2] for g in gaps if g[0] == "bot"}.items())
    if any(g[0] == "implausible" for g in gaps):
        return None
    if missing:
        refetch = src.get("refetch")
        got = refetch(src["article"], missing[0], missing[-1]) if refetch else {}
        filled = [d for d in missing if d in got]
        for d in filled:
            new["series"][d] = got[d]
        if not filled:
            return None
        new["repairs"].append(f"re-fetched {len(filled)} missing day(s)")
    if bots:
        peak = max(cand["series"], key=cand["series"].get)
        if len(bots) > 0.1 * len(cand["series"]) or any(d == peak for d, _ in bots):
            return None
        for d, why in bots:
            new["excluded"][d] = why
            new["series"].pop(d, None)
        new["repairs"].append(f"excluded {len(bots)} bot-like day(s)")
    return new


def _series_sig(c):
    return (tuple(sorted(c["series"].items())), tuple(sorted(c["excluded"])))


def _verdict(cand, converged, residual, what):
    if not converged:
        return {"status": "refused", "gaps": residual, "output": cand,
                "reason": f"{what}: " + "; ".join(sorted({" ".join(str(x) for x in g) for g in residual})[:4])}
    if cand.get("repairs"):
        return {"status": "corrected", "reason": "; ".join(cand["repairs"]), "gaps": [], "output": cand}
    return {"status": "accepted", "reason": "all checks passed", "gaps": [], "output": cand}


def _source(article, series, aux=None, refetch=None, expected=None):
    ds = sorted(d for d, _ in series)
    src = {"article": article, "series": series, "start": ds[0] if ds else "", "end": ds[-1] if ds else "",
           "aux": aux, "refetch": refetch}
    if expected:
        src["start"], src["end"] = expected
    return src


def verify_series(article, series, *, aux=None, expected=None, refetch=None):
    """series [[day, views]] (user agent); aux {'all_agents': {...}, 'desktop': {...}}."""
    src = _source(article, series, aux, refetch, expected)
    cand, _initial, _hist, ok = feedback_loop(src, extract=_series_extract, reference=SERIES_REFERENCE,
                                              repair=_series_repair, signature=_series_sig, max_iters=3,
                                              label=article)
    return _verdict(cand, ok, SERIES_REFERENCE(src, cand), "series")


# ---- claims: views(t) = A * 2**(-t/h) fitted after the recent peak ----
def _eval_expr(expr, params, t):
    rhs = expr.split("=", 1)[1].replace("^", "**")
    return eval(rhs, {"__builtins__": {}}, {**params, "t": t, "exp": math.exp, "log": math.log})


def _r2(y, yhat):
    m = sum(y) / len(y)
    ss = sum((a - m) ** 2 for a in y)
    return 1 - sum((a - b) ** 2 for a, b in zip(y, yhat)) / ss if ss > 0 else float("nan")


def fit_decay_claim(article, series, lookback=21):
    """Deterministic generator: OLS of log(views) on t after the recent peak.
    Like many quick pipelines it reports the LOG-space R^2; the gate checks it."""
    s = dict((str(d), v) for d, v in series)
    ds = sorted(s)[-lookback:]
    t0 = max(ds, key=lambda d: s[d])
    post = [d for d in ds if d >= t0]
    y = [math.log(max(s[d], 1)) for d in post]
    t = list(range(len(post)))
    if len(t) < 2:
        slope, icpt, r2 = 0.0, (y[0] if y else 0.0), float("nan")
    else:
        mt, my = sum(t) / len(t), sum(y) / len(y)
        stt = sum((a - mt) ** 2 for a in t)
        slope = sum((a - mt) * (b - my) for a, b in zip(t, y)) / stt
        icpt = my - slope * mt
        r2 = _r2(y, [icpt + slope * a for a in t])
    h = (-math.log(2) / slope) if slope < 0 else float("inf")
    return {"article": article, "t0": t0, "expression": "views(t) = A * 2**(-t/h)",
            "params": {"A": round(math.exp(icpt)), "h": round(h, 3)},
            "r2": round(r2, 3), "half_life": round(h, 2), "n": len(post)}


def ref_claim(src, cand):
    """Exact controller: recompute every number the claim states from the data."""
    claim, s = cand["claim"], cand["series"]
    post = sorted(d for d in s if d >= claim["t0"])
    if len(post) < 5:
        return [("insufficient_data", len(post), claim["t0"])]
    gaps = []
    if any(s[d] > s[claim["t0"]] * 1.05 for d in post[1:]):
        gaps.append(("not_decaying",))
    y = [s[d] for d in post]
    yhat = [_eval_expr(claim["expression"], claim["params"], k) for k in range(len(post))]
    r2 = _r2(y, yhat)
    if not (r2 >= 0.5):
        gaps.append(("poor_fit", round(r2, 2)))
    if abs(claim["r2"] - r2) > 0.02:
        gaps.append(("r2_mismatch", claim["r2"], round(r2, 3)))
    h = claim["params"].get("h")
    if h is None or abs(claim["half_life"] - h) > 0.05 * abs(h):
        gaps.append(("half_life_mismatch", claim["half_life"], h))
    return gaps


UNFIXABLE = {"insufficient_data", "not_decaying", "poor_fit"}


def _claim_repair(src, cand, gaps):
    if any(g[0] in UNFIXABLE for g in gaps):
        return None
    claim, s = dict(cand["claim"]), cand["series"]
    post = sorted(d for d in s if d >= claim["t0"])
    yhat = [_eval_expr(claim["expression"], claim["params"], k) for k in range(len(post))]
    fixes = []
    for g in gaps:
        if g[0] == "r2_mismatch":
            claim["r2"] = round(_r2([s[d] for d in post], yhat), 3)
            fixes.append(f"R² {g[1]:.2f} -> {g[2]:.2f}")
        if g[0] == "half_life_mismatch":
            claim["half_life"] = claim["params"]["h"]
            fixes.append("half-life")
    return {**cand, "claim": claim, "repairs": cand["repairs"] + fixes}


def verify_claim(claim, series, *, aux=None, refetch=None):
    """Cascade: stage 1 gates the series, stage 2 gates the claim against the
    verified series. A refused series halts the cascade (claim refused too)."""
    s1 = loop_stage(extract=_series_extract, reference=SERIES_REFERENCE, repair=_series_repair,
                    signature=_series_sig, max_iters=3, label="series")
    s2 = loop_stage(extract=lambda sc: {"claim": dict(claim), "series": sc["series"], "repairs": []},
                    reference=ref_claim, repair=_claim_repair,
                    signature=lambda c: json.dumps(c["claim"], sort_keys=True), max_iters=3, label="claim")
    src = _source(claim["article"], series, aux, refetch)
    final, ok, trace = cascade(s1, s2)(src)
    if len(trace) == 1:
        v = _verdict(final, False, SERIES_REFERENCE(src, final), "input series")
        return {**v, "stage": "series", "output": claim}
    v = _verdict(final, ok, ref_claim(None, final), "claim")
    return {**v, "stage": "claim", "output": final.get("claim", claim)}
