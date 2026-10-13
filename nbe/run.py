"""News by Equation nightly run.

    python -m nbe.run [--date YYYY-MM-DD] [--cache DIR] [--state DIR] [--out output/nbe.json]

Default date = yesterday (UTC). Every stage is timed; a failing stage leaves a
warning (shown in the panel as a `warn` line) and the run continues.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from . import select as sel
from .fetch import FetchError, Fetcher
from .state import State
from .util import shift, ymd

HERE = Path(__file__).resolve().parent
WINDOW_DAYS = 60          # per-article history window
LIST_DAYS = 240           # top lists used for the PETRA life-cycle log
LINEAGE_DAYS = 60         # day clusters kept up to date for lineage
N_STORIES = 24            # stories on screen
DF_TOP = 50               # articles whose link sets define link specificity


class Run:
    def __init__(self, date, cache, state_dir, offline=False, petra_steps=300):
        self.D = date
        self.f = Fetcher(cache, offline=offline)
        self.state = State(state_dir)
        self.petra_steps = petra_steps
        self.R = {"date": date, "warnings": [], "timings": {}}
        self.start = shift(date, -(WINDOW_DAYS - 1))

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def stage(self, name):
        t = time.perf_counter()
        try:
            yield
        except Exception as e:                      # keep going, but say so
            self.R["warnings"].append({"stage": name, "error": f"{type(e).__name__}: {e}"})
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            self.R["timings"][name] = self.R["timings"].get(name, 0.0) + time.perf_counter() - t

    def ok(self, name):
        return not any(w["stage"] == name for w in self.R["warnings"])

    def series(self, a, start=None, end=None, access="all-access", agent="user"):
        return self.f.series(a, start or self.start, end or self.D, access, agent)

    def aux(self, a):
        return {"all_agents": self.f.views(a, self.start, self.D, agent="all-agents"),
                "desktop": self.f.views(a, self.start, self.D, access="desktop")}

    def refetch(self, a, s, e):
        try:
            return self.f.views(a, s, e)
        except FetchError:
            return {}

    # ------------------------------------------------------------------ stages
    def fetch_lists(self):
        R = self.R
        days = [shift(self.D, -k) for k in range(LIST_DAYS - 1, -1, -1)]
        raw = {}
        for d in days:
            try:
                raw[d] = self.f.top(d)
            except FetchError:
                pass
        if not raw.get(self.D):
            raise RuntimeError(f"no top list for {self.D}")
        self.lists = {}
        for d, rows in raw.items():
            clean, junk = sel.clean_list(rows)
            if clean:
                self.lists[d] = clean
            if d == self.D:
                R["junk_today"] = junk
        self.pres = sel.presence(self.lists, self.D)
        self.stories, skipped = sel.select_stories(self.lists, self.D, N_STORIES, self.pres)
        R["skipped"] = skipped
        R["n_list_days"] = len(self.lists)
        R["stories"] = list(self.stories)
        R["views_today"] = {a["article"]: a["views"] for a in self.lists[self.D]}

    def fetch_histories(self):
        from .stages import petra_lfc as PL
        cases, _ = PL.build_cases(self.lists, self.D)
        self.live = sorted({c.name for c, ongoing in cases if ongoing})
        names = sorted(set(self.stories) | set(self.live))
        specs = [(a, self.start, self.D, "all-access", ag) for a in names for ag in ("user", "all-agents")]
        specs += [(a, self.start, self.D, "desktop", "user") for a in names]
        specs += [(a, self.start, self.D, acc, ag) for a in self.stories
                  for acc, ag in (("mobile-web", "user"), ("mobile-app", "user"),
                                  ("all-access", "spider"), ("all-access", "automated"))]
        self.f.prefetch_views(specs)
        self.hist = {}
        for a in names:
            try:
                s = self.series(a)
                if s:
                    self.hist[a] = s
            except FetchError as e:
                self.R["warnings"].append({"stage": "fetch", "error": f"{a}: {e}"})
        self.R["hist_len"] = {a: len(self.hist[a]) for a in self.stories if a in self.hist}
        self.R["hist_tail"] = {a: [int(v) for _d, v in self.hist[a][-21:]] for a in self.stories if a in self.hist}
        self.edits = {}
        for a in self.stories:
            try:
                self.edits[a] = self.f.revisions(a, self.start, self.D)
            except FetchError:
                pass

    def verify(self):
        from .stages import petra_lfc as PL
        self.gate = {}
        for a in sorted(set(self.stories) | set(self.live)):
            if a not in self.hist:
                continue
            try:
                aux = self.aux(a)
            except FetchError:
                aux = None
            # completeness is required from the page's first day with views (new pages start late)
            self.gate[a] = PL.verify_series(a, self.hist[a], aux=aux, expected=(self.hist[a][0][0], self.D),
                                            refetch=self.refetch)
        g = {a: {"status": v["status"], "reason": v["reason"], "gaps": [list(x) for x in v["gaps"]]}
             for a, v in self.gate.items()}
        self.R["gate"] = {a: g[a] for a in self.stories if a in g}
        self.R["gate_live"] = {"n": len(self.live), "refused": sorted(a for a in self.live
                                                                      if g.get(a, {}).get("status") == "refused")}
        # downstream analysis uses the verified series (corrections applied)
        self.clean = {}
        for a, v in self.gate.items():
            if v["status"] != "refused":
                self.clean[a] = sorted([d, n] for d, n in v["output"]["series"].items())

    def prior(self):
        from .stages import tnn_ppf as TP
        st = self.state.load("prior_events.json")
        if st is None:
            st = {"provenance": {"source": "grown nightly by nbe.run", "seeded": False}, "processed": [],
                  "events": []}
        processed = set(st["processed"])
        # grow: stories that entered the list 21..45 days ago (decay observable), peaked >= 100k
        first, peak = {}, {}
        for d in sorted(self.lists):
            for a in self.lists[d]:
                first.setdefault(a["article"], d)
                peak[a["article"]] = max(peak.get(a["article"], 0), a["views"])
        lo, hi = shift(self.D, -45), shift(self.D, -21)
        cands = sorted(a for a in first if lo <= first[a] <= hi and peak[a] >= 100_000
                       and self.pres.get(a, 0) <= 0.4 and a not in processed)[:20]
        added = 0
        for a in cands:
            try:
                s = self.series(a, start=shift(self.D, -250))
            except FetchError:
                continue
            ev = TP.spike_events(a, s)
            have = {(e["article"], e["date"]) for e in st["events"]}
            new = [e for e in ev if (e["article"], e["date"]) not in have]
            st["events"] += new
            added += len(new)
            processed.add(a)
        st["processed"] = sorted(processed)
        if cands:
            st["provenance"]["last_grown"] = self.D
        self.prior_state = st
        # strictly as-of: only events whose decay window ended on or before D-1
        events = TP.prior_as_of(st["events"], shift(self.D, -1))
        self.priorobj = TP.Prior(events)
        self.R["prior"] = {"n_events": len(events), "n_articles": len({e["article"] for e in events}),
                           "n_stored": len(st["events"]), "as_of": shift(self.D, -1),
                           "added_today": added, "candidates": len(cands), "provenance": st["provenance"]}

    def analyse(self):
        from .stages import tnn_ppf as TP
        self.an = {}
        for a in self.stories:
            if a not in self.clean:
                continue
            s = self.clean[a]
            if len(s) < 30:
                self.R["warnings"].append({"stage": "tnn", "error": f"{a}: only {len(s)} days of history"})
                continue
            t = time.perf_counter()
            self.an[a] = TP.analyse(s, self.priorobj, edits=self.edits.get(a))
            self.an[a]["seconds"] = time.perf_counter() - t
        self.R["analysis"] = {a: self._summary(a, r) for a, r in self.an.items()}

    def _summary(self, a, r):
        from .stages import tnn_ppf as TP
        cp, fit = r["detect"]["primary"], r["fit"]
        out = {"shape": r["shape"], "fresh": r["fresh"], "forecast": r["forecast"], "method": r["method"],
               "guarded": r["guarded"],
               "naive": r["naive"], "prior_forecast": r["prior_forecast"], "seconds": round(r["seconds"], 2),
               "cp": cp, "n_cps": len(r["detect"]["change_points"]), "backtest": r["backtest"]}
        if fit is not None:
            out.update(formula=TP.render_formula(fit), expression=fit["expression"], segment=fit["segment"],
                       x1_date=fit["x1_date"], r2=fit["r2"], r2_raw=fit["r2_raw"], r2_log=fit["r2_log"],
                       half_life=fit["half_life_days"], long_run=fit["long_run_level"],
                       peak=max(fit["y"]), n_fit=len(fit["y"]),
                       tail_err=TP.tail_error(fit))
        if r["fresh"] and cp:
            out["bucket"] = self.priorobj.bucket_stats(cp["multiple"])
        return out

    def policy(self):
        """Forecast policy; the tail guard was adopted on 2026-10-13 after seeing the
        2026-10-12 backtest, so grades of forecasts issued before that are post-hoc."""
        pol = self.state.load("policy.json")
        if pol is None:
            pol = {"forecast_guard": {
                "rule": "formula forecast counts as 'guarded' only if the fit tracks the last 3 days "
                        "within 25% median relative error (tnn_ppf.TAIL_MAX)",
                "adopted": "20261013",
                "note": "chosen after seeing the 2026-10-12 backtest (all fits 25% vs repeat-today 20%); "
                        "scores of forecasts issued before the adoption date are post-hoc. Forward "
                        "scores ('since 13 Oct') use only forecasts issued on or after it."}}
            self.state.save("policy.json", pol, indent=1)
        return pol

    def score(self):
        R = self.R
        pol = self.policy()
        adopted = pol["forecast_guard"]["adopted"]
        R["policy"] = pol
        # 1. grade the forecasts the previous run issued for today
        prev = self.state.forecasts_for(shift(self.D, -1))
        graded = []
        if prev:
            for row in prev["forecasts"]:
                a = row["article"]
                try:
                    v = self.f.views(a, self.D, self.D).get(self.D)
                except FetchError:
                    v = None
                if v is None:
                    continue
                graded.append({"issued": prev["issued_by_run"], "target": self.D, "article": a,
                               "method": row["method"], "forecast": row["forecast"], "naive": row["naive"],
                               "guarded": row.get("guarded"), "post_hoc": prev["issued_by_run"] < adopted,
                               "actual": v, "ape": 100 * abs(row["forecast"] - v) / max(v, 1),
                               "ape_naive": 100 * abs(row["naive"] - v) / max(v, 1)})
            sc = self.state.add_graded(graded)
            R["graded"] = {"issued": prev["issued_by_run"], "rows": graded, "adopted": adopted,
                           "forward_rows": [r for r in sc["graded"] if r["issued"] >= adopted],
                           "all_nights": len({r["issued"] for r in sc["graded"]}), "all_rows": sc["graded"]}
        else:
            R["graded"] = {"issued": None, "rows": [], "adopted": adopted,
                           "forward_rows": [r for r in self.state.load("scorecard.json", {"graded": []})["graded"]
                                            if r["issued"] >= adopted]}
        # 2. issue today's forecasts for tomorrow
        rows = []
        for a, r in self.an.items():
            if r["forecast"] is not None:
                pf = r.get("prior_forecast") or {}
                rows.append({"article": a, "method": r["method"], "forecast": round(r["forecast"]),
                             "guarded": r["guarded"],
                             "naive": round(r["naive"]), "lo": pf.get("lo"), "hi": pf.get("hi")})
        self.state.save_forecasts(self.D, rows, {"run_date": self.D, "generator": "nbe.run",
                                                 "prior_events": R["prior"]["n_events"]})
        R["issued"] = rows
        # 3. backtest: refit as of the day before, forecast today
        bt = [dict(r["backtest"], article=a) for a, r in self.an.items()
              if r["backtest"] and "ape" in r["backtest"]]
        R["backtest"] = bt

    def connect(self):
        from .stages import nrf_holant as NH
        R = self.R
        today_top = [a["article"] for a in self.lists[self.D][:DF_TOP]]
        need = sorted(set(self.stories) | set(today_top))
        links = {}
        for a in need:
            try:
                links[a] = self.f.links(a)
            except FetchError as e:
                R["warnings"].append({"stage": "holant", "error": f"links {a}: {e}"})
        arts = [a for a in self.stories if a in links]
        W, D = NH.relatedness(arts, links)
        edges = NH.edges_of(W)
        cls = [c for c in NH.clusters(edges)]
        df = NH.link_df(links[a] for a in today_top if a in links)
        n_docs = sum(1 for a in today_top if a in links)
        corpus = {x["article"] for d in self.lists.values() for x in d}
        lead = {}
        for c in cls:
            for a in c:
                try:
                    lead[a] = self.f.lead_links(a)
                except FetchError:
                    lead[a] = None
        # day clusters for lineage (history from state; missing days computed now)
        dc = self.state.day_clusters()
        dc[self.D] = cls
        days = [shift(self.D, -k) for k in range(1, LINEAGE_DAYS + 1)]
        computed = 0
        for d in days:
            if d in dc or d not in self.lists:
                continue
            st, _ = sel.select_stories(self.lists, d, N_STORIES, self.pres)
            ls = {}
            for a in st:
                try:
                    ls[a] = self.f.links(a)
                except FetchError:
                    pass
            ar = [a for a in st if a in ls]
            dc[d] = NH.clusters(NH.edges_of(NH.relatedness(ar, ls)[0]))
            computed += 1
        self.day_clusters = dc
        out = []
        for c in sorted(cls, key=lambda c: -sum(R["views_today"].get(a, 0) for a in c)):
            ce = [e for e in edges if e[0] in c and e[1] in c]
            ties = []
            for e in sorted(ce, key=lambda e: -W[e]):
                ties.append({"pair": list(e), "w": W[e], "direct": D[e]["direct"], "n_shared": len(D[e]["shared"]),
                             "before": NH.via_names_before(e, D, corpus),
                             "after": [[x, round(s, 2), ld] for x, s, ld in NH.via_names(e, links, lead, df, n_docs)]})
            lab, lab_src = NH.cluster_label(c, ties)
            hist = {a: dict(map(tuple, self.clean[a])) for a in c if a in self.clean}
            out.append({"members": c, "ties": ties, "label": lab, "label_source": lab_src, "bridges": NH.bridges(c, ce),
                        "attention": NH.cluster_attention(c, hist, self.D),
                        "lineage": NH.lineage(c, dc, self.D)})
        pairing = NH.best_pairing(arts, W)
        R["net"] = {"n_articles": len(arts), "n_edges": len(edges), "clusters": out,
                    "pairing": {k: v for k, v in pairing.items() if k != "perfect_matchings"},
                    "links": [[a, b, W[(a, b)]] for a, b in edges], "day_clusters_computed": computed,
                    "lineage_days": sum(1 for d in days if d in dc)}

    def counters(self):
        from .stages import nrf_holant as NH
        ch = {}
        for a in self.stories:
            if a not in self.clean:
                continue
            try:
                ch[a] = {f"{acc}/{ag}": self.f.views(a, self.start, self.D, acc, ag) for acc, ag in
                         (("all-access", "user"), ("desktop", "user"), ("mobile-web", "user"),
                          ("mobile-app", "user"), ("all-access", "all-agents"), ("all-access", "spider"),
                          ("all-access", "automated"))}
            except FetchError:
                pass
        self.R["counters"] = NH.counter_laws(ch, self.D)

    def lifecycle(self):
        from .stages import petra_lfc as PL
        res = PL.lifecycle(self.lists, self.D, refused_today=self.R["gate_live"]["refused"], steps=self.petra_steps)
        self.R["petra"] = res

    def claims(self):
        from .stages import petra_lfc as PL
        out = {}
        for a in self.stories:
            if a not in self.hist:
                continue
            src = self.clean.get(a, self.hist[a])
            c = PL.fit_decay_claim(a, src)
            try:
                aux = self.aux(a)
            except FetchError:
                aux = None
            v = PL.verify_claim(c, self.hist[a], aux=aux, refetch=self.refetch)
            out[a] = {"claim": c, "status": v["status"], "stage": v["stage"], "reason": v["reason"],
                      "gaps": [list(g) for g in v["gaps"]], "final": v["output"]}
        self.R["claims"] = out

    # ------------------------------------------------------------------ main
    def run(self):
        T = time.perf_counter()
        with self.stage("fetch"):
            self.fetch_lists()
            self.fetch_histories()
        if not self.ok("fetch"):
            raise SystemExit("fetch failed; nothing to say")
        with self.stage("verify"):
            self.verify()
        if not self.ok("verify"):
            self.clean = dict(self.hist)
            self.R.setdefault("gate", {})
            self.R.setdefault("gate_live", {"n": 0, "refused": []})
        with self.stage("prior"):
            self.prior()
        if not self.ok("prior"):
            from .stages.tnn_ppf import Prior
            self.priorobj = Prior([])
        with self.stage("tnn+ppf"):
            self.analyse()
        if not self.ok("tnn+ppf"):
            self.an = getattr(self, "an", {})
        with self.stage("score"):
            self.score()
        with self.stage("holant+nrf"):
            self.connect()
        with self.stage("nrf-counters"):
            self.counters()
        with self.stage("petra"):
            self.lifecycle()
        with self.stage("lfc-claims"):
            self.claims()
        self.R["timings"]["total"] = time.perf_counter() - T
        self.R["fetch_stats"] = dict(self.f.stats)
        return self.R

    def persist(self):
        if hasattr(self, "prior_state"):
            self.state.save("prior_events.json", self.prior_state)
        if hasattr(self, "day_clusters"):
            self.state.save_day_clusters(self.day_clusters)
        self.state.log_run({"date": self.D, "timings": {k: round(v, 2) for k, v in self.R["timings"].items()},
                            "warnings": self.R["warnings"], "stories": len(self.stories),
                            "fetch": self.R.get("fetch_stats")})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nbe.run")
    ap.add_argument("--date", default=(dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat())
    ap.add_argument("--cache", default=str(HERE.parent / "cache"))
    ap.add_argument("--state", default=str(HERE.parent / "state"))
    ap.add_argument("--out", default=str(HERE.parent / "output" / "nbe.json"))
    ap.add_argument("--offline", action="store_true", help="cache only; never touch the network")
    ap.add_argument("--petra-steps", type=int, default=300)
    ap.add_argument("--dump", help="also write the raw results (debug JSON) here")
    a = ap.parse_args(argv)
    date = ymd(dt.date.fromisoformat(a.date))
    run = Run(date, a.cache, a.state, offline=a.offline, petra_steps=a.petra_steps)
    R = run.run()
    from .story import compose
    t = time.perf_counter()
    doc = compose(R)
    R["timings"]["story"] = time.perf_counter() - t
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    run.persist()
    if a.dump:
        Path(a.dump).write_text(json.dumps(R, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
    tm = R["timings"]
    print(f"nbe {date}: wrote {out} ({out.stat().st_size // 1024} KB); " +
          ", ".join(f"{k} {v:.1f}s" for k, v in tm.items()) +
          f"; requests {R['fetch_stats']['requests']}, warnings {len(R['warnings'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
