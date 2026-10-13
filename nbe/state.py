"""Persistent analysis state, small enough for the nightly job to commit back.

    state/prior_events.json   empirical spike prior: provenance, processed articles, events
    state/forecasts/<day>.json forecasts issued by the run for <day> (for <day>+1), graded next run
    state/scorecard.json      graded forecasts (rolling 90 days)
    state/day_clusters.json   story clusters per day (rolling 90 days), for lineage
    state/runs.json           last 30 run summaries (timings, counts, warnings)
"""
from __future__ import annotations

import json
from pathlib import Path

KEEP_DAYS = 90


class State:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, name):
        return self.root / name

    def load(self, name, default=None):
        p = self._p(name)
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))

    def save(self, name, obj, indent=None):
        p = self._p(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=indent, ensure_ascii=False, separators=None if indent else (",", ":")),
                       encoding="utf-8")
        tmp.replace(p)

    # forecasts ---------------------------------------------------------------
    def forecasts_for(self, day):
        return self.load(f"forecasts/{day}.json")

    def save_forecasts(self, day, rows, provenance):
        self.save(f"forecasts/{day}.json", {"issued_by_run": day, "provenance": provenance, "forecasts": rows},
                  indent=1)
        files = sorted((self.root / "forecasts").glob("*.json"))
        for f in files[:-KEEP_DAYS]:
            f.unlink()

    # scorecard ----------------------------------------------------------------
    def add_graded(self, rows):
        sc = self.load("scorecard.json", {"graded": []})
        keys = {(r["issued"], r["article"]) for r in rows}
        sc["graded"] = [r for r in sc["graded"] if (r["issued"], r["article"]) not in keys] + rows
        days = sorted({r["issued"] for r in sc["graded"]})[-KEEP_DAYS:]
        sc["graded"] = [r for r in sc["graded"] if r["issued"] in set(days)]
        self.save("scorecard.json", sc, indent=1)
        return sc

    # day clusters --------------------------------------------------------------
    def day_clusters(self):
        return self.load("day_clusters.json", {})

    def save_day_clusters(self, dc):
        keep = sorted(dc)[-KEEP_DAYS:]
        self.save("day_clusters.json", {d: dc[d] for d in keep}, indent=None)

    # runs ------------------------------------------------------------------------
    def log_run(self, summary):
        runs = self.load("runs.json", [])
        runs = [r for r in runs if r.get("date") != summary.get("date")] + [summary]
        self.save("runs.json", runs[-30:], indent=1)
