"""Wikimedia / MediaWiki fetching with a polite, retrying client and a file cache.

Cache layout (all JSON, under one configurable directory):
    top/YYYYMMDD.json                         daily top list [{article, views, rank}]
    views/<key>__<access>__<agent>.json       {"days": {YYYYMMDD: n}, "covered": [[start, end], ...]}
    links/<key>.json                          all main-namespace out-links (sorted list)
    lead/<key>.json                           out-links of the lead section only (sorted list)
    revs/<key>.json                           {"days": {YYYYMMDD: edits}, "covered": [[start, end], ...]}

`covered` records which date ranges were actually requested, so a page with no
views on some day (the API omits it) is not re-requested every night. Only the
missing part of a requested range is fetched.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .util import day_range, safe

UA = "NewsByEquation/1.0 (https://sapientronic.ai; info@sapientronic.ai)"
PV = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
MW = "https://en.wikipedia.org/w/api.php"


class FetchError(RuntimeError):
    pass


class _RateLimiter:
    """At most one request per `interval` seconds (shared across threads)."""

    def __init__(self, interval: float):
        self.interval, self._next, self._lock = interval, 0.0, threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


def _merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _uncovered(covered, start, end):
    want = day_range(start, end)
    return [d for d in want if not any(s <= d <= e for s, e in covered)]


class Fetcher:
    def __init__(self, cache_dir, offline: bool = False, workers: int = 4):
        self.root = Path(cache_dir)
        self.offline = offline
        self.workers = workers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept": "application/json"})
        self._limits = {"wikimedia.org": _RateLimiter(0.05), "en.wikipedia.org": _RateLimiter(0.15)}
        self._io = threading.Lock()
        self.stats = {"requests": 0, "cache_hits": 0, "errors": 0}

    # ------------------------------------------------------------------ http
    def _get(self, url, params=None):
        """JSON body, None on 404. Retries 429/5xx/network errors with backoff."""
        if self.offline:
            raise FetchError(f"offline: {url}")
        host = urllib.parse.urlparse(url).hostname or ""
        limiter = self._limits.get(host, _RateLimiter(0.2))
        last = None
        for attempt in range(6):
            limiter.wait()
            try:
                r = self.session.get(url, params=params, timeout=30)
                self.stats["requests"] += 1
            except requests.RequestException as e:
                last = repr(e)
                time.sleep(min(30, 2 ** attempt))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            last = f"HTTP {r.status_code}"
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else min(30, 2 ** attempt))
                continue
            break
        self.stats["errors"] += 1
        raise FetchError(f"{url}: {last}")

    def _read(self, path: Path):
        if path.exists():
            self.stats["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _write(self, path: Path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------ top lists
    def top(self, day: str) -> list[dict]:
        """Raw daily top list (first 300 entries) for en.wikipedia, all-access."""
        f = self.root / "top" / f"{day}.json"
        got = self._read(f)
        if got is not None:
            return got
        d = self._get(f"{PV}/top/en.wikipedia/all-access/{day[:4]}/{day[4:6]}/{day[6:]}")
        arts = d["items"][0]["articles"][:300] if d else []
        rows = [{"article": a["article"], "views": a["views"], "rank": a["rank"]} for a in arts]
        if rows:                        # do not cache "not published yet"
            self._write(f, rows)
        return rows

    # ------------------------------------------------------------------ pageviews
    def _views_path(self, article, access, agent):
        return self.root / "views" / f"{safe(article)}__{access}__{agent}.json"

    def views(self, article: str, start: str, end: str, access="all-access", agent="user") -> dict:
        """{YYYYMMDD: views} for [start, end] (days without views are absent)."""
        f = self._views_path(article, access, agent)
        with self._io:
            rec = self._read(f) or {"days": {}, "covered": []}
        miss = _uncovered(rec["covered"], start, end)
        if miss:
            a = urllib.parse.quote(article, safe="")
            d = self._get(f"{PV}/per-article/en.wikipedia/{access}/{agent}/{a}/daily/{miss[0]}00/{miss[-1]}00")
            for i in (d or {}).get("items", []):
                rec["days"][i["timestamp"][:8]] = i["views"]
            rec["covered"] = _merge(rec["covered"] + [[miss[0], miss[-1]]])
            with self._io:
                self._write(f, rec)
        return {k: v for k, v in rec["days"].items() if start <= k <= end}

    def series(self, article, start, end, access="all-access", agent="user") -> list:
        """[[YYYYMMDD, views], ...] sorted, for the stage code."""
        return sorted([d, v] for d, v in self.views(article, start, end, access, agent).items())

    def prefetch_views(self, specs):
        """specs: iterable of (article, start, end, access, agent). Fetch concurrently,
        swallowing errors (the caller sees them again on the real call)."""
        def one(s):
            try:
                self.views(*s)
                return True
            except FetchError:
                return False
        specs = list(specs)
        if not specs:
            return 0
        with ThreadPoolExecutor(self.workers) as ex:
            return sum(ex.map(one, specs))

    # ------------------------------------------------------------------ MediaWiki
    def links(self, title: str) -> set[str]:
        """All main-namespace titles linked from `title` (follows continuation)."""
        f = self.root / "links" / f"{safe(title)}.json"
        got = self._read(f)
        if got is not None:
            return set(got)
        links, cont = set(), {}
        while True:
            p = {"action": "query", "prop": "links", "titles": title.replace("_", " "), "pllimit": "max",
                 "plnamespace": 0, "format": "json", "redirects": 1, **cont}
            d = self._get(MW, p) or {}
            for page in d.get("query", {}).get("pages", {}).values():
                links.update(l["title"].replace(" ", "_") for l in page.get("links", []))
            if "continue" not in d:
                break
            cont = d["continue"]
        self._write(f, sorted(links))
        return links

    def lead_links(self, title: str) -> list[str]:
        """Main-namespace links in the article's lead section (section 0), existing pages only."""
        f = self.root / "lead" / f"{safe(title)}.json"
        got = self._read(f)
        if got is not None:
            return got
        d = self._get(MW, {"action": "parse", "page": title.replace("_", " "), "section": 0, "prop": "links",
                           "redirects": 1, "format": "json", "formatversion": 2}) or {}
        out = sorted({l["title"].replace(" ", "_") for l in d.get("parse", {}).get("links", [])
                      if l.get("ns") == 0 and l.get("exists", True)})
        self._write(f, out)
        return out

    def revisions(self, title: str, start: str, end: str) -> dict:
        """{YYYYMMDD: number of edits} to the article in [start, end]."""
        f = self.root / "revs" / f"{safe(title)}.json"
        rec = self._read(f) or {"days": {}, "covered": []}
        miss = _uncovered(rec["covered"], start, end)
        if miss:
            s, e = miss[0], miss[-1]
            params = dict(action="query", prop="revisions", titles=title.replace("_", " "),
                          rvprop="timestamp", rvlimit="max", format="json", formatversion="2",
                          rvstart=f"{e[:4]}-{e[4:6]}-{e[6:]}T23:59:59Z", rvend=f"{s[:4]}-{s[4:6]}-{s[6:]}T00:00:00Z")
            rec["days"] = {k: v for k, v in rec["days"].items() if not s <= k <= e}   # recount the span
            while True:
                d = self._get(MW, params) or {}
                for p in d.get("query", {}).get("pages", []):
                    for rv in p.get("revisions", []):
                        k = rv["timestamp"][:10].replace("-", "")
                        rec["days"][k] = rec["days"].get(k, 0) + 1
                if "continue" not in d:
                    break
                params.update(d["continue"])
            rec["covered"] = _merge(rec["covered"] + [[s, e]])
            self._write(f, rec)
        return {k: v for k, v in rec["days"].items() if start <= k <= end}
