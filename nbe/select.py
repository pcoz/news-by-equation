"""Which pages count as today's news stories."""
from __future__ import annotations

import re

from .stages.petra_lfc import EVERGREEN, PERENNIAL

# Pages that are always "trending" but aren't news, plus list-of-deaths style pages.
JUNK = re.compile(
    r"^(Main_Page|Special:|Wikipedia:|Portal:|File:|Help:|Talk:|User:|Template:|Category:|Draft:|"
    r"Deaths_in_|List_of_|Lists_of_|.*_deaths_in_|.*\.(com|net|org|xxx|tv)$|XXX|Pornhub|XVideos|XHamster|"
    r"Cleopatra|Undefined|-$|Bible|YouTube|Google|Facebook|Instagram|ChatGPT|Wikipedia$|\d{4}$)",
    re.I,
)

# Titles we never spotlight in a "hi" line (neutral treatment of deaths and crimes).
SENSITIVE = re.compile(r"(shooting|murder|killing|massacre|bombing|attack|rape|abuse|death|assassinat|"
                       r"execution|suicide|crash|disaster|stabbing|kidnap)", re.I)


def clean_list(raw, n=100):
    """Junk-filtered top-n of one day's raw list -> (rows, n_junk_skipped)."""
    out, junk = [], 0
    for a in raw:
        if JUNK.match(a["article"]):
            junk += 1
            continue
        out.append(a)
        if len(out) >= n:
            break
    return out, junk


def presence(top_lists, upto):
    """Share of days (<= upto) each article is in the cleaned top 100."""
    days = [d for d in top_lists if d <= upto]
    cnt = {}
    for d in days:
        for a in top_lists[d]:
            cnt[a["article"]] = cnt.get(a["article"], 0) + 1
    return {a: c / max(1, len(days)) for a, c in cnt.items()}


def select_stories(top_lists, day, n=24, pres=None):
    """First n of the day's cleaned list that are neither perennial nor evergreen.
    -> (stories, {"perennial": [...], "evergreen": [...]}) (skips above the cut)."""
    pres = pres or presence(top_lists, day)
    out, skipped = [], {"perennial": [], "evergreen": []}
    for a in top_lists.get(day, []):
        t = a["article"]
        if EVERGREEN.match(t):
            skipped["evergreen"].append(t)
        elif pres.get(t, 0) > PERENNIAL:
            skipped["perennial"].append(t)
        else:
            out.append(t)
        if len(out) >= n:
            break
    return out, skipped
