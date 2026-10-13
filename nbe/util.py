"""Small shared helpers: dates, file-safe names, number formatting, titles."""
from __future__ import annotations

import datetime as dt
import re

_SAFE = re.compile(r"[^\w.-]")


def safe(name: str) -> str:
    """File-safe cache key for an article title (same scheme as the prototype cache)."""
    return _SAFE.sub("_", name)


def ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def parse_ymd(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y%m%d").date()


def shift(day: str, days: int) -> str:
    return ymd(parse_ymd(day) + dt.timedelta(days=days))


def day_range(start: str, end: str) -> list[str]:
    a, b = parse_ymd(start), parse_ymd(end)
    return [ymd(a + dt.timedelta(days=k)) for k in range((b - a).days + 1)]


def human_day(day: str) -> str:
    """'20261007' -> '7 Oct'."""
    d = parse_ymd(day)
    return f"{d.day} {d.strftime('%b')}"


def fmt_views(v) -> str:
    """Compact view counts: 1.41M, 98k, 860."""
    if v is None:
        return "-"
    v = float(v)
    if v >= 1e6:
        return f"{v / 1e6:.2f}M" if v < 1e7 else f"{v / 1e6:.0f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k" if v >= 1e4 else f"{v / 1e3:.1f}k"
    return f"{v:.0f}"


def label(title: str, maxlen: int = 22) -> str:
    """Display label: underscores to spaces, disambiguation parentheses dropped."""
    s = re.sub(r"\s*\([^)]*\)\s*$", "", title.replace("_", " ")).strip()
    if len(s) > maxlen:
        s = s[: maxlen - 1].rstrip() + "…"
    return s


def pct(x) -> str:
    return "-" if x is None else f"{x:.0f}%"
