"""Stage: TNN change-points + PPF attention curves + empirical spike prior + forecasts.

    detect_change_points(series)      one self-tuning TNN leaky neuron over log(1+views)
    fit_attention_curve(series, cp)   PPF symbolic regression "from the spike onward"
    classify_shape(fit)               spike-and-decay / step change / growth / ...
    Prior                             Kaplan-Meier half-lives + next-day ratios of past spikes
    analyse(series, prior)            all of the above for one article
    backtest_last_day(series, prior)  refit as of the day before, forecast the last day

`series` is [[YYYYMMDD, views], ...] (user agent, all access), oldest first,
ending on the run date.
"""
from __future__ import annotations

import datetime as dt
import math
import warnings

import numpy as np
from ppf import BinaryOp, MacroOp, NodeType, PrimitiveSet, SymbolicRegressor, UnaryOp
from tnn import NeuronType, TemporalNeuron

from ..util import fmt_views, parse_ymd

warnings.filterwarnings("ignore", message="Unknown solver options")
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")   # GP explores overflowing candidates; they score -inf anyway

# --- TNN detector settings (empirical, daily data) ---------------------------
# fast_steps=1 (a one-day jump IS the event), slow_steps=10 (recent normal),
# sigma_steps=25 (noise window + warm-up). z is measured against the neuron's
# pre-update sigma, and the variance update is clipped (sigma_clip), so one
# event does not mask the next.
TNN_PARAMS = dict(tau=5.0, fast_steps=1, slow_steps=10, sigma_steps=25,
                  z_thresh=3.0, z_exit=1.5, caught_thresh=0.2, sigma_clip=3.0)
TNN_TAU_MIN = 1.0
FRESH_DAYS = 2          # change-point within the last 2 days -> "fresh spike"
FRESH_MULTIPLE = 3.0    # ...and at least 3x the pre-spike baseline


def unpack(series):
    dates = [d for d, _ in series]
    views = np.array([v for _, v in series], dtype=float)
    return dates, views


def _run_neuron(x, sigma_steps):
    nu = TemporalNeuron(NeuronType.LEAKY_INTEGRATOR, params=dict(TNN_PARAMS, sigma_steps=sigma_steps))
    nu.enable_self_tuning(tau_min=TNN_TAU_MIN)
    out = []
    for xi in x:
        nu.step(dt=1.0, inputs=float(xi))
        out.append((float(nu.surprise), float(nu.zscore), float(nu.noise_scale),
                    float(nu.tau_eff), float(nu._m_fast - nu._m_slow)))
    return np.array(out)   # surprise, z, sigma, tau_eff, signed band-pass


def detect_change_points(series, edits: dict | None = None) -> dict:
    """Change-points = days where the neuron's own significance z crosses
    z_thresh upward (after warm-up), plus escalations (z jumps to >= 1.5x its
    previous value while already above threshold). Magnitude = z in sigma units
    against the neuron's online noise floor. With `edits` ({day: n}) a second
    neuron runs on log(1+edits/day) and its z on each change-point day is kept."""
    dates, views = unpack(series)
    x = np.log1p(views)
    warm = min(TNN_PARAMS["sigma_steps"], max(10, len(x) // 2))
    tr = _run_neuron(x, warm)
    zt = TNN_PARAMS["z_thresh"]
    etr = None
    if edits is not None:
        etr = _run_neuron(np.log1p(np.array([edits.get(d, 0) for d in dates], dtype=float)), warm)
    cps = []
    for k in range(warm, len(x)):
        z_prev = tr[k - 1, 1] if k > warm else 0.0
        crossing = tr[k, 1] > zt and z_prev <= zt
        escalation = tr[k, 1] > zt and tr[k, 1] >= 1.5 * max(z_prev, zt) and tr[k, 4] > 0 and not crossing
        if crossing or escalation:
            base = float(np.median(views[max(0, k - 14):k]))
            cp = dict(index=k, date=dates[k], z=round(float(tr[k, 1]), 1),
                      direction="up" if tr[k, 4] > 0 else "down", escalation=bool(escalation),
                      baseline=base, multiple=float(views[k] / max(base, 1.0)))
            if etr is not None:
                cp["edit_z"] = round(float(etr[k, 1]), 1)
            cps.append(cp)
    peak = int(np.argmax(views))
    ups = [c for c in cps if c["direction"] == "up"]
    fresh = [c for c in ups if c["index"] >= len(x) - FRESH_DAYS and c["multiple"] >= FRESH_MULTIPLE]
    primary = fresh[0] if fresh else None
    if primary is None:
        for c in ups:
            if c["index"] <= peak:
                primary = c
        if primary is None and ups:
            primary = max(ups, key=lambda c: c["z"])
    return dict(change_points=cps, primary=primary, n_days=len(dates), warmup_days=warm)


def is_fresh(det) -> bool:
    cp = det["primary"]
    return cp is not None and cp["index"] >= det["n_days"] - FRESH_DAYS and cp["multiple"] >= FRESH_MULTIPLE


# --- PPF ---------------------------------------------------------------------
MAIN_PRIMS = PrimitiveSet(
    unary_ops=[UnaryOp.EXP, UnaryOp.NEG],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL],
    macro_ops=[MacroOp.EXP_DECAY, MacroOp.RC_CHARGE, MacroOp.TANH_STEP, MacroOp.POWER_LAW],
    macro_probability=0.5,
)
SEED = 0
MIN_SEGMENT = 5      # PPF needs >= 5 points
PRE_DAYS = 14
R2_SLACK = 0.02      # prefer the simplest Pareto member within this R^2 of the knee


def _r2(y, yhat):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else 0.0


class Fit(dict):
    """Fit results + to_views(x) mapping the formula back to daily views."""

    def to_views(self, x):
        z = self["_expr"].evaluate(np.asarray(x, dtype=float)) * self["scale"]
        return np.expm1(z) if self["space"] == "log" else z


def _half_life(fit, x1, x_last):
    """Days for the fitted EXCESS over the long-run level to halve, from the fit
    start. Works for any fitted form."""
    grid = np.arange(x1, x_last + 120, 0.05)
    f = fit.to_views(grid)
    level = max(float(f[-1]), 0.0)
    f0 = float(f[0])
    if not (f0 > fit.to_views([x_last])[0] and f0 > level * 1.05):
        return None, level
    idx = np.where(f - level <= 0.5 * (f0 - level))[0]
    return (float(grid[idx[0]] - x1) if len(idx) else None), level


def fit_attention_curve(series, cp: dict | None = None, space: str = "lin") -> Fit:
    """Fit a readable formula to the attention curve from the spike onward.

    Fit start = the views peak at/after the change-point if >= MIN_SEGMENT days
    remain, else the change-point day, else a [cp-14, end] window. No change-
    point: from the global peak if pronounced (>= 3x median), else the whole
    window. x = 1 on the fit-start day. space="lin" fits views/scale (the
    prototype backtest favoured it: median APE 30% vs 35% in log space).
    """
    dates, views = unpack(series)
    N = len(views)
    cpi = cp["index"] if cp else None
    anchor = cpi if cpi is not None else 0
    peak = anchor + int(np.argmax(views[anchor:]))
    pronounced = cpi is not None or views[peak] >= 3 * np.median(views)
    if pronounced and N - peak >= MIN_SEGMENT:
        segment, lo, origin = "from peak", peak, peak
    elif cpi is not None and N - cpi >= MIN_SEGMENT:
        segment, lo, origin = "from change-point", cpi, cpi
    elif cpi is not None:
        segment, lo, origin = "pre+post window", max(0, cpi - PRE_DAYS), cpi
    else:
        segment, lo, origin = "whole window", 0, 0
    base_ref = cpi if cpi is not None else origin
    y = views[lo:]
    x = np.arange(lo, N, dtype=float) - origin + 1.0
    yt = np.log1p(y) if space == "log" else y
    scale = float(yt.max()) or 1.0
    yn = yt / scale
    reg = SymbolicRegressor(population_size=300, generations=30, max_depth=4,
                            parsimony_coefficient=0.002, random_state=SEED)
    res = reg.discover(x, yn, primitives=MAIN_PRIMS)
    knee = res.best_tradeoff
    near = [r for r in res.pareto_front if r.r_squared >= knee.r_squared - R2_SLACK]
    best = min(near, key=lambda r: (r.complexity, -r.r_squared)) if near else knee
    expr = best.expression.copy()
    fit = Fit(space=space, scale=scale, _expr=expr, expression=expr.to_string(), segment=segment,
              x1_date=dates[origin], x=x.tolist(), x_last=float(x[-1]), y=y.tolist(),
              n_post_change=(N - cpi) if cpi is not None else None,
              pre_baseline=float(np.median(views[max(0, base_ref - PRE_DAYS):base_ref])) if base_ref > 0 else None)
    yhat = fit.to_views(x)
    fit["yhat"] = yhat.tolist()
    fit["r2"] = _r2(yn, expr.evaluate(x))
    fit["r2_raw"] = _r2(y, yhat)
    fit["r2_log"] = _r2(np.log1p(y), np.log1p(np.clip(yhat, 0, None)))
    hl, level = _half_life(fit, 1.0, float(x[-1])) if segment != "pre+post window" else (None, None)
    fit["half_life_days"], fit["long_run_level"] = hl, level
    return fit


def classify_shape(fit, fresh: bool = False) -> str:
    if fresh:
        return "fresh spike"
    npc = fit.get("n_post_change")
    if (npc is not None and npc < 3) or fit["r2"] < 0.5:
        return "no clean shape"
    x = np.asarray(fit["x"])
    f = np.asarray(fit["yhat"])[x >= 1]
    y = np.asarray(fit["y"])[x >= 1]
    n = len(f)
    imax = int(np.argmax(f))
    fmax = max(f.max(), 1e-9)
    pre = fit.get("pre_baseline")
    if imax <= 0.4 * (n - 1) and f[-1] < 0.6 * fmax:
        return "spike-and-decay"
    jumped = pre is not None and np.median(y) > 2 * pre
    plateau = n >= 3 and abs(f[-1] - f[-3]) / fmax < 0.05
    if f[-1] >= 1.3 * max(f[0], 1e-9) and imax >= 0.7 * (n - 1) and not plateau:
        return "sustained growth"
    if jumped and (plateau or (f.max() - f.min()) / fmax < 0.5):
        return "step change"
    return "no clean shape"


TAIL_MAX = 0.25     # a fit must track the last 3 days within 25% (median) to issue a forecast


def tail_error(fit) -> float:
    """Median relative error of the fitted curve on the last 3 observed days."""
    p, y = fit["yhat"][-3:], fit["y"][-3:]
    return float(np.median([abs(a - b) / max(b, 1.0) for a, b in zip(p, y)]))


def forecast_next_day(fit) -> float | None:
    if fit is None or (fit.get("n_post_change") or 99) < 3:
        return None
    return max(float(fit.to_views([fit["x_last"] + 1.0])[0]), 0.0)


# --- compact formula rendering ------------------------------------------------
def _num(v):
    """Round a view-scale constant: 98k, 4.1k, 860."""
    return fmt_views(abs(v))


def _rate(k):
    return f"{k:.2f}".rstrip("0").rstrip(".") if k >= 0.1 else f"{k:.2g}"


def _linear_terms(node):
    """Decompose expr into (macro, mult, add) with expr = mult*macro + add, or None."""
    if node.node_type == NodeType.MACRO:
        return node, 1.0, 0.0
    if node.node_type == NodeType.CONSTANT:
        return None, 0.0, float(node.value)
    if node.node_type == NodeType.BINARY_OP and node.binary_op in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL):
        L, R = _linear_terms(node.left), _linear_terms(node.right)
        if L is None or R is None:
            return None
        if node.binary_op in (BinaryOp.ADD, BinaryOp.SUB):
            sg = 1.0 if node.binary_op == BinaryOp.ADD else -1.0
            if L[0] is not None and R[0] is not None:
                return None
            m = L[0] or R[0]
            mult = L[1] if L[0] is not None else sg * R[1]
            return m, mult, L[2] + sg * R[2]
        if L[0] is None and R[0] is None:
            return None, 0.0, L[2] * R[2]
        if L[0] is not None and R[0] is not None:
            return None
        c, m = (L, R) if L[0] is None else (R, L)
        return m[0], m[1] * c[2], m[2] * c[2]
    return None


def render_formula(fit, maxlen: int = 40) -> str | None:
    """Compact human formula in days-since-fit-start t (= x - 1), lin space only:
    'v(t) ≈ 98k·e^(−0.29t) + 4k'. None if the fitted form has no compact rendering."""
    if fit is None or fit["space"] != "lin":
        return None
    dec = _linear_terms(fit["_expr"])
    if dec is None or dec[0] is None:
        return None
    m, mult, add = dec
    s = fit["scale"] * mult
    p = list(m.macro_params)
    if m.macro_op == MacroOp.EXP_DECAY:
        a, k, c = p[0] * s, abs(p[1]), p[2] * s + add * fit["scale"]
        amp, const = a * math.exp(-k), c
        if amp <= 0:
            return None
        body = f"{_num(amp)}·e^(−{_rate(k)}t)"
    elif m.macro_op == MacroOp.RC_CHARGE:   # a(1-e^-kx)+c = (a+c) - a e^-k e^-kt
        a, k, c = p[0] * s, abs(p[1]), p[2] * s + add * fit["scale"]
        amp, const = -a * math.exp(-k), a + c
        if amp <= 0:
            return None
        body = f"{_num(amp)}·e^(−{_rate(k)}t)"
    elif m.macro_op == MacroOp.POWER_LAW:
        a, b, c = p[0] * s, p[1], p[2] * s + add * fit["scale"]
        if a <= 0:
            return None
        amp, const = a, c
        body = f"{_num(a)}·(t+1)^{'−' if b < 0 else ''}{_rate(abs(b))}"
    else:
        return None
    peak = max(fit["y"]) if fit.get("y") else None
    if peak and (amp > 1.5 * peak or abs(const) > peak):   # cancelling giants: not a readable law
        return None
    out = f"v(t) ≈ {body}"
    if abs(const) >= 0.5:
        out += f" {'+' if const > 0 else '−'} {_num(const)}"
    return out if len(out) <= maxlen else None


# --- empirical prior from past spikes ---------------------------------------
BUCKETS = [(3, 10), (10, 40), (40, 200), (200, 1e9)]


def bucket_of(m):
    for lo, hi in BUCKETS:
        if lo <= m < hi:
            return (lo, hi)
    return None


def _km_quantiles(t, observed, qs=(0.25, 0.5, 0.75)):
    """Kaplan-Meier quantiles of time-to-event with right censoring."""
    t = np.asarray(t, float)
    obs = np.asarray(observed, bool)
    order = np.argsort(t)
    t, obs = t[order], obs[order]
    s, out, n = 1.0, {}, len(t)
    for i, ti in enumerate(t):
        if obs[i]:
            s *= 1 - 1 / (n - i)
            for q in qs:
                if q not in out and s <= 1 - q:
                    out[q] = float(ti)
    return [out.get(q) for q in qs]


class Prior:
    def __init__(self, events):
        self.events = events

    def bucket_stats(self, m):
        b = bucket_of(m)
        ev = [e for e in self.events if bucket_of(e["multiple"]) == b]
        if not ev:
            return None
        q = _km_quantiles([e["t_half"] for e in ev], [e["half_observed"] for e in ev])
        return dict(bucket=b, n=len(ev), n_censored=sum(not e["half_observed"] for e in ev), half_life_q=q)

    def ratio(self, m, j):
        """Median and IQR of v[cp+j+1]/v[cp+j] for past spikes of similar multiple."""
        b = bucket_of(m)
        r = [e["ratios"][j] for e in self.events if bucket_of(e["multiple"]) == b and len(e["ratios"]) > j]
        if len(r) < 5:
            return None
        return float(np.median(r)), float(np.percentile(r, 25)), float(np.percentile(r, 75)), len(r)

    def forecast(self, series, cp):
        _, views = unpack(series)
        j = len(views) - 1 - cp["index"]
        r = self.ratio(cp["multiple"], j)
        if r is None:
            return None
        return dict(value=float(views[-1] * r[0]), lo=float(views[-1] * r[1]), hi=float(views[-1] * r[2]),
                    n=r[3], j=j)


def decay_window_end(e):
    """Last day whose views an event's stored numbers depend on: the day the excess
    fell below half (t_half bracket) and the 3 next-day ratios after the change-point.
    None for right-censored events (their t_half depends on where the series ended)."""
    if not e.get("half_observed"):
        return None
    cp = parse_ymd(e["date"])
    end = max(cp + dt.timedelta(days=e["rise"] + math.floor(e["t_half"]) + 1), cp + dt.timedelta(days=3))
    return end.strftime("%Y%m%d")


def prior_as_of(events, day):
    """Events fully resolved by `day` (decay window ended on or before it)."""
    out = []
    for e in events:
        w = decay_window_end(e)
        if w is not None and w <= day:
            out.append(e)
    return out


def spike_events(article, series, horizon_end=None):
    """Spike events in one full daily history (used to grow the prior).

    Event = TNN up-change-point >= 3x its 14-day pre-spike median, >= 14 days
    after the previous one. Peak = max within the first 4 days; t_half = days
    from peak until views fall to baseline + half the excess (interpolated in
    log-excess); right-censored at the series end (Kaplan-Meier)."""
    dates, v = unpack(series)
    if len(v) < 60:
        return []
    det = detect_change_points(series)
    events, last_k = [], -999
    for c in det["change_points"]:
        k = c["index"]
        if c["direction"] != "up" or c["multiple"] < 3 or k - last_k < 14 or k < 14:
            continue
        last_k = k
        pk = k + int(np.argmax(v[k:k + 4]))
        b = c["baseline"]
        thr = b + 0.5 * (v[pk] - b)
        below = np.where(v[pk:] <= thr)[0]
        ratios = [float(v[k + j + 1] / max(v[k + j], 1.0)) for j in range(3) if k + j + 1 < len(v)]
        if len(below):
            t1 = int(below[0])
            e0 = v[pk] - b
            ea, eb = max(v[pk + t1 - 1] - b, 1e-9), max(v[pk + t1] - b, 1e-9)
            frac = (math.log(ea) - math.log(0.5 * e0)) / (math.log(ea) - math.log(eb)) if ea > eb else 1.0
            t_half = (t1 - 1) + min(max(frac, 0.0), 1.0)
        else:
            t_half = float(len(v) - 1 - pk)
        events.append(dict(article=article, date=dates[k], multiple=c["multiple"], z=c["z"], peak=float(v[pk]),
                           rise=pk - k, ratios=ratios, t_half=t_half, half_observed=bool(len(below))))
    return events


# --- driver ------------------------------------------------------------------
def analyse(series, prior: Prior | None = None, edits: dict | None = None, backtest: bool = True) -> dict:
    det = detect_change_points(series, edits=edits)
    cp = det["primary"]
    fresh = is_fresh(det)
    fit = None if fresh else fit_attention_curve(series, cp)
    shape = classify_shape(fit, fresh)
    fc, method, pf, guarded = None, None, None, None
    if fresh and prior is not None:
        pf = prior.forecast(series, cp)
        if pf:
            fc, method, guarded = pf["value"], "prior", True
    elif not fresh and shape != "no clean shape":
        # every clean fit forecasts ("all fits"); the tail guard (adopted 2026-10-13, see
        # state/policy.json) marks the ones that also track the last 3 days within TAIL_MAX
        fc = forecast_next_day(fit)
        if fc is not None:
            method, guarded = "formula", tail_error(fit) <= TAIL_MAX
    out = dict(detect=det, fit=fit, fresh=fresh, shape=shape, forecast=fc, method=method, prior_forecast=pf,
               guarded=guarded, naive=float(series[-1][1]))
    out["backtest"] = backtest_last_day(series, prior) if backtest else None
    return out


def backtest_last_day(series, prior=None):
    """Re-run as of the day before, forecast the last day.
    Returns {method, forecast, naive, actual, ape, ape_naive} or {method: reason}."""
    if len(series) < 30:
        return {"method": "short"}
    det = detect_change_points(series)
    if any(c["index"] == len(series) - 1 and c["direction"] == "up" for c in det["change_points"]):
        return {"method": "new"}          # the spike began on the last day: nothing to forecast it from
    r = analyse(series[:-1], prior, backtest=False)
    if r["forecast"] is None:
        return {"method": "none"}
    actual, naive = float(series[-1][1]), float(series[-2][1])
    return {"method": r["method"], "forecast": r["forecast"], "guarded": r["guarded"], "naive": naive,
            "actual": actual,
            "ape": 100.0 * abs(r["forecast"] - actual) / max(actual, 1.0),
            "ape_naive": 100.0 * abs(naive - actual) / max(actual, 1.0)}
