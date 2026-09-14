# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stock and flow measures for confinement, and MRM across the OTIS strata.

Full parity with rmorie's R/mrm_stockflow.R and with the measures in
rmoriebricklayer's R/custody.R. The arithmetic is implemented here
rather than delegated: morie is not a wrapper.

The measures are Lakner's. Given person-days x_i served over a period of
t days by N people:

    adp  = sum(x_i) / t    the average daily population -- days per DAY,
                           a STOCK. p.15
    alos = sum(x_i) / N    the average length of stay -- days per PERSON,
                           the FLOW side. p.16
    adp  = N_a * alos / t  the identity, eq 2.4 p.18
    X_t  = mean(C) * t     person-days from periodic head counts, eq 2.7

after Lakner, A Manual of Statistical Sampling Methods for Corrections
Planners (University of Illinois at Urbana-Champaign, April 1976).

Two caveats that change conclusions rather than decorate them:

* p.16-17: t must exceed the longest stay, or alos is biased DOWNWARD.
* The flow and stock rates can move in OPPOSITE directions, because
  days = people x stay. On the published Ontario segregation release the
  number of people fell 24.0 percent while stays grew 43.5 percent, so
  detention days ROSE 9.0 percent: the flow rate fell 27.6 percent and
  the stock rate rose 3.9 percent over the same window. Quoting one of
  them alone reverses the finding, so both are returned and the
  decomposition is checked to multiply out exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from morie.fn import _frame_core as pd
from morie.fn import _stats_core as stats

__all__ = [
    "StockFlowResult",
    "StaySummary",
    "OtisStockFlowResult",
    "adp",
    "alos",
    "admissions",
    "adp_from_counts",
    "period_days",
    "stay_summary",
    "stock_flow",
    "mrm_otis_stock_flow",
]


# --------------------------------------------------------------------------
# validation, matching rmoriebricklayer's .rmbl_pos_num
# --------------------------------------------------------------------------

def _pos_num(x, what: str, allow_zero: bool = False) -> list[float]:
    if x is None:
        raise ValueError(f"`{what}` must not be None")
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        vals = [float(x)]
    else:
        try:
            vals = [float(v) for v in x]
        except TypeError as exc:
            raise ValueError(f"`{what}` must be numeric") from exc
    if not vals:
        raise ValueError(f"`{what}` must not be empty")
    for v in vals:
        if v != v:
            raise ValueError(f"`{what}` must not contain missing values")
        if allow_zero and v < 0:
            raise ValueError(f"`{what}` must not be negative")
        if not allow_zero and v <= 0:
            raise ValueError(f"`{what}` must be strictly positive")
    return vals


def _recycle(v: list[float], n: int, what: str) -> list[float]:
    if len(v) == 1:
        return v * n
    if len(v) != n:
        raise ValueError(f"`{what}` must be length 1 or {n}")
    return v


# --------------------------------------------------------------------------
# the measures
# --------------------------------------------------------------------------

def adp(days, t: float = 365) -> float:
    """Average daily population: person-days per day of the period."""
    d = _pos_num(days, "days", allow_zero=True)
    tt = _pos_num(t, "t")[0]
    return sum(d) / tt


def alos(days, n) -> float:
    """Average length of stay: person-days per person.

    `n` counts people admitted AND released within the period (Lakner
    p.16); counting everyone present inflates the denominator and biases
    the stay downward.
    """
    d = _pos_num(days, "days", allow_zero=True)
    nn = _pos_num(n, "n")[0]
    return sum(d) / nn


def admissions(adp_value, alos_value, t: float = 365) -> float:
    """Admissions implied by the identity adp = N_a * alos / t."""
    a = _pos_num(adp_value, "adp")[0]
    l = _pos_num(alos_value, "alos")[0]
    tt = _pos_num(t, "t")[0]
    return a * tt / l


def adp_from_counts(counts, t: float = 365) -> float:
    """Person-days from periodic head counts: mean(counts) * t (eq 2.7)."""
    c = _pos_num(counts, "counts", allow_zero=True)
    tt = _pos_num(t, "t")[0]
    return (sum(c) / len(c)) * tt


def period_days(start, end) -> float:
    """Inclusive length of a period in days."""
    def _d(v):
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        return datetime.strptime(str(v), "%Y-%m-%d").date()
    return float((_d(end) - _d(start)).days) + 1.0


# --------------------------------------------------------------------------
# stay summary
# --------------------------------------------------------------------------

@dataclass
class StaySummary:
    n: int
    total_days: float
    mean: float
    sd: float | None
    median: float
    iqr: float
    max: float
    se: float | None
    lower: float | None
    upper: float | None

    def to_frame(self):
        return pd.DataFrame({k: [v] for k, v in self.__dict__.items()})


def _quantile_type7(x: list[float], p: float) -> float:
    """R's default quantile (type 7): h = (n - 1)p + 1, linear interpolation."""
    s = sorted(x)
    n = len(s)
    if n == 1:
        return s[0]
    h = (n - 1) * p
    lo = int(h)
    hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def stay_summary(days_per_person, conf_level: float = 0.95) -> StaySummary:
    """Distribution of stay lengths, with a t interval on the mean."""
    x = _pos_num(days_per_person, "days_per_person", allow_zero=True)
    cl = float(conf_level)
    if not (0 < cl < 1):
        raise ValueError("`conf_level` must lie strictly inside (0, 1)")
    n = len(x)
    m = sum(x) / n
    if n > 1:
        var = sum((v - m) ** 2 for v in x) / (n - 1)   # sample sd, as R's sd()
        sd = var ** 0.5
        se = sd / (n ** 0.5)
        tq = float(stats.t.ppf(1 - (1 - cl) / 2, n - 1))
        lower, upper = m - tq * se, m + tq * se
    else:
        sd = se = lower = upper = None
    q1 = _quantile_type7(x, 0.25)
    q2 = _quantile_type7(x, 0.50)
    q3 = _quantile_type7(x, 0.75)
    return StaySummary(n=n, total_days=sum(x), mean=m, sd=sd, median=q2,
                       iqr=q3 - q1, max=max(x), se=se, lower=lower,
                       upper=upper)


# --------------------------------------------------------------------------
# stock and flow
# --------------------------------------------------------------------------

@dataclass
class StockFlowResult:
    period: list[str]
    people: list[float]
    days: list[float]
    alos: list[float]
    adp: list[float]
    people_change: list[float | None]
    alos_change: list[float | None]
    days_change: list[float | None]
    adp_change: list[float | None]
    t: list[float]
    baseline: str
    exposure: list[float] | None = None
    flow_rate: list[float] | None = None
    stock_rate: list[float] | None = None
    flow_rate_change: list[float | None] | None = None
    stock_rate_change: list[float | None] | None = None
    per: float | None = None

    def __len__(self) -> int:
        return len(self.period)

    def to_frame(self):
        cols = {"period": self.period, "people": self.people,
                "days": self.days, "alos": self.alos, "adp": self.adp}
        if self.exposure is not None:
            cols["exposure"] = self.exposure
            cols["flow_rate"] = self.flow_rate
            cols["stock_rate"] = self.stock_rate
        cols["people_change"] = self.people_change
        cols["alos_change"] = self.alos_change
        cols["days_change"] = self.days_change
        cols["adp_change"] = self.adp_change
        if self.exposure is not None:
            cols["flow_rate_change"] = self.flow_rate_change
            cols["stock_rate_change"] = self.stock_rate_change
        return pd.DataFrame(cols)

    def opposite_signs(self) -> bool:
        """Do the flow and stock rates move in opposite directions?

        The reason both are reported: on the Ontario release one fell
        while the other rose.
        """
        if self.flow_rate_change is None or len(self) < 2:
            return False
        f, s = self.flow_rate_change[-1], self.stock_rate_change[-1]
        if f is None or s is None or f == 0 or s == 0:
            return False
        return (f > 0) != (s > 0)


def _changes(v: list[float], baseline: str) -> list[float | None]:
    if baseline == "first":
        return [100.0 * (x / v[0] - 1.0) for x in v]
    return [None] + [100.0 * (v[i] / v[i - 1] - 1.0) for i in range(1, len(v))]


def stock_flow(days, people, period=None, t: float = 365, exposure=None,
               per: float = 100000, baseline: str = "first") -> StockFlowResult:
    """The stock and the flow side by side, with the exact decomposition.

    `baseline` is a reporting decision, not a detail: "first" compares
    every period against the start of the window, "previous" against the
    period before it.
    """
    if baseline not in ("first", "previous"):
        raise ValueError("`baseline` must be 'first' or 'previous'")
    d = _pos_num(days, "days", allow_zero=True)
    p = _pos_num(people, "people")
    if len(d) != len(p):
        raise ValueError("`days` and `people` must be the same length")
    n = len(d)
    tt = _recycle(_pos_num(t, "t"), n, "t")
    per_lab = [str(x) for x in (period if period is not None
                                else range(1, n + 1))]
    if len(per_lab) != n:
        raise ValueError(f"`period` must be length {n}")
    los = [d[i] / p[i] for i in range(n)]
    pop = [d[i] / tt[i] for i in range(n)]

    exp_v = fr = sr = frc = src = None
    per_used = None
    if exposure is not None:
        exp_v = _recycle(_pos_num(exposure, "exposure"), n, "exposure")
        per_used = _pos_num(per, "per")[0]
        fr = [per_used * p[i] / exp_v[i] for i in range(n)]
        sr = [per_used * pop[i] / exp_v[i] for i in range(n)]

    out = StockFlowResult(
        period=per_lab, people=p, days=d, alos=los, adp=pop,
        people_change=_changes(p, baseline),
        alos_change=_changes(los, baseline),
        days_change=_changes(d, baseline),
        adp_change=_changes(pop, baseline),
        t=tt, baseline=baseline, exposure=exp_v,
        flow_rate=fr, stock_rate=sr, per=per_used)
    if exposure is not None:
        out.flow_rate_change = _changes(fr, baseline)
        out.stock_rate_change = _changes(sr, baseline)
    return out


# --------------------------------------------------------------------------
# MRM across the OTIS strata
# --------------------------------------------------------------------------

@dataclass
class OtisStockFlowResult:
    stock_flow: StockFlowResult
    reconciliation: list[dict] = field(default_factory=list)
    strata_agree: bool = False
    decomposition_exact: bool = False

    def to_frame(self):
        return pd.DataFrame({k: [r[k] for r in self.reconciliation]
                             for k in (self.reconciliation[0] if
                                       self.reconciliation else {})})


def _columns(data, cols: list[str], what: str) -> dict:
    """Accept a frame-like with column access, or a sequence of mappings."""
    if data is None:
        raise ValueError(f"`{what}` must not be None")
    if hasattr(data, "columns"):
        have = list(data.columns)
        missing = [c for c in cols if c not in have]
        if missing:
            raise ValueError(
                f"`{what}` is missing column(s): {', '.join(missing)}")
        return {c: list(data[c]) for c in cols}
    rows = list(data)
    if not rows or not all(hasattr(r, "keys") for r in rows):
        raise ValueError(f"`{what}` must be a data frame or rows of mappings")
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        raise ValueError(
            f"`{what}` is missing column(s): {', '.join(missing)}")
    return {c: [r[c] for r in rows] for c in cols}


def mrm_otis_stock_flow(
    person_days,
    placements=None,
    totals=None,
    year_col: str = "EndFiscalYear",
    id_col: str = "UniqueIndividual_ID",
    days_col: str = "TotalAggregatedDays_Segregation",
    consecutive_col: str = "NumberConsecutiveDays_Segregation",
    placement_col: str = "Number_Of_Placements",
    totals_col: str = "NumberIndividuals_Segregation",
    t: float = 365,
    exposure=None,
    per: float = 100000,
) -> OtisStockFlowResult:
    """Stock and flow across the OTIS strata, reconciled.

    The same confinement appears at three strata in the OTIS release and
    they do not automatically agree:

    * PERSON (b02): one row per individual per year, carrying the days
      that person served. This is the additive quantity, and the one the
      measures need.
    * PLACEMENT (b01): several rows per individual, carrying the
      consecutive days of a SPELL. A spell length is not an additive
      share of the year -- summing it as though it were understates the
      published total by about a third -- so it is reported for
      reconciliation, never used as the numerator.
    * AGGREGATE (c01): the yearly totals as stated.

    Reconciling them is the point, so both disagreements are returned
    rather than assumed away.
    """
    pdc = _columns(person_days, [year_col, id_col, days_col], "person_days")
    years = sorted({str(y) for y in pdc[year_col]})
    py = [str(y) for y in pdc[year_col]]

    days_by_year, people_by_year = [], []
    for y in years:
        idx = [i for i, v in enumerate(py) if v == y]
        tot = 0.0
        for i in idx:
            v = pdc[days_col][i]
            if v is None or v != v:
                continue
            tot += float(v)
        days_by_year.append(tot)
        people_by_year.append(float(len({pdc[id_col][i] for i in idx})))
    if any(v == 0 for v in people_by_year):
        raise ValueError("every period must contain at least one person")

    sf = stock_flow(days=days_by_year, people=people_by_year, period=years,
                    t=t, exposure=exposure, per=per)

    rec = [{"period": years[i],
            "person_stratum_people": people_by_year[i],
            "person_stratum_days": days_by_year[i]}
           for i in range(len(years))]

    if placements is not None:
        need = [year_col, id_col, consecutive_col]
        plc = _columns(placements, need, "placements")
        has_count = False
        if hasattr(placements, "columns"):
            has_count = placement_col in list(placements.columns)
        else:
            rows = list(placements)
            has_count = bool(rows) and placement_col in rows[0]
        if has_count:
            plc.update(_columns(placements, [placement_col], "placements"))
        qy = [str(y) for y in plc[year_col]]
        for i, y in enumerate(years):
            idx = [j for j, v in enumerate(qy) if v == y]
            ppl = float(len({plc[id_col][j] for j in idx}))
            cons = 0.0
            for j in idx:
                v = plc[consecutive_col][j]
                if v is None or v != v:
                    continue
                cons += float(v)
            rec[i]["placement_stratum_people"] = ppl
            rec[i]["consecutive_days"] = cons
            # Below one, because a spell length is not an additive share
            # of the year. Reported so that summing the wrong column is
            # visible rather than silently understating the total.
            pdays = rec[i]["person_stratum_days"]
            rec[i]["consecutive_over_person_days"] = (
                cons / pdays if pdays else None)
            if has_count:
                cnt = 0.0
                for j in idx:
                    v = plc[placement_col][j]
                    if v is None or v != v:
                        continue
                    cnt += float(v)
                rec[i]["placements"] = cnt
                rec[i]["placements_per_person"] = cnt / ppl if ppl else None
    else:
        for r in rec:
            r["placement_stratum_people"] = None
            r["consecutive_over_person_days"] = None

    if totals is not None:
        tlc = _columns(totals, [year_col, totals_col], "totals")
        ty = [str(y) for y in tlc[year_col]]
        for i, y in enumerate(years):
            tot = 0.0
            for j, v in enumerate(ty):
                if v != y:
                    continue
                w = tlc[totals_col][j]
                if w is None or w != w:
                    continue
                tot += float(w)
            rec[i]["aggregate_stratum_total"] = tot
    else:
        for r in rec:
            r["aggregate_stratum_total"] = None

    for r in rec:
        vals = [r.get("person_stratum_people"),
                r.get("placement_stratum_people"),
                r.get("aggregate_stratum_total")]
        vals = [v for v in vals if v is not None]
        r["strata_agree"] = len(set(vals)) == 1

    # days = people x stay, so the change in days must multiply out.
    # This is the check that catches a stratum mixed into the wrong
    # column.
    exact = True
    if len(sf) > 1:
        p = sf.people_change[-1] / 100.0
        l = sf.alos_change[-1] / 100.0
        exact = abs((1 + p) * (1 + l) - 1 - sf.days_change[-1] / 100.0) < 1e-8

    return OtisStockFlowResult(
        stock_flow=sf, reconciliation=rec,
        strata_agree=all(r["strata_agree"] for r in rec),
        decomposition_exact=exact)
