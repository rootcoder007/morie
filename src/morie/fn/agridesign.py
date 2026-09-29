# morie.fn -- function file (rootcoder007/morie)
"""Agricultural and landscape design formulas: vegetated buffer width by first-order trapping,
Kadlec-Knight P-k-C* constructed-wetland sizing, Dyksterhuis range condition, crop-rotation
screening and scoring, variable-rate management zones with Stanford mass-balance rates, and
Hooghoudt drain spacing and mid-drain water-table height."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "buffer_strip_width",
    "wetland_area_pkc",
    "range_condition",
    "rotation_score",
    "variable_rate_zones",
    "hooghoudt_spacing",
    "drain_water_table",
]


def buffer_strip_width(target: float, *, k: float | None = None, widths=None, efficiencies=None) -> RichResult:
    r"""Vegetated buffer (filter-strip, riparian) width for a target removal under first-order trapping.

    Removal efficiency ``E(w) = 1 - exp(-k w)``, so the width achieving
    ``target`` is ``w = -ln(1 - target) / k``. When ``k`` is not given it is
    estimated from observed ``(widths, efficiencies)`` by least squares
    through the origin on ``-ln(1 - E) = k w``.

    References
    ----------
    Zhang, X., Liu, X., Zhang, M., Dahlgren, R. A. and Eitzel, M. (2010). A
    review of vegetated buffers and a meta-analysis of their mitigation
    efficacy in reducing nonpoint source pollution. J. Environ. Qual. 39,
    76-84. Dillaha, T. A. et al. (1989). Vegetative filter strips for
    agricultural nonpoint source pollution control. Trans. ASAE 32, 513-519.

    Examples
    --------
    >>> r = buffer_strip_width(0.9, k=0.1)
    >>> round(r.width, 12)
    23.02585092994
    """
    if k is None:
        y = [-math.log(1.0 - float(e)) for e in efficiencies]
        w = [float(v) for v in widths]
        k = ssum(a * b for a, b in zip(w, y)) / ssum(a * a for a in w)
    return RichResult(payload={"width": -math.log(1.0 - target) / k, "k": k})


def wetland_area_pkc(
    flow: float, c_in: float, c_out: float, k: float, c_star: float = 0.0, *, n_tanks: float | None = None
) -> RichResult:
    r"""Constructed-wetland area by the Kadlec-Knight P-k-C* model.

    Plug flow (Kadlec and Knight 1996): ``A = (365 Q / k) ln((C_i - C*) / (C_o - C*))``
    (m^2; ``Q`` in m^3/d, areal rate ``k`` in m/yr). Tanks in series
    (Kadlec and Wallace 2009): ``(C_o - C*) / (C_i - C*) = (1 + k / (P q))^(-P)``
    with hydraulic loading ``q = 365 Q / A``, solved for ``A``. Also reports
    hectares and the hydraulic loading rate.

    References
    ----------
    Kadlec, R. H. and Knight, R. L. (1996). Treatment Wetlands. CRC Press.
    Kadlec, R. H. and Wallace, S. D. (2009). Treatment Wetlands, 2nd ed., ch. 6.

    Examples
    --------
    >>> r = wetland_area_pkc(1000.0, 100.0, 20.0, 34.0, 5.0)
    >>> round(r.area_ha, 10)
    1.9815492413
    """
    ratio = (c_out - c_star) / (c_in - c_star)
    if n_tanks is None:
        area = 365.0 * flow / k * math.log(1.0 / ratio)
    else:
        q = k / (n_tanks * (ratio ** (-1.0 / n_tanks) - 1.0))
        area = 365.0 * flow / q
    return RichResult(payload={"area": area, "area_ha": area / 1e4, "hydraulic_loading": 365.0 * flow / area})


def range_condition(current, climax) -> RichResult:
    r"""Dyksterhuis (1949) range condition: percent similarity to the climax (reference) community.

    ``score = sum_i min(current_i, climax_i)`` over species, both given as
    percent composition (e.g. by weight); classes excellent (76-100), good
    (51-75), fair (26-50) and poor (0-25).

    References
    ----------
    Dyksterhuis, E. J. (1949). Condition and management of range land based
    on quantitative ecology. J. Range Management 2, 104-115.

    Examples
    --------
    >>> r = range_condition([40.0, 30.0, 30.0], [60.0, 40.0, 0.0])
    >>> r.score, r.condition
    (70.0, 'good')
    """
    score = ssum(min(float(a), float(b)) for a, b in zip(current, climax))
    cls = "excellent" if score > 75 else ("good" if score > 50 else ("fair" if score > 25 else "poor"))
    return RichResult(payload={"score": score, "condition": cls})


def rotation_score(sequence, effect, *, min_return=None, max_frequency=None, forbidden=()) -> RichResult:
    r"""Screen and score a cyclic crop rotation (ROTAT criteria of Dogliotti et al. 2003).

    ``sequence`` lists crop indices of one rotation cycle; ``effect[a][b]``
    is the pre-crop effect of crop ``a`` on the following crop ``b``. The
    score is ``sum_t effect[c_t][c_{t+1}]`` over the cyclic sequence. A
    rotation is feasible when no crop recurs sooner than ``min_return[c]``
    years (cyclic gaps), no crop exceeds ``max_frequency[c]`` (share of
    years) and no forbidden pair ``(a, b)`` follows directly.

    References
    ----------
    Dogliotti, S., Rossing, W. A. H. and van Ittersum, M. K. (2003). ROTAT, a
    tool for systematically generating crop rotations. Eur. J. Agron. 19, 239-250.

    Examples
    --------
    >>> E = [[0.0, 1.0], [2.0, -1.0]]
    >>> r = rotation_score([0, 1, 1], E, min_return=[2, 1])
    >>> r.score, r.feasible
    (2.0, True)
    """
    n = len(sequence)
    score = ssum(float(effect[sequence[t]][sequence[(t + 1) % n]]) for t in range(n))
    violations = []
    for c in sorted(set(sequence)):
        pos = [t for t in range(n) if sequence[t] == c]
        gaps = [(pos[(i + 1) % len(pos)] - pos[i]) % n or n for i in range(len(pos))]
        if min_return is not None and min(gaps) < min_return[c]:
            violations.append(("return", c))
        if max_frequency is not None and len(pos) / n > max_frequency[c]:
            violations.append(("frequency", c))
    for t in range(n):
        if (sequence[t], sequence[(t + 1) % n]) in set(map(tuple, forbidden)):
            violations.append(("sequence", t))
    return RichResult(payload={"score": score, "feasible": not violations, "violations": violations})


def _q7(s, p):
    h = (len(s) - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, len(s) - 1)
    w = h - lo
    return (1.0 - w) * s[lo] + w * s[hi] if w > 0 else s[lo]


def variable_rate_zones(
    yield_map, n_zones: int = 3, *, n_per_yield: float = 20.0, soil_n=None, efficiency: float = 1.0
) -> RichResult:
    r"""Management zones from a yield map and per-zone Stanford mass-balance nitrogen rates.

    Cells are classed into ``n_zones`` by the type 7 quantiles of the yield
    map (zone ``z`` holds values above the ``z / n_zones`` quantile). Each
    zone's yield goal is its mean yield ``Y_z``; the rate is
    ``max((n_per_yield Y_z - N_soil,z) / efficiency, 0)`` (Stanford 1973),
    with ``N_soil,z`` the zone mean of ``soil_n`` (0 if not given).

    References
    ----------
    Stanford, G. (1973). Rationale for optimum nitrogen fertilization in corn
    production. J. Environ. Qual. 2, 159-166. Khosla, R. et al. (2002). Use
    of site-specific management zones to improve nitrogen management for
    precision agriculture. J. Soil Water Conserv. 57, 513-518.

    Examples
    --------
    >>> r = variable_rate_zones([4.0, 6.0, 8.0, 10.0], 2, n_per_yield=20.0)
    >>> r.zone, r.rate
    ([0, 0, 1, 1], [100.0, 180.0])
    """
    y = [float(v) for v in yield_map]
    s = sorted(y)
    breaks = [_q7(s, z / n_zones) for z in range(1, n_zones)]
    zone = [sum(1 for b in breaks if v > b) for v in y]
    sn = [0.0] * len(y) if soil_n is None else [float(v) for v in soil_n]
    means, rates = [], []
    for z in range(n_zones):
        idx = [i for i in range(len(y)) if zone[i] == z]
        if not idx:
            means.append(float("nan"))
            rates.append(float("nan"))
            continue
        m = ssum(y[i] for i in idx) / len(idx)
        ns = ssum(sn[i] for i in idx) / len(idx)
        means.append(m)
        rates.append(max((n_per_yield * m - ns) / efficiency, 0.0))
    return RichResult(payload={"zone": zone, "breaks": breaks, "zone_mean": means, "rate": rates})


def _equivalent_depth(D, L, r):
    x = D / L
    if x <= 0.3:
        return D / (1.0 + x * (8.0 / math.pi * math.log(D / r) - 3.4))
    return math.pi * L / (8.0 * (math.log(L / r) - 1.15))


def hooghoudt_spacing(
    q: float,
    h: float,
    k_above: float,
    k_below: float,
    depth_to_barrier: float,
    *,
    radius: float = 0.1,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> RichResult:
    r"""Drain spacing by the Hooghoudt equation with Moody's equivalent depth.

    ``q L^2 = 8 K_b d h + 4 K_a h^2`` (steady drainage rate ``q``, mid-drain
    head ``h`` above drain level, conductivities above/below drain level,
    ``d`` the equivalent depth). ``d`` follows Moody (1966):
    ``D / (1 + (D/L)(8/pi ln(D/r) - 3.4))`` for ``D/L <= 0.3`` else
    ``pi L / (8 (ln(L/r) - 1.15))``; ``L`` is found by fixed-point iteration
    from ``d = D``.

    References
    ----------
    Hooghoudt, S. B. (1940). Bijdragen tot de kennis van eenige natuurkundige
    grootheden van den grond, 7. Verslagen Landbouwkundig Onderzoek 46.
    Ritzema, H. P. (ed.) (1994). Drainage Principles and Applications. ILRI
    Publication 16, ch. 8. Moody, W. T. (1966). J. Irrig. Drain. Div. ASCE 92, 1-9.

    Examples
    --------
    >>> r = hooghoudt_spacing(0.007, 0.8, 0.5, 0.5, 5.0)
    >>> round(r.spacing, 8), round(r.equivalent_depth, 8)
    (37.4281598, 2.66439688)
    """
    d = depth_to_barrier
    L = math.sqrt((8.0 * k_below * d * h + 4.0 * k_above * h * h) / q)
    for _ in range(max_iter):
        d = _equivalent_depth(depth_to_barrier, L, radius)
        new = math.sqrt((8.0 * k_below * d * h + 4.0 * k_above * h * h) / q)
        if abs(new - L) <= tol * new:
            L = new
            break
        L = new
    return RichResult(payload={"spacing": L, "equivalent_depth": _equivalent_depth(depth_to_barrier, L, radius)})


def drain_water_table(q: float, spacing: float, k_above: float, k_below: float, equivalent_depth: float) -> float:
    r"""Mid-drain water-table height above drain level from the Hooghoudt equation.

    The positive root of ``4 K_a h^2 + 8 K_b d h - q L^2 = 0``.

    Examples
    --------
    >>> round(drain_water_table(0.007, 30.0, 0.5, 0.5, 2.0), 12)
    0.673948391424
    """
    a, b, c = 4.0 * k_above, 8.0 * k_below * equivalent_depth, -q * spacing * spacing
    if a == 0:
        return -c / b
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


def cheatsheet() -> str:
    return (
        "buffer_strip_width / wetland_area_pkc / range_condition / rotation_score / variable_rate_zones / "
        "hooghoudt_spacing / drain_water_table -> agricultural and landscape design formulas."
    )
