"""Parity dump for ksr01..ksr20: prints each estimate for comparison with
the rmorie arm (no assertions; collected so that it must at least run)."""

import importlib

from morie.fn import _array_core as np
from morie.fn import _stats_core as stats

mods = {f"ksr{i:02d}": importlib.import_module(f"morie.fn.ksr{i:02d}") for i in range(1, 21)}


def fmt(d, keys):
    parts = []
    for k in keys:
        v = d.get(k)
        if v is None or isinstance(v, str):
            parts.append(f"{k}=NA")
        else:
            try:
                parts.append(f"{k}={float(v):.10g}")
            except Exception:
                parts.append(f"{k}=NA")
    return " | ".join(parts)


xs = np.array(
    [0.1, 0.4, -0.3, 0.7, 0.05, -0.9, 1.2, -0.4, 0.6, -0.1, 0.3, -0.2, 0.5, -0.7, 0.0, 0.2, -0.1, 0.4, -0.5, 0.8]
)
ys = 1.5 * xs + np.array(
    [0.2, -0.1, 0.05, 0.3, -0.2, 0.1, -0.3, 0.0, 0.1, -0.05, -0.1, 0.0, 0.2, -0.2, 0.1, 0.05, -0.1, 0.2, -0.3, 0.1]
)
ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
ev = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
X3 = np.zeros((100, 3))

print("ksr01:", fmt(mods["ksr01"].kosorok_empirical_process(xs, [0.0], [0.5]), ["sup_abs"]))
print("ksr02:", fmt(mods["ksr02"].kosorok_donsker_class(xs), ["estimate"]))
print(
    "ksr03:", fmt(mods["ksr03"].kosorok_glivenko_cantelli(xs, stats.norm.cdf(np.sort(xs))), ["statistic", "dkw_bound"])
)
print("ksr04:", fmt(mods["ksr04"].kosorok_vc_dimension(X3), ["estimate"]))
print("ksr05:", fmt(mods["ksr05"].kosorok_bracketing_number(xs, e=0.1), ["estimate"]))
print("ksr06:", fmt(mods["ksr06"].kosorok_maximal_inequality(xs), ["estimate"]))
print("ksr07_se_approx:", fmt(mods["ksr07"].kosorok_bootstrap_empirical(xs, B=2000, seed=42), ["se"]))
print("ksr08_se_approx:", fmt(mods["ksr08"].kosorok_multiplier_bootstrap(xs, B=2000, seed=42), ["se"]))
print("ksr09:", fmt(mods["ksr09"].kosorok_z_estimator(xs), ["estimate", "se"]))
print("ksr10:", fmt(mods["ksr10"].kosorok_m_estimator(xs), ["estimate", "se"]))
print("ksr11:", fmt(mods["ksr11"].kosorok_efficient_score(xs, ys), ["estimate", "se"]))
print("ksr12:", fmt(mods["ksr12"].kosorok_information_bound(xs, ys), ["estimate"]))
print("ksr13:", fmt(mods["ksr13"].kosorok_tangent_space(xs), ["estimate"]))
print("ksr14:", fmt(mods["ksr14"].kosorok_profile_likelihood(xs, ys), ["estimate", "se"]))
print("ksr15:", fmt(mods["ksr15"].kosorok_one_step_estimator(xs, 0.0), ["estimate", "se"]))
print("ksr16:", fmt(mods["ksr16"].kosorok_influence_function(xs, ys), ["estimate"]))
print("ksr17:", fmt(mods["ksr17"].kosorok_counting_process(ts, ev), ["estimate"]))
print("ksr18:", fmt(mods["ksr18"].kosorok_nelson_aalen(ts, ev), ["estimate", "se"]))
xs_cox = xs[:10]
print("ksr19:", fmt(mods["ksr19"].kosorok_cox_partial_likelihood(xs_cox, ts, ev), ["estimate", "se"]))
print("ksr20:", fmt(mods["ksr20"].kosorok_censoring_survival(ts, ev), ["estimate", "se"]))
