# SPDX-License-Identifier: AGPL-3.0-or-later
"""The research programme: methods for the hardest problems in criminology, each resting on a Lean 4 theorem.

Every function names the theorems in ``research/lean`` (Lean 4 + Mathlib, 0 sorry,
standard axioms only) that cover its arithmetic, and says where the empirical
assumption enters. Lean certifies the implication, never the antecedent. The R
package carries the same functions with the ``morie_`` prefix; both arms agree
to rounding on the same inputs, and the stochastic ones share the Philox stream.
"""

from morie.research.cheeger import cheeger_bound
from morie.research.concentration import (
    concentration_decompose,
    concentration_dispersion,
    concentration_distinct_growth,
    concentration_gini,
)
from morie.research.contagion import contagion_branching
from morie.research.dark_figure import (
    dark_figure_bounds,
    dark_figure_breakdown,
    dark_figure_hierarchy,
    dark_figure_three_list,
    dark_figure_two_source,
)
from morie.research.ecological import ecological_bounds, ecological_decompose
from morie.research.fairness_bounds import (
    fairness_base_rate_bounds,
    fairness_compare_groups,
    fairness_implied_fpr,
    fairness_rates,
    fairness_true_rate,
    hazard_selection,
    logit_rescale,
    ranking_resolution,
)
from morie.research.feedback_loop import (
    feedback_loop_bound,
    feedback_loop_limit,
    feedback_loop_meanfield,
    feedback_loop_sim,
    feedback_loop_urn_law,
)
from morie.research.logit_separation import logit_separation
from morie.research.meta_pooling import meta_dl_bias, meta_random_effects
from morie.research.recording_map import detection_rate_shift, recording_map
from morie.research.selection import (
    age_crime_aggregate,
    collider_arrest,
    deterrence_design_check,
    deterrence_response,
    disparity_benchmark,
    disparity_exposure_bounds,
    interracial_rates,
    probability_of_necessity,
    relative_risk_from_or,
)
from morie.research.sentence_bounds import (
    bounds_confidence,
    contaminated_bounds,
    sentence_effect_bounds,
    sentence_effect_mtr,
    sentence_effect_mts,
)
from morie.research.spillover import (
    spillover_effects,
    spillover_exposure,
    spillover_exposure_probs,
    spillover_ht,
    spillover_ht_variance,
)

__all__ = [
    "age_crime_aggregate", "bounds_confidence", "cheeger_bound", "collider_arrest", "concentration_decompose",
    "concentration_dispersion", "concentration_distinct_growth", "concentration_gini", "contagion_branching",
    "contaminated_bounds", "dark_figure_bounds", "dark_figure_breakdown", "dark_figure_hierarchy",
    "dark_figure_three_list", "dark_figure_two_source", "detection_rate_shift", "deterrence_design_check",
    "deterrence_response", "disparity_benchmark", "disparity_exposure_bounds", "ecological_bounds", "ecological_decompose",
    "fairness_base_rate_bounds", "fairness_compare_groups", "fairness_implied_fpr", "fairness_rates",
    "fairness_true_rate", "feedback_loop_bound", "feedback_loop_limit", "feedback_loop_meanfield",
    "feedback_loop_sim", "feedback_loop_urn_law", "hazard_selection", "interracial_rates", "logit_rescale",
    "logit_separation", "meta_dl_bias", "meta_random_effects", "probability_of_necessity", "ranking_resolution",
    "recording_map", "relative_risk_from_or", "sentence_effect_bounds", "sentence_effect_mtr", "sentence_effect_mts",
    "spillover_effects", "spillover_exposure", "spillover_exposure_probs", "spillover_ht",
    "spillover_ht_variance",
]
