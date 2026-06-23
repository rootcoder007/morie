"""Complex survey GLM with weights, clustering, and strata."""

from morie.survey import complex_survey_glm as _fn

svyglm = _fn
complex_survey_glm = _fn


def cheatsheet() -> str:
    return "svyglm() -> Complex survey GLM with weights, clustering, and strata."
