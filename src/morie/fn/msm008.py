# morie.fn -- function file (rootcoder007/morie)
"""Generalized sensitivity and specificity.

Implements eq. (4.10)-(4.11) p.132 of Montesinos López, Montesinos López & Crossa
(2022), *Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer (DOI 10.1007/978-3-030-89010-0).
"""

import math

from . import _gp_core as _gp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["mvsml_overfitting_resampling_eq_4_10"]


def mvsml_overfitting_resampling_eq_4_10(y_true, y_pred, class_index=0, n_classes=None):
    """Se_i = TTP_all/(TTP_all + TFN_i) (eq. 4.10) and
    Sp_i = TTN_i/(TTN_i + TFP_i) (eq. 4.11), with TFN_i from eq. (4.5)
    and TTN_i from eq. (4.7). Keys: estimate."""
    conf = _gp.confusion_counts(y_true, y_pred, n_classes)
    m = _gp.class_metrics(conf, int(class_index))
    res = RichResult(payload={"estimate": m["sensitivity"],
                              "sensitivity": m["sensitivity"],
                              "specificity": m["specificity"],
                              "TFN": m["TFN"], "TTN": m["TTN"],
                              "method": "generalized sensitivity/specificity (MVSML 2022 eq. 4.10-4.11)"})
    return with_describe_pointer(res, "msm008")


def cheatsheet():
    return "msm008: Generalized sensitivity and specificity"
