"""cfafit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cfafit import cfa_fit_indices


def test_cfafit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cfa_fit_indices(fit=None)
