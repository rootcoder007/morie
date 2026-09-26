"""gwrcomp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gwrcomp import gwrcomp


def test_gwrcomp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwrcomp(aicc_fixed=None, aicc_adapt=None)
