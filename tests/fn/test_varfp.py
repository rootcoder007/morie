"""varfp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.varfp import varfp


def test_varfp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        varfp()
