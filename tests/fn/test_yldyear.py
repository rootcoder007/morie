"""yldyear is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.yldyear import yld_calculation


def test_yldyear_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        yld_calculation(prevalence=None, disability=None, duration=None)
