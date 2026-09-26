"""aitmad is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aitmad import compositional_mad


def test_aitmad_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        compositional_mad(X=None)
