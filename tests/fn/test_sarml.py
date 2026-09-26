"""sarml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sarml import sarml


def test_sarml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sarml(y=None, X=None, W=None)
