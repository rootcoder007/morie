"""semml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semml import semml


def test_semml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        semml(y=None, X=None, W=None)
