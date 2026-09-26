"""semhet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semhet import semhet


def test_semhet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        semhet(y=None, X=None, W=None)
