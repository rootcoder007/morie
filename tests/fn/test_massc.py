"""massc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.massc import massc


def test_massc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        massc()
