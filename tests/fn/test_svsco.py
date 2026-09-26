"""svsco is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svsco import svsco


def test_svsco_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svsco()
