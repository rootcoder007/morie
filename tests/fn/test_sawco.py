"""sawco is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawco import sawco


def test_sawco_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawco()
