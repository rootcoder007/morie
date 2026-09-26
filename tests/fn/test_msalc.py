"""msalc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msalc import alienation


def test_msalc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        alienation(data=None)
