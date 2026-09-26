"""enrfl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enrfl import enrfl


def test_enrfl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enrfl()
