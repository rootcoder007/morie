"""trmet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trmet import trmet


def test_trmet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trmet()
